"""Decide what each robot's map shows, fetching the floor file only when it changes.

The shadow's ``fileList`` carries the time each property file was last written,
so noticing a new floor file costs no extra requests. While a robot is out the
previous path is hidden and the rooms being cleaned are marked; when the run
ends, polling stays fast until Shark has written the new path.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .floor_map import (
    AREA_ROOM,
    FloorPlan,
    parse_floor_file,
    path_share_in_area,
    render_floor_svg,
    rooms_on_path,
)
from .floor_style import Caption, room_list, when
from .shark_device import SharkVacuum

logger = logging.getLogger(__name__)

FLOOR_FILE = re.compile(r"^Persistent_Floor_\d+$")
WAIT_FOR_PATH_SECONDS = 180   # how long to keep polling fast after a run
MAX_ATTEMPTS = 3              # tries per version of a floor file
AREA_NAME = "a marked area"   # what a spot or marked-area clean is called
AREA_SHARE = 0.5              # this much of a path in the area marks a run not seen from here

Version = tuple[str, str]     # (property name, time Shark wrote it)


@dataclass
class _Robot:
    plan: FloorPlan | None = None
    loaded: Version | None = None
    tried: Version | None = None
    attempts: int = 0
    out: bool = False
    before_run: Version | None = None     # the file as it was when the run began
    rooms: list[str] = field(default_factory=list)
    wait_until: float | None = None
    run_rooms: list[str] = field(default_factory=list)
    run_version: Version | None = None    # the file that holds the last run seen here
    started: datetime | None = None       # when this run was first seen, if seen starting
    seen_docked: bool = False
    area: list[tuple[float, float]] = field(default_factory=list)       # marked area of this run
    run_area: list[tuple[float, float]] = field(default_factory=list)   # ... of the last one


def newest_floor_file(reported: dict[str, Any]) -> Version | None:
    """The most recently written floor file listed in a device shadow."""
    listed = reported.get("fileList")
    if isinstance(listed, dict) and isinstance(listed.get("value"), dict):
        listed = listed["value"]
    if not isinstance(listed, dict):
        return None
    found = []
    for name, entry in listed.items():
        if not (FLOOR_FILE.match(name) and isinstance(entry, dict)):
            continue
        value = entry.get("value")
        if isinstance(value, dict) and value.get("fileName"):
            found.append((str(entry.get("updatedAt") or ""), name))
    if not found:
        return None
    written, name = max(found)
    return name, written


def _jobs(reported: dict[str, Any]) -> list[tuple[str, str]]:
    """(mode, room) for each entry of the current job."""
    for key in ("AreasToClean_V3", "AreasToClean_V2"):
        entry = reported.get(key)
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if not isinstance(value, dict):
            continue
        # V2: {"areas_to_clean": ["UserRoom:Kitchen"]}
        # V3: {"areas_to_clean": {"UserRoom": ["Kitchen"]}}
        areas = value.get("areas_to_clean")
        jobs: list[tuple[str, str]] = []
        if isinstance(areas, dict):
            jobs = [(str(mode), a) for mode, group in areas.items() if isinstance(group, list)
                    for a in group if isinstance(a, str)]
        elif isinstance(areas, list):
            for a in areas:
                if isinstance(a, str):
                    mode, colon, room = a.partition(":")
                    jobs.append((mode, room) if colon else ("", a))
        jobs = [(mode, room) for mode, room in jobs if room not in ("", "*")]
        if jobs:
            return jobs
    return []


def areas_being_cleaned(reported: dict[str, Any]) -> list[str]:
    """Room names in the current job; empty when idle or cleaning everywhere."""
    names = [AREA_NAME if mode == AREA_ROOM else room for mode, room in _jobs(reported)]
    return list(dict.fromkeys(names))


def area_being_cleaned(reported: dict[str, Any]) -> list[tuple[float, float]]:
    """Corners (map metres) of the marked area in the current job, if it is one."""
    for mode, room in _jobs(reported):
        if mode == AREA_ROOM:
            try:
                numbers = [float(v) for v in room.split(",")]
            except ValueError:
                return []
            if len(numbers) >= 6 and len(numbers) % 2 == 0:
                return list(zip(numbers[::2], numbers[1::2]))
    return []


def friendly_time(written: str) -> str:
    """Shark's UTC timestamp in the container's time zone (set TZ), e.g. "Wed, Nov 25 · 8:40 AM"."""
    try:
        return when(datetime.fromisoformat(written.replace("Z", "+00:00")).astimezone())
    except ValueError:
        return ""


def clock_time(moment: datetime) -> str:
    return when(moment).rsplit(" · ", 1)[1]


class FloorMapTracker:
    """One per process; remembers each robot's floor file and run."""

    def __init__(self, api: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._api = api
        self._clock = clock
        self._robots: dict[str, _Robot] = {}

    def waiting(self, dsn: str) -> bool:
        """Whether a finished run's path is still expected (keep polling fast)."""
        robot = self._robots.get(dsn)
        return bool(robot and robot.wait_until is not None and self._clock() < robot.wait_until)

    def plan_for(self, dsn: str) -> FloorPlan | None:
        """The floor plan last read for a robot."""
        robot = self._robots.get(dsn)
        return robot.plan if robot else None

    async def update(self, device: SharkVacuum, raw: dict[str, Any]) -> str | None:
        """The SVG to show for this poll, or None if there is no floor file to draw."""
        reported = ((raw.get("shadow") or {}).get("properties") or {}).get("reported") or {}
        version = newest_floor_file(reported)
        if version is None:
            return None
        robot = self._robots.setdefault(device.dsn, _Robot())
        out = device.ha_state != "docked" or bool(device.job_active)
        if out and not robot.out:
            robot.before_run, robot.rooms, robot.wait_until = version, [], None
            robot.area = []
            robot.started = datetime.now().astimezone() if robot.seen_docked else None
        if not out:
            robot.seen_docked = True

        await self._load(device, robot, version)

        if out:
            for name in areas_being_cleaned(reported):
                if name not in robot.rooms:
                    robot.rooms.append(name)
            robot.area = area_being_cleaned(reported) or robot.area
        elif robot.out:  # the run just ended
            robot.run_rooms, robot.run_version = robot.rooms, None
            robot.run_area = robot.area
            robot.wait_until = self._clock() + WAIT_FOR_PATH_SECONDS
        robot.out = out

        if robot.wait_until is not None:
            if robot.loaded != robot.before_run and robot.loaded == version:
                robot.run_version, robot.wait_until = robot.loaded, None
            elif self._clock() >= robot.wait_until:
                robot.wait_until = None
        if robot.plan is None:
            return None
        return self._draw(device, robot)

    async def _load(self, device: SharkVacuum, robot: _Robot, version: Version) -> None:
        if version == robot.loaded:
            return
        if version != robot.tried:
            robot.tried, robot.attempts = version, 0
        if robot.attempts >= MAX_ATTEMPTS:
            return
        robot.attempts += 1
        body = await self._api.fetch_property_file(device.dsn, version[0])
        plan = parse_floor_file(body) if body else None
        if plan is None:
            logger.debug("No readable %s for %s yet", version[0], device.product_name)
            return
        robot.plan, robot.loaded = plan, version
        logger.info(
            "Loaded %s for %s: %d rooms, %d path points",
            version[0], device.product_name, len(plan.rooms), len(plan.path),
        )

    def _draw(self, device: SharkVacuum, robot: _Robot) -> str:
        assert robot.plan is not None and robot.loaded is not None
        names = device.room_name_map or {}

        def listed(rooms: list[str]) -> str:
            return room_list(names.get(r, r) for r in rooms)

        if robot.out:
            state = {"returning": "returning", "paused": "paused", "idle": "paused",
                     "error": "error"}.get(device.ha_state, "cleaning")
            title = {
                "returning": "Returning to the dock",
                "paused": "Paused",
                "error": "Stopped with an error",
            }.get(device.ha_state)
            if device.ha_state == "idle":
                title = "Stopped away from the dock"
            if title is None:
                title = f"Cleaning {listed(robot.rooms)}" if robot.rooms else "Cleaning"
            # a path is shown only if it was written during this run
            new_path = robot.loaded != robot.before_run
            detail = []
            if state != "cleaning" and robot.rooms:
                detail.append(listed(robot.rooms))
            if robot.started:
                detail.append(f"Started {clock_time(robot.started)}")
            if not new_path:
                detail.append("Path appears after docking")
            return render_floor_svg(
                robot.plan, Caption(state, title, " · ".join(detail)), show_path=new_path,
                cleaning=robot.rooms, name_map=names, area=robot.area,
            )
        if robot.wait_until is not None:
            return render_floor_svg(
                robot.plan, Caption("waiting", "Docked", "Waiting for the cleaning path"),
                show_path=False, cleaned=robot.run_rooms, name_map=names, area=robot.run_area,
            )
        seen = robot.run_version == robot.loaded and bool(robot.run_rooms)
        area: list[tuple[float, float]] = []
        if seen and AREA_NAME in robot.run_rooms:
            area = robot.run_area or robot.plan.area
        elif not seen and path_share_in_area(robot.plan) >= AREA_SHARE:
            area = robot.plan.area
        if seen:
            rooms = robot.run_rooms
        else:
            rooms = [AREA_NAME] if area else rooms_on_path(robot.plan, names)
        detail = [friendly_time(robot.loaded[1])]
        if robot.plan.minutes:
            detail.append(f"{robot.plan.minutes} min")
        title = f"Cleaned {listed(rooms)}" if rooms else "Last clean"
        return render_floor_svg(
            robot.plan, Caption("docked", title, " · ".join(d for d in detail if d)),
            cleaned=rooms, name_map=names, area=area,
        )
