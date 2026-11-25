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

from .floor_map import FloorPlan, parse_floor_file, render_floor_svg, rooms_on_path
from .floor_style import Caption, room_list, when
from .shark_device import SharkVacuum

logger = logging.getLogger(__name__)

FLOOR_FILE = re.compile(r"^Persistent_Floor_\d+$")
WAIT_FOR_PATH_SECONDS = 180   # how long to keep polling fast after a run
MAX_ATTEMPTS = 3              # tries per version of a floor file

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


def areas_being_cleaned(reported: dict[str, Any]) -> list[str]:
    """Room names in the current job; empty when idle or cleaning everywhere."""
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
        if isinstance(areas, dict):
            areas = [a for group in areas.values() if isinstance(group, list) for a in group]
        if isinstance(areas, list):
            names = [a.split(":", 1)[-1] for a in areas if isinstance(a, str) and a not in ("", "*")]
            if names:
                return names
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
            robot.started = datetime.now().astimezone() if robot.seen_docked else None
        if not out:
            robot.seen_docked = True

        await self._load(device, robot, version)

        if out:
            for name in areas_being_cleaned(reported):
                if name not in robot.rooms:
                    robot.rooms.append(name)
        elif robot.out:  # the run just ended
            robot.run_rooms, robot.run_version = robot.rooms, None
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
                cleaning=robot.rooms, name_map=names,
            )
        if robot.wait_until is not None:
            return render_floor_svg(
                robot.plan, Caption("waiting", "Docked", "Waiting for the cleaning path"),
                show_path=False, cleaned=robot.run_rooms, name_map=names,
            )
        rooms = robot.run_rooms if robot.run_version == robot.loaded and robot.run_rooms else None
        rooms = rooms or rooms_on_path(robot.plan, names)
        detail = [friendly_time(robot.loaded[1])]
        if robot.plan.minutes:
            detail.append(f"{robot.plan.minutes} min")
        title = f"Cleaned {listed(rooms)}" if rooms else "Last clean"
        return render_floor_svg(
            robot.plan, Caption("docked", title, " · ".join(d for d in detail if d)),
            cleaned=rooms, name_map=names,
        )
