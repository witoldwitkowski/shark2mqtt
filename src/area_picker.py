"""Mark four corners on a vacuum's map and clean inside them.

The SharkClean app's spot clean is a room clean with the mode ``PinDrop`` whose
"room" is four corners in map metres: ``x1,y1,x2,y2,x3,y3,x4,y4``. The robot
follows the shape it is given, so the corners need not make a rectangle.

Home Assistant side: an image entity shows the map in a frame that does not move
when the robot rewrites it, and its attributes list the points a dashboard can
offer for tapping. A tap arrives as ``vacuum.send_command`` with the command
``area_corner``; the picture is redrawn with the corners so far. Nothing is sent
to the robot until ``area_clean``.

Off unless ``AREA_PICKER_ROBOTS`` names a robot.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import floor_style
from .floor_map import FloorPlan, _area, _enclosed, render_floor_svg
from .floor_style import Caption
from .shark_device import SharkVacuum

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Frame = tuple[float, float, float, float]     # x0, y0, x1, y1 in map metres

MODE = "PinDrop"
PASSES = 2                   # what the app's spot clean sends
CORNERS = 4
TAP_METRES = 0.762           # spacing of the tap points (2.5 ft)
NEAR_METRES = 0.4            # a tap point may sit this far off the floor, to reach walls
MARGIN_METRES = 0.4          # around the floor, inside the frame
EXPIRES_SECONDS = 600        # forget an unfinished or unsent shape
SAME_METRES = 0.2            # closer than this is the same corner
MIN_AREA = 0.25              # square metres; smaller is no job for a robot this size
SQUARE_FEET = 10.7639
COMMANDS = ("area_corner", "area_undo", "area_clear", "area_clean")


# --- shape ---------------------------------------------------------------------

def _signed_area(points: list[Point]) -> float:
    return sum(points[i - 1][0] * y - x * points[i - 1][1] for i, (x, y) in enumerate(points)) / 2


def _crosses(a: Point, b: Point, c: Point, d: Point) -> bool:
    def side(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _self_crossing(points: list[Point]) -> bool:
    n = len(points)
    return any(_crosses(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n])
               for i in range(n) for j in range(i + 2, n) if (j + 1) % n != i)


def ordered(corners: list[Point]) -> list[Point]:
    """The corners as an outline, wound and started like the app's squares.

    Tap order is kept unless the edges would cross; then the corners are taken in
    order around their centre.
    """
    points = list(corners)
    if _self_crossing(points):
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    if _signed_area(points) > 0:
        points.reverse()
    start = max(range(len(points)), key=lambda i: (points[i][1] - points[i][0], points[i][1]))
    return points[start:] + points[:start]


def coordinates(shape: list[Point]) -> str:
    """The shape as the robot takes it."""
    return ",".join(str(round(value, 3)) for point in shape for value in point)


# --- map -----------------------------------------------------------------------

def floor_extent(plan: FloorPlan) -> Frame:
    """The drawn floor and walls, in metres, with a margin."""
    shown = _enclosed(plan)
    cols, rows = [], []
    for row in range(plan.height):
        line = shown[row * plan.width:(row + 1) * plan.width]
        if any(line):
            rows.append(row)
            cols.append(len(line) - len(line.lstrip(b"\x00")))
            cols.append(len(line.rstrip(b"\x00")))
    if not rows:
        cols, rows = [0, plan.width], [0, plan.height - 1]
    ox, oy, res = plan.origin[0], plan.origin[1], plan.resolution
    return (math.floor((ox + min(cols) * res - MARGIN_METRES) * 10) / 10,
            math.floor((oy + rows[0] * res - MARGIN_METRES) * 10) / 10,
            math.ceil((ox + max(cols) * res + MARGIN_METRES) * 10) / 10,
            math.ceil((oy + (rows[-1] + 1) * res + MARGIN_METRES) * 10) / 10)


def tap_points(plan: FloorPlan, frame: Frame) -> list[Point]:
    """Grid points on or just beside the floor, on a grid fixed to the map's zero."""
    shown = _enclosed(plan)
    res, (ox, oy) = plan.resolution, plan.origin
    reach = int(NEAR_METRES / res)
    offsets = sorted(((dc, dr) for dc in range(-reach, reach + 1) for dr in range(-reach, reach + 1)
                      if dc * dc + dr * dr <= reach * reach), key=lambda o: o[0] ** 2 + o[1] ** 2)
    on_floor = (floor_style.FLOOR, floor_style.OBSTACLE)

    def near_floor(x: float, y: float) -> bool:
        col, row = int((x - ox) / res), int((y - oy) / res)
        return any(0 <= col + dc < plan.width and 0 <= row + dr < plan.height
                   and shown[(row + dr) * plan.width + col + dc] in on_floor for dc, dr in offsets)

    x0, y0, x1, y1 = frame
    return [(round(i * TAP_METRES, 3), round(j * TAP_METRES, 3))
            for j in range(math.ceil(y0 / TAP_METRES), math.floor(y1 / TAP_METRES) + 1)
            for i in range(math.ceil(x0 / TAP_METRES), math.floor(x1 / TAP_METRES) + 1)
            if near_floor(i * TAP_METRES, j * TAP_METRES)]


def picture_size(frame: Frame) -> tuple[int, int]:
    return floor_style.WIDTH, round(floor_style.layout(frame[2] - frame[0], frame[3] - frame[1])[3])


def position(frame: Frame, point: Point) -> tuple[float, float]:
    """Where a map point is in the picture: per cent from the left and from the top.

    The map is drawn mirrored, so a larger x is further left.
    """
    x0, y0, x1, y1 = frame
    scale, left, top, height = floor_style.layout(x1 - x0, y1 - y0)
    return (100 * (left + (x1 - point[0]) * scale) / floor_style.WIDTH,
            100 * (top + (point[1] - y0) * scale) / round(height))


# --- one picker per robot --------------------------------------------------------

@dataclass
class _Robot:
    device: SharkVacuum | None = None
    plan: FloorPlan | None = None
    frame: Frame | None = None
    points: list[Point] = field(default_factory=list)
    points_for: FloorPlan | None = None
    corners: list[Point] = field(default_factory=list)
    sent: list[Point] = field(default_factory=list)       # the shape last sent, still shown
    touched: float = 0.0
    note: tuple[str, str] | None = None                   # title and detail to show instead
    announced: bool = False
    published: str = ""


class AreaPicker:
    def __init__(self, config: Any, api: Any, mqtt: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._config = config
        self._api = api
        self._mqtt = mqtt
        self._clock = clock
        self._robots: dict[str, _Robot] = {}

    def enabled(self, device: SharkVacuum) -> bool:
        wanted = {v.strip().lower() for v in str(self._config.area_picker_robots or "").split(",") if v.strip()}
        return bool(wanted & {device.product_name.lower(), device.dsn.lower()})

    def corners(self, dsn: str) -> list[Point]:
        robot = self._robots.get(dsn)
        return list(robot.corners) if robot else []

    async def refresh(self, device: SharkVacuum, plan: FloorPlan | None) -> None:
        """Called each poll with the robot's latest floor plan."""
        if plan is None or not self.enabled(device):
            return
        robot = self._robots.setdefault(device.dsn, _Robot())
        robot.device, robot.plan = device, plan
        if robot.points_for is not plan:
            robot.frame = self._frame(device.dsn, plan)
            robot.points, robot.points_for = tap_points(plan, robot.frame), plan
        self._expire(robot)
        if not robot.announced:
            await self._mqtt.publish_area_discovery(device)
            robot.announced = True
        await self._publish(robot)

    async def handle(self, dsn: str, command: str, params: dict[str, Any]) -> bool:
        """Act on an area command. True means a clean was sent, so poll soon."""
        robot = self._robots.get(dsn)
        if robot is None or robot.plan is None or command not in COMMANDS:
            logger.warning("Area command %s ignored: no area picker for this robot yet", command)
            return False
        self._expire(robot)
        robot.note = None
        started = False
        if command == "area_corner":
            self._add(robot, params)
        elif command == "area_undo":
            if robot.corners:
                robot.corners.pop()
        elif command == "area_clear":
            robot.corners, robot.sent = [], []
        else:
            started = await self._clean(robot)
        robot.touched = self._clock()
        await self._publish(robot)
        return started

    # --- internals ---

    def _expire(self, robot: _Robot) -> None:
        if (robot.corners or robot.sent or robot.note) and self._clock() - robot.touched > EXPIRES_SECONDS:
            robot.corners, robot.sent, robot.note = [], [], None

    def _add(self, robot: _Robot, params: dict[str, Any]) -> None:
        assert robot.frame is not None
        try:
            x, y = float(params["x"]), float(params["y"])
        except (KeyError, TypeError, ValueError):
            logger.warning("area_corner needs numbers x and y, got %r", params)
            return
        x0, y0, x1, y1 = robot.frame
        if not (x0 <= x <= x1 and y0 <= y <= y1):      # also refuses NaN
            logger.warning("area_corner (%s, %s) is outside the map", x, y)
            return
        if len(robot.corners) >= CORNERS:
            robot.corners = []
        robot.sent = []
        if any(math.dist((x, y), corner) < SAME_METRES for corner in robot.corners):
            robot.note = ("Corner already marked", "Tap a different point")
            return
        corners = [*robot.corners, (x, y)]
        if len(corners) == CORNERS and _area(ordered(corners)) < MIN_AREA:
            robot.note = ("That area is too small", "Pick a fourth corner further out")
            return
        robot.corners = corners

    async def _clean(self, robot: _Robot) -> bool:
        assert robot.device is not None and robot.plan is not None
        device = robot.device
        name = device.product_name
        if len(robot.corners) != CORNERS:
            robot.note = ("Mark four corners first", f"{len(robot.corners)} of {CORNERS} so far")
            return False
        # the last poll can be minutes old, and an offline robot keeps its last state
        try:
            raw = await self._api.get_device(device.dsn)
            now = SharkVacuum.from_skegox(raw)
        except Exception:  # noqa: BLE001
            logger.warning("Area clean: could not read %s", name, exc_info=True)
            robot.note = ("Not started", "Could not reach Shark; try again")
            return False
        if not now.is_online:
            robot.note = ("Not started", f"{name} is offline")
            return False
        if now.ha_state != "docked" or now.job_active:
            robot.note = ("Not started", f"{name} must be on its dock first")
            return False
        floor_id = robot.plan.floor_id or device.floor_id
        if not floor_id:
            robot.note = ("Not started", "The map has no floor id yet")
            return False
        shape = ordered(robot.corners)
        try:
            await self._api.clean_rooms(
                device.dsn, rooms=[coordinates(shape)], floor_id=floor_id, clean_type="dry",
                clean_count=PASSES, mode=MODE, use_v3=now.has_areas_v3,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Area clean: sending to %s failed", name, exc_info=True)
            robot.note = ("Not started", "Shark did not take the command; try again")
            return False
        logger.info("Area clean sent to %s: %s", name, coordinates(shape))
        robot.sent, robot.corners = shape, []
        robot.note = (f"Sent to {name}", f"{_area(shape) * SQUARE_FEET:.0f} sq ft · {PASSES} passes")
        return True

    def _frame(self, dsn: str, plan: FloorPlan) -> Frame:
        """The saved frame for this floor, so tap points stay put; saved on first sight."""
        current = floor_extent(plan)
        path = Path(self._config.token_dir) / f"area_frame_{dsn}.json"
        try:
            saved = json.loads(path.read_text())
            if saved["floor_id"] == plan.floor_id:
                frame = tuple(float(v) for v in saved["frame"])
                if (current[0] < frame[0] - MARGIN_METRES or current[1] < frame[1] - MARGIN_METRES
                        or current[2] > frame[2] + MARGIN_METRES or current[3] > frame[3] + MARGIN_METRES):
                    logger.warning("The map has outgrown the area picker frame; delete %s and rebuild "
                                   "the dashboard to include the new part", path)
                return frame  # type: ignore[return-value]
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            logger.warning("Could not read %s; making a new frame", path, exc_info=True)
        try:
            path.write_text(json.dumps({"floor_id": plan.floor_id, "frame": current}))
        except OSError:
            logger.warning("Could not save %s; tap points may move after a restart", path, exc_info=True)
        return current

    def _draw(self, robot: _Robot) -> str:
        assert robot.plan is not None and robot.frame is not None and robot.device is not None
        corners = robot.corners
        count = len(corners)
        shape = ordered(corners) if count == CORNERS else robot.sent
        if robot.note:
            title, detail = robot.note
        elif count == CORNERS:
            title = "Ready to clean"
            detail = f"{_area(shape) * SQUARE_FEET:.0f} sq ft · press Clean area, or tap to start over"
        elif count:
            title, detail = f"{count} of {CORNERS} corners", "Tap the next corner"
        else:
            title, detail = f"Tap {CORNERS} corners", f"{robot.device.product_name} cleans inside them"
        state = "error" if title.startswith(("Not started", "That area")) else (
            "cleaning" if count == CORNERS or robot.sent else "docked")
        names = robot.device.room_name_map or {}
        return render_floor_svg(
            robot.plan, Caption(state, title, detail), show_path=False, name_map=names,
            frame=robot.frame, area=shape, corners=corners,
        )

    async def _publish(self, robot: _Robot) -> None:
        assert robot.device is not None and robot.frame is not None
        svg = self._draw(robot)
        attributes = {
            "floor_id": robot.plan.floor_id if robot.plan else "",
            "frame": list(robot.frame),
            "picture_size": list(picture_size(robot.frame)),
            "tap_points": [list(p) for p in robot.points],
        }
        signature = hashlib.sha256(
            f"{svg}|{len(robot.corners)}|{json.dumps(attributes)}".encode()).hexdigest()
        if signature == robot.published:
            return
        await self._mqtt.publish_area(robot.device, svg, attributes, len(robot.corners))
        robot.published = signature
