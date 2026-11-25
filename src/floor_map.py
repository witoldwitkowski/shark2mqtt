"""Draw a robot's floor plan and cleaning path from its floor scan file.

The Shark cloud keeps one ``Persistent_Floor_N`` property file per floor and
rewrites it when a run ends. It is a protobuf message. The fields used here were
observed on UR250BEYUS and UR2360EEUS robots (inferred, not documented by Shark):

==== =========================================================================
3    floor id
5    grid: 1 = metres per cell, 2 = origin {1: x, 2: y}, 3 = width,
     4 = height, 6 = one byte per cell, row-major
7    dock position {1: x, 2: y, 3: heading}
8    path of the last run: {1: metres per unit, 2: 2, 3: little-endian int16
     x, y pairs}
11   last run: {1: start time, 2: minutes}
15   rooms: {2: robot name, 3: name, 4: {1: [{1: x, 2: y}, ...]}}
16   no-go zones: {4: polygon, as for rooms}
==== =========================================================================

Cells: 75 unexplored, 100 wall, 0 / 10 / 15 floor; anything else is drawn as a
faint obstacle. Coordinates are metres; zero values are left out of protobuf.

Columns are mirrored when drawn, which gives a true top-down view in the same
orientation as the room file (``MARD``). The drawing itself is in
:mod:`.floor_style`, shared with other vacuum integrations.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Iterable

from . import floor_style
from .map_render import _centroid, _varint

logger = logging.getLogger(__name__)

Point = tuple[float, float]

UNEXPLORED = 75
WALL = 100
FLOOR = (0, 10, 15)
MAX_CELLS = 4_000_000       # refuse absurd grids rather than draw them
JUMP_M = 2.0                # a longer step in the path is a gap, not movement

@dataclass
class Room:
    robot_name: str
    name: str
    points: list[Point]


@dataclass
class FloorPlan:
    width: int
    height: int
    resolution: float
    origin: Point
    cells: bytes
    floor_id: str = ""
    dock: Point | None = None
    path: list[Point] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    zones: list[list[Point]] = field(default_factory=list)
    minutes: int = 0
    shapes: bytes | None = field(default=None, repr=False, compare=False)  # drawing cache


# --- protobuf ------------------------------------------------------------------

def _message(b: bytes) -> dict[int, list]:
    """Every top-level field of a protobuf message."""
    i, out = 0, {}
    while i < len(b):
        key, i = _varint(b, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(b, i)
        elif wire == 1:
            value = struct.unpack("<d", b[i:i + 8])[0]
            i += 8
        elif wire == 5:
            value = struct.unpack("<f", b[i:i + 4])[0]
            i += 4
        elif wire == 2:
            size, i = _varint(b, i)
            value = b[i:i + size]
            if len(value) != size:
                raise ValueError("truncated field")
            i += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > len(b):
            raise ValueError("truncated message")
        out.setdefault(num, []).append(value)
    return out


def _first(msg: dict[int, list], num: int, default=0):
    return msg[num][0] if num in msg else default


def _point(b: bytes) -> Point:
    msg = _message(b)
    x, y = float(_first(msg, 1, 0.0)), float(_first(msg, 2, 0.0))
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("coordinate is not a number")
    return x, y


def _polygon(entry: dict[int, list]) -> list[Point]:
    return [_point(p) for p in _message(_first(entry, 4, b"")).get(1, [])]


def _text(entry: dict[int, list], num: int) -> str:
    value = _first(entry, num, b"")
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else ""


def _path(b: bytes) -> list[Point]:
    msg = _message(b)
    scale = float(_first(msg, 1, 0.0))
    raw = _first(msg, 3, b"")
    if _first(msg, 2) != 2 or not 0 < scale < 1:
        return []
    count = len(raw) // 4
    values = struct.unpack(f"<{count * 2}h", raw[:count * 4])
    return [(values[i] * scale, values[i + 1] * scale) for i in range(0, count * 2, 2)]


def parse_floor_file(body: bytes) -> FloorPlan | None:
    """Read a floor file, or return None if it isn't one.

    The grid is required. Anything else that can't be read is left out, so an
    unexpected model still gets a floor plan.
    """
    try:
        top = _message(body)
        grid = _message(top[5][0])
        width, height, cells = int(grid[3][0]), int(grid[4][0]), grid[6][0]
        resolution = float(grid[1][0])
        origin = _point(_first(grid, 2, b""))
    except Exception:  # noqa: BLE001 - not a floor file, or a new format
        logger.debug("Could not read the floor file grid", exc_info=True)
        return None
    if not (isinstance(cells, bytes) and width > 0 and height > 0 and 0 < resolution < 1
            and width * height == len(cells) <= MAX_CELLS):
        return None
    plan = FloorPlan(width, height, resolution, origin, cells)

    def optional(label: str, read) -> None:
        try:
            read()
        except Exception:  # noqa: BLE001
            logger.debug("Could not read %s in the floor file", label, exc_info=True)

    def read_dock() -> None:
        if 7 in top:
            plan.dock = _point(top[7][0])

    def read_path() -> None:
        if 8 in top:
            plan.path = _path(top[8][0])

    def read_rooms() -> None:
        for blob in top.get(15, []):
            entry = _message(blob)
            points = _polygon(entry)
            robot_name = _text(entry, 2)
            name = _text(entry, 3) or robot_name
            if name and len(points) >= 3:
                plan.rooms.append(Room(robot_name or name, name, points))

    def read_zones() -> None:
        for blob in top.get(16, []):
            points = _polygon(_message(blob))
            if len(points) >= 3:
                plan.zones.append(points)

    def read_run() -> None:
        plan.floor_id = _text(top, 3)
        if 11 in top:
            plan.minutes = int(_first(_message(top[11][0]), 2))

    for label, read in (("the dock", read_dock), ("the path", read_path), ("rooms", read_rooms),
                        ("no-go zones", read_zones), ("the run summary", read_run)):
        optional(label, read)
    return plan


# --- geometry ------------------------------------------------------------------

def _inside(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _area(points: list[Point]) -> float:
    return abs(sum(points[i - 1][0] * y - x * points[i - 1][1] for i, (x, y) in enumerate(points))) / 2


def group_rooms(rooms: list[Room]) -> set[int]:
    """Indexes of rooms that span other rooms ("whole floor" areas).

    A room holding the centre of two or more smaller rooms is a grouping made in
    the app; labelling it would bury the labels of the rooms inside.
    """
    centres = [_centroid(r.points) for r in rooms]
    sizes = [_area(r.points) for r in rooms]
    return {
        i for i, room in enumerate(rooms)
        if sum(1 for j, c in enumerate(centres)
               if j != i and sizes[j] < sizes[i] and _inside(c, room.points)) >= 2
    }


def _label_cell(plan: FloorPlan, room: Room) -> Point:
    """Where to write a room's name, in cells: its centre, moved onto floor."""
    cells = [((x - plan.origin[0]) / plan.resolution, (y - plan.origin[1]) / plan.resolution)
             for x, y in room.points]
    cx, cy = _centroid(cells)

    def is_floor(col: int, row: int) -> bool:
        return 0 <= col < plan.width and 0 <= row < plan.height and plan.cells[row * plan.width + col] in FLOOR

    if is_floor(int(cx), int(cy)) and _inside((cx, cy), cells):
        return cx, cy
    xs, ys = [c for c, _ in cells], [r for _, r in cells]
    floor = [(col + 0.5, row + 0.5)
             for row in range(max(int(min(ys)), 0), min(int(max(ys)) + 1, plan.height), 2)
             for col in range(max(int(min(xs)), 0), min(int(max(xs)) + 1, plan.width), 2)
             if is_floor(col, row) and _inside((col + 0.5, row + 0.5), cells)]
    return min(floor, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2) if floor else (cx, cy)


def _enclosed(plan: FloorPlan) -> bytes:
    """Cell classes for drawing, without the scan rays that leak through windows.

    Floor counts only where a wall closes it in on at least three of four sides,
    and walls only where they border such floor.
    """
    w, h, cells = plan.width, plan.height, plan.cells
    sides = bytearray(w * h)

    def sweep(start: int, step: int, count: int) -> None:
        walled, i = False, start
        for _ in range(count):
            value = cells[i]
            if value == WALL:
                walled = True
            elif value == UNEXPLORED:
                walled = False
            elif walled:
                sides[i] += 1
            i += step

    for row in range(h):
        sweep(row * w, 1, w)
        sweep(row * w + w - 1, -1, w)
    for col in range(w):
        sweep(col, w, h)
        sweep((h - 1) * w + col, -w, h)

    out = bytearray(w * h)
    for i, value in enumerate(cells):
        if value not in (UNEXPLORED, WALL) and sides[i] >= 3:
            out[i] = floor_style.FLOOR if value in FLOOR else floor_style.OBSTACLE
    for i, value in enumerate(cells):
        if value == WALL:
            row, col = divmod(i, w)
            if any(out[r * w + c] in (floor_style.FLOOR, floor_style.OBSTACLE)
                   for r in range(max(row - 2, 0), min(row + 3, h))
                   for c in range(max(col - 2, 0), min(col + 3, w))):
                out[i] = floor_style.WALL
    return bytes(out)


# --- drawing -------------------------------------------------------------------

def _scene(plan: FloorPlan, name_map: dict[str, str]) -> floor_style.Scene:
    """The floor plan in drawing cells, mirrored to a true top-down view."""

    def cell(point: Point) -> Point:
        col = (point[0] - plan.origin[0]) / plan.resolution
        row = (point[1] - plan.origin[1]) / plan.resolution
        return plan.width - col, row

    if plan.shapes is None:
        shown = _enclosed(plan)
        # mirror each row
        plan.shapes = b"".join(shown[r * plan.width:(r + 1) * plan.width][::-1] for r in range(plan.height))
    groups = group_rooms(plan.rooms)
    rooms = []
    for i, room in enumerate(plan.rooms):
        col, row = _label_cell(plan, room)
        corners = [cell(p) for p in room.points]
        outline = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in corners) + "Z"
        rooms.append(floor_style.Room(
            name_map.get(room.robot_name) or room.name, (plan.width - col, row), outline, i in groups))
    return floor_style.Scene(
        plan.width, plan.height, plan.shapes, plan.resolution, rooms=rooms,
        zones=[[cell(p) for p in zone] for zone in plan.zones],
        dock=cell(plan.dock) if plan.dock else None,
    )


def _strokes(plan: FloorPlan) -> list[list[Point]]:
    """The path split wherever it jumps."""
    strokes: list[list[Point]] = []
    previous = None
    for point in plan.path:
        if previous is None or math.dist(point, previous) > JUMP_M:
            strokes.append([])
        strokes[-1].append(point)
        previous = point
    return strokes


def rooms_on_path(plan: FloorPlan, name_map: dict[str, str] | None = None) -> list[str]:
    """The rooms the recorded path was mostly in, going by the smallest room at each point."""
    name_map = name_map or {}
    groups = group_rooms(plan.rooms)
    candidates = sorted((r for i, r in enumerate(plan.rooms) if i not in groups), key=lambda r: _area(r.points))
    counts: dict[str, int] = {}
    for point in plan.path:
        room = next((r for r in candidates if _inside(point, r.points)), None)
        if room:
            name = name_map.get(room.robot_name) or room.name
            counts[name] = counts.get(name, 0) + 1
    return floor_style.main_rooms(counts, len(plan.path))


def render_floor_svg(
    plan: FloorPlan,
    caption: floor_style.Caption,
    *,
    show_path: bool = True,
    cleaned: Iterable[str] = (),
    cleaning: Iterable[str] = (),
    name_map: dict[str, str] | None = None,
) -> str:
    """The floor plan as an SVG.

    ``cleaned`` and ``cleaning`` hold room names (robot or display names) of a
    finished or running clean. ``name_map`` turns robot names into the names
    shown in the app.
    """
    name_map = name_map or {}
    scene = _scene(plan, name_map)
    if show_path:
        mirrored = []
        for stroke in _strokes(plan):
            mirrored.append([(plan.width - (x - plan.origin[0]) / plan.resolution,
                              (y - plan.origin[1]) / plan.resolution) for x, y in stroke])
        scene.path = mirrored
    return floor_style.render(
        scene, caption,
        cleaned=[name_map.get(n, n) for n in cleaned],
        cleaning=[name_map.get(n, n) for n in cleaning],
    )
