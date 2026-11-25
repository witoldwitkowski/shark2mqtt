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
orientation as the room file (``MARD``). Output is SVG and needs no third-party
library, like :mod:`.map_render`.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Iterable
from xml.sax.saxutils import escape

from .map_render import _centroid, _varint

logger = logging.getLogger(__name__)

Point = tuple[float, float]

PX = 6                      # drawing units per grid cell
UNEXPLORED = 75
WALL = 100
FLOOR = (0, 10, 15)
MAX_CELLS = 4_000_000       # refuse absurd grids rather than draw them
JUMP_M = 2.0                # a longer step in the path is a gap, not movement

INK = "#16242a"
FLOOR_COLOUR = "#e8eeec"
WALL_COLOUR = "#35444b"
OBSTACLE_COLOUR = "#a6b2b6"
PATH_COLOUR = "#2168d6"
ZONE_COLOUR = "#c23b34"
CLEAN_FILL = "#12a594"
CLEAN_EDGE = "#0b6e63"


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


def group_rooms(rooms: list[Room]) -> set[int]:
    """Indexes of rooms that span other rooms ("whole floor" areas).

    A room holding the centre of two or more smaller rooms is a grouping made in
    the app; labelling it would bury the labels of the rooms inside.
    """
    def size(points: list[Point]) -> float:
        return abs(sum(points[i - 1][0] * y - x * points[i - 1][1] for i, (x, y) in enumerate(points))) / 2

    centres = [_centroid(r.points) for r in rooms]
    sizes = [size(r.points) for r in rooms]
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


# --- drawing -------------------------------------------------------------------

def _grid_paths(plan: FloorPlan) -> dict[str, str]:
    """One SVG path per colour, made of horizontal runs of cells (mirrored)."""
    paths: dict[str, list[str]] = {"floor": [], "obstacle": [], "wall": []}
    for row in range(plan.height):
        line = plan.cells[row * plan.width:(row + 1) * plan.width]
        start, current = 0, None
        for col in range(plan.width + 1):
            if col < plan.width:
                value = line[col]
                kind = (None if value == UNEXPLORED else "wall" if value == WALL
                        else "floor" if value in FLOOR else "obstacle")
            else:
                kind = None
            if kind != current:
                if current:
                    run = col - start
                    paths[current].append(f"M{plan.width - col} {row}h{run}v1h-{run}z")
                start, current = col, kind
    return {kind: "".join(parts) for kind, parts in paths.items()}


def render_floor_svg(
    plan: FloorPlan,
    caption: str,
    *,
    show_path: bool = True,
    highlight: Iterable[str] = (),
    name_map: dict[str, str] | None = None,
) -> str:
    """The floor plan as an SVG.

    ``highlight`` holds room names (robot or display names) to mark as being
    cleaned. ``name_map`` turns robot names into the names shown in the app.
    """
    name_map = name_map or {}
    wanted = {h.lower() for h in highlight}

    def shown(room: Room) -> str:
        return name_map.get(room.robot_name) or room.name

    def to_svg(point: Point) -> Point:
        col = (point[0] - plan.origin[0]) / plan.resolution
        row = (point[1] - plan.origin[1]) / plan.resolution
        return (plan.width - col) * PX, row * PX

    def points_attr(points: list[Point]) -> str:
        return " ".join("{:.1f},{:.1f}".format(*to_svg(p)) for p in points)

    # frame what the robot has seen
    seen = [i for i, v in enumerate(plan.cells) if v != UNEXPLORED]
    if seen:
        cols = [i % plan.width for i in seen]
        x0, x1 = (plan.width - max(cols) - 1) * PX, (plan.width - min(cols)) * PX
        y0, y1 = (seen[0] // plan.width) * PX, (seen[-1] // plan.width + 1) * PX
    else:
        x0, y0, x1, y1 = 0, 0, plan.width * PX, plan.height * PX
    pad = 4 * PX
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    width = max(x1 - x0, 600)
    x0 -= (width - (x1 - x0)) / 2
    font = width / 42
    footer = font * 4.6
    height = (y1 - y0) + footer

    highlighted = [r for r in plan.rooms
                   if r.robot_name.lower() in wanted or shown(r).lower() in wanted]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
        f'<defs><pattern id="nogo" width="{PX * 2}" height="{PX * 2}" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="100%" height="100%" fill="{ZONE_COLOUR}" fill-opacity="0.10"/>'
        f'<line x1="0" y1="0" x2="0" y2="{PX * 2}" stroke="{ZONE_COLOUR}" stroke-width="3" stroke-opacity="0.55"/>'
        '</pattern></defs>',
        f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
    ]
    grid = _grid_paths(plan)
    out.append(f'<g transform="scale({PX})" shape-rendering="crispEdges">'
               f'<path d="{grid["floor"]}" fill="{FLOOR_COLOUR}"/></g>')
    for room in highlighted:
        out.append(f'<polygon class="cleaning" points="{points_attr(room.points)}" fill="{CLEAN_FILL}" '
                   f'fill-opacity="0.28" stroke="{CLEAN_EDGE}" stroke-width="{PX * 0.7:.1f}" '
                   'stroke-linejoin="round"/>')
    out.append(f'<g transform="scale({PX})" shape-rendering="crispEdges">'
               f'<path d="{grid["obstacle"]}" fill="{OBSTACLE_COLOUR}"/>'
               f'<path d="{grid["wall"]}" fill="{WALL_COLOUR}"/></g>')
    for zone in plan.zones:
        out.append(f'<polygon class="nogo" points="{points_attr(zone)}" fill="url(#nogo)" '
                   f'stroke="{ZONE_COLOUR}" stroke-width="{PX * 0.45:.1f}" stroke-linejoin="round"/>')

    if show_path and len(plan.path) > 1:
        commands, previous = [], None
        for point in plan.path:
            moved = previous is None or math.dist(point, previous) > JUMP_M
            commands.append("{}{:.1f} {:.1f}".format("M" if moved else "L", *to_svg(point)))
            previous = point
        out.append(f'<path class="path" d="{"".join(commands)}" fill="none" stroke="{PATH_COLOUR}" '
                   f'stroke-width="{max(PX * 0.5, width / 360):.1f}" stroke-linecap="round" '
                   'stroke-linejoin="round" stroke-opacity="0.92"/>')

    if plan.dock:
        dx, dy = to_svg(plan.dock)
        r = font * 0.62
        out.append(f'<g class="dock" transform="translate({dx:.1f},{dy:.1f})">'
                   f'<circle r="{r:.1f}" fill="{INK}" stroke="#ffffff" stroke-width="{r * 0.24:.1f}"/>'
                   f'<path d="M{-r * .5:.1f} {-r * .02:.1f}L0 {-r * .5:.1f}L{r * .5:.1f} {-r * .02:.1f}'
                   f'H{r * .33:.1f}V{r * .45:.1f}H{-r * .33:.1f}V{-r * .02:.1f}Z" fill="#ffffff"/></g>')

    groups = group_rooms(plan.rooms)
    placed: list[tuple[float, float, float]] = []
    for i, room in enumerate(plan.rooms):
        if i in groups:
            continue
        col, row = _label_cell(plan, room)
        x, y = (plan.width - col) * PX, row * PX
        half = len(shown(room)) * font * 0.29
        while any(abs(x - px) < half + ph and abs(y - py) < font * 1.15 for px, py, ph in placed):
            y += font * 1.15
        placed.append((x, y, half))
        weight = 700 if room in highlighted else 500
        out.append(f'<text x="{x:.0f}" y="{y + font * 0.35:.0f}" text-anchor="middle" font-size="{font:.0f}" '
                   f'font-weight="{weight}" fill="{INK}" stroke="#ffffff" stroke-width="{font * 0.28:.1f}" '
                   f'stroke-linejoin="round" paint-order="stroke">{escape(shown(room))}</text>')

    # footer: what the picture shows, then a key
    left = x0 + pad
    size = min(font * 1.3, (width - 2 * pad) / max(len(caption) * 0.58, 1))
    out.append(f'<text class="caption" x="{left:.0f}" y="{y1 + font * 1.5:.0f}" font-size="{size:.0f}" '
               f'font-weight="700" fill="{INK}">{escape(caption)}</text>')
    key_y = y1 + font * 3.3
    x = left
    entries = []
    if show_path and len(plan.path) > 1:
        entries.append(("Path", f'<line x1="0" y1="0" x2="{font * 1.4:.0f}" y2="0" stroke="{PATH_COLOUR}" '
                                f'stroke-width="{font * 0.24:.1f}" stroke-linecap="round"/>'))
    if highlighted:
        entries.append(("Cleaning", f'<rect y="{-font * .45:.1f}" width="{font * 1.4:.0f}" height="{font * .9:.1f}" '
                                    f'fill="{CLEAN_FILL}" fill-opacity="0.28" stroke="{CLEAN_EDGE}" stroke-width="2"/>'))
    if plan.zones:
        entries.append(("No-go zone", f'<rect y="{-font * .45:.1f}" width="{font * 1.4:.0f}" height="{font * .9:.1f}" '
                                      f'fill="url(#nogo)" stroke="{ZONE_COLOUR}" stroke-width="2"/>'))
    if plan.dock:
        entries.append(("Dock", f'<circle cx="{font * .7:.1f}" r="{font * .45:.1f}" fill="{INK}"/>'))
    for label, swatch in entries:
        out.append(f'<g transform="translate({x:.0f},{key_y:.0f})">{swatch}<text x="{font * 1.8:.0f}" '
                   f'y="{font * 0.33:.0f}" font-size="{font * 0.92:.0f}" fill="#4a5b61">{label}</text></g>')
        x += font * (3.2 + len(label) * 0.52)
    out.append("</svg>")
    return "\n".join(out)
