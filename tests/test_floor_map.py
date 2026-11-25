"""Floor scan map: parsing, drawing and what is shown during and after a run.

The floor file here is made up; it follows the field layout described in
``src/floor_map.py``.
"""

from __future__ import annotations

import json
import struct
import xml.dom.minidom
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.floor_map import group_rooms, parse_floor_file, render_floor_svg
from src.floor_tracker import (
    FloorMapTracker,
    areas_being_cleaned,
    friendly_time,
    newest_floor_file,
)

W, H = 40, 30


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        byte = v & 0x7F
        v >>= 7
        out.append(byte | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _int(num: int, v: int) -> bytes:
    return _varint(num << 3) + _varint(v)


def _float(num: int, v: float) -> bytes:
    return _varint(num << 3 | 5) + struct.pack("<f", v)


def _bytes(num: int, v: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(v)) + v


def _point(x: float, y: float) -> bytes:
    return (_float(1, x) if x else b"") + (_float(2, y) if y else b"")


def _polygon(x0, y0, x1, y1) -> bytes:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return b"".join(_bytes(1, _point(*c)) for c in corners)


def _room(name: str, box) -> bytes:
    return _bytes(15, _bytes(2, name.encode()) + _bytes(3, name.encode()) + _bytes(4, _polygon(*box)))


def floor_file(path=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0)), minutes=12) -> bytes:
    cells = bytearray([75]) * (W * H)
    for row in range(2, 28):
        for col in range(2, 38):
            edge = row in (2, 27) or col in (2, 37)
            cells[row * W + col] = 100 if edge else 15
    cells[10 * W + 10] = 90
    grid = (_float(1, 0.1) + _bytes(2, b"") + _int(3, W) + _int(4, H) + _bytes(6, bytes(cells)))
    units = [int(v / 0.002) for p in path for v in p]
    return b"".join([
        _bytes(3, b"AABBCCDD"),
        _bytes(5, grid),
        _bytes(7, _point(0.4, 0.4)),
        _bytes(8, _float(1, 0.002) + _int(2, 2) + _bytes(3, struct.pack(f"<{len(units)}h", *units))),
        _bytes(11, _int(1, 1700000000) + _int(2, minutes)),
        _room("Study", (0.3, 0.3, 1.8, 1.5)),
        _room("Den", (1.9, 0.3, 3.6, 1.5)),
        _room("Whole Floor", (0.2, 0.2, 3.7, 2.6)),
        _bytes(16, _int(1, 1) + _bytes(4, _polygon(2.5, 1.8, 3.0, 2.2))),
    ])


class TestParse:
    def test_reads_every_part(self):
        plan = parse_floor_file(floor_file())
        assert (plan.width, plan.height, plan.floor_id, plan.minutes) == (W, H, "AABBCCDD", 12)
        assert plan.resolution == pytest.approx(0.1)
        assert plan.origin == (0.0, 0.0)  # zero values are omitted in protobuf
        assert plan.dock == pytest.approx((0.4, 0.4))
        assert [r.name for r in plan.rooms] == ["Study", "Den", "Whole Floor"]
        assert len(plan.zones) == 1
        assert plan.path[1] == pytest.approx((1.0, 0.5))

    @pytest.mark.parametrize("body", [b"", b"not a floor file", floor_file()[:40], b"{}"])
    def test_other_content_is_refused(self, body):
        assert parse_floor_file(body) is None

    def test_unreadable_extra_keeps_the_floor_plan(self):
        broken = floor_file() + _bytes(15, b"\xff\xff\xff")
        plan = parse_floor_file(broken)
        assert plan is not None and plan.path


class TestDraw:
    def test_output_is_valid_svg_with_all_layers(self):
        svg = render_floor_svg(parse_floor_file(floor_file()), "Last clean: 12 min")
        xml.dom.minidom.parseString(svg)
        for expected in ('class="path"', 'class="nogo"', 'class="dock"', ">Study<", ">Den<", "Last clean: 12 min"):
            assert expected in svg
        assert 'class="cleaning"' not in svg

    def test_grouping_rooms_are_not_labelled(self):
        plan = parse_floor_file(floor_file())
        assert group_rooms(plan.rooms) == {2}
        assert ">Whole Floor<" not in render_floor_svg(plan, "x")

    def test_highlight_without_path(self):
        svg = render_floor_svg(parse_floor_file(floor_file()), "Cleaning Den", show_path=False, highlight=["Den"])
        assert svg.count('class="cleaning"') == 1
        assert 'class="path"' not in svg

    def test_names_from_the_app_are_shown_and_escaped(self):
        svg = render_floor_svg(parse_floor_file(floor_file()), "a < b", name_map={"Den": "Play & Den"},
                               highlight=["Play & Den"])
        assert ">Play &amp; Den<" in svg and "a &lt; b" in svg and 'class="cleaning"' in svg
        xml.dom.minidom.parseString(svg)

    def test_columns_are_mirrored(self):
        svg = render_floor_svg(parse_floor_file(floor_file()), "x")
        # the dock is at column 4 of 40, so it is drawn near the right-hand side
        assert f'translate({(W - 4) * 6:.1f},{4 * 6:.1f})' in svg


def shadow(written="2026-01-01T10:00:00.000Z", areas="*"):
    return {"shadow": {"properties": {"reported": {
        "fileList": {
            "Persistent_Floor_1": {"value": {"fileName": "a.bin"}, "updatedAt": written},
            "Persistent_Floor_2": {"value": {"fileName": ""}, "updatedAt": "2027-01-01T00:00:00.000Z"},
            "Visual_Floor_1": {"value": {"fileName": "b.bin"}, "updatedAt": "2027-01-01T00:00:00.000Z"},
        },
        "AreasToClean_V2": {"value": areas},
    }}}}


def job(*rooms):
    return json.dumps({"floor_id": "AABBCCDD", "areas_to_clean": [f"UserRoom:{r}" for r in rooms]})


def robot(state="docked", job_active=None):
    device = MagicMock()
    device.dsn, device.product_name = "SND1", "Test"
    device.ha_state, device.job_active, device.room_name_map = state, job_active, {}
    return device


OLD, NEW = "2026-01-01T10:00:00.000Z", "2026-01-01T11:00:00.000Z"


class TestShadowReading:
    def test_newest_listed_floor_file_with_a_name(self):
        reported = shadow()["shadow"]["properties"]["reported"]
        assert newest_floor_file(reported) == ("Persistent_Floor_1", OLD)
        assert newest_floor_file({}) is None

    def test_areas_in_both_formats(self):
        assert areas_being_cleaned({"AreasToClean_V2": {"value": job("Den", "Study")}}) == ["Den", "Study"]
        v3 = json.dumps({"areas_to_clean": {"UserRoom": ["Den"]}})
        assert areas_being_cleaned({"AreasToClean_V3": {"value": v3}}) == ["Den"]
        assert areas_being_cleaned({"AreasToClean_V3": {"value": "*"}, "AreasToClean_V2": {"value": "*"}}) == []

    def test_friendly_time(self):
        assert friendly_time("rubbish") == ""
        assert friendly_time(OLD).startswith("Jan 1, ")


@pytest.mark.asyncio
class TestRun:
    def make(self):
        api = AsyncMock()
        api.fetch_property_file.return_value = floor_file()
        now = [0.0]
        return api, now, FloorMapTracker(api, clock=lambda: now[0])

    async def test_docked_shows_last_path_and_fetches_once(self):
        api, _, tracker = self.make()
        first = await tracker.update(robot(), shadow())
        second = await tracker.update(robot(), shadow())
        assert first == second and 'class="path"' in first and "Last clean: " in first and "12 min" in first
        assert api.fetch_property_file.await_count == 1
        api.fetch_property_file.assert_awaited_with("SND1", "Persistent_Floor_1")

    async def test_whole_run(self):
        api, now, tracker = self.make()
        await tracker.update(robot(), shadow())

        out = await tracker.update(robot("cleaning"), shadow(areas=job("Den")))
        assert "Cleaning Den" in out and 'class="cleaning"' in out and 'class="path"' not in out

        back = await tracker.update(robot("returning"), shadow(areas="*"))
        assert "Returning to the dock" in back and 'class="cleaning"' in back

        docked = await tracker.update(robot(), shadow())
        assert "waiting for the cleaning path" in docked and 'class="path"' not in docked
        assert tracker.waiting("SND1")

        now[0] = 20.0
        done = await tracker.update(robot(), shadow(NEW))
        assert 'class="path"' in done and "Last clean: Den · " in done
        assert not tracker.waiting("SND1")
        assert api.fetch_property_file.await_count == 2

    async def test_new_file_already_there_when_docked(self):
        _, _, tracker = self.make()
        await tracker.update(robot(), shadow())
        await tracker.update(robot("cleaning"), shadow(areas=job("Den")))
        done = await tracker.update(robot(), shadow(NEW))
        assert "Last clean: Den · " in done and not tracker.waiting("SND1")

    async def test_gives_up_waiting(self):
        _, now, tracker = self.make()
        await tracker.update(robot(), shadow())
        await tracker.update(robot("cleaning"), shadow(areas=job("Den")))
        await tracker.update(robot(), shadow())
        now[0] = 181.0
        done = await tracker.update(robot(), shadow())
        assert not tracker.waiting("SND1")
        # the old path is not claimed to be of the run just finished
        assert "Last clean: Jan 1" in done and "Den ·" not in done

    async def test_charging_in_the_middle_of_a_job_counts_as_out(self):
        _, _, tracker = self.make()
        await tracker.update(robot(), shadow())
        await tracker.update(robot("cleaning", True), shadow(areas=job("Den")))
        charging = await tracker.update(robot("docked", True), shadow(areas=job("Den")))
        assert "Cleaning Den" in charging and not tracker.waiting("SND1")

    async def test_restart_during_a_run_hides_the_previous_path(self):
        _, _, tracker = self.make()
        svg = await tracker.update(robot("cleaning"), shadow(areas=job("Den")))
        assert 'class="path"' not in svg

    async def test_no_floor_file_means_nothing_to_draw(self):
        api, _, tracker = self.make()
        assert await tracker.update(robot(), {"shadow": {"properties": {"reported": {}}}}) is None
        api.fetch_property_file.assert_not_awaited()

    async def test_unreadable_file_is_tried_three_times(self):
        api, _, tracker = self.make()
        api.fetch_property_file.return_value = None
        for _ in range(6):
            assert await tracker.update(robot(), shadow()) is None
        assert api.fetch_property_file.await_count == 3

    async def test_failed_download_keeps_the_last_map_and_retries(self):
        api, _, tracker = self.make()
        await tracker.update(robot(), shadow())
        api.fetch_property_file.return_value = None
        kept = await tracker.update(robot(), shadow(NEW))
        assert 'class="path"' in kept
        api.fetch_property_file.return_value = floor_file()
        await tracker.update(robot(), shadow(NEW))
        assert api.fetch_property_file.await_count == 3
