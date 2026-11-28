"""Area picker: corners to shape, the payload sent, the fixed frame and the pictures.

Floor files here are made up (see test_floor_map).
"""

from __future__ import annotations

import json
import math
import re
import xml.dom.minidom
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import area_picker, floor_style
from src.area_picker import AreaPicker, coordinates, floor_extent, ordered, position, tap_points
from src.floor_map import parse_floor_file, path_share_in_area, render_floor_svg, rooms_on_path
from src.floor_style import Caption
from src.floor_tracker import FloorMapTracker, area_being_cleaned, areas_being_cleaned
from src.mqtt_client import MqttClient
from src.skegox_api import SkegoxApi

from .test_floor_map import H, W, _bytes, _room, floor_file, robot, shadow

DSN = "SND1"
CORNERS = [(0.762, 0.762), (3.048, 0.762), (1.524, 1.524), (0.762, 1.524)]
# a slanted shape and a square, wound and started like the ones a real robot accepted
SLANT = [(1.0, -2.0), (3.0, -2.0), (1.6, -4.0), (1.0, -4.0)]
SQUARE = [(0.5, -5.0), (2.0, -5.0), (2.0, -6.5), (0.5, -6.5)]


def area_file(box=(0.5, 0.5, 1.5, 1.5), **kwargs) -> bytes:
    return floor_file(**kwargs) + _room("PinDrop", box)


def raw(docked=True, online=True, v3=False):
    reported = {
        "Operating_Mode": {"value": 3 if docked else 2},
        "DockedStatus": {"value": docked},
        "Robot_Room_List": {"value": "AABBCCDD:Study:Den"},
    }
    if v3:
        reported["AreasToClean_V3"] = {"value": "*"}
    return {
        "deviceId": DSN,
        "registry": {"Battery_Serial_Num": f"X-{DSN}"},
        "metadata": {"deviceName": "Test Vac"},
        "connectivityStatus": {"connected": online},
        "shadow": {"properties": {"reported": reported}},
    }


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def make_picker(tmp_path, robots="Test Vac", state=None):
    config = MagicMock()
    config.area_picker_robots, config.token_dir = robots, str(tmp_path)
    api = AsyncMock()
    api.get_device.return_value = state or raw()
    mqtt = AsyncMock()
    clock = Clock()
    return AreaPicker(config, api, mqtt, clock), api, mqtt, clock


def vac():
    device = MagicMock()
    device.dsn, device.product_name, device.floor_id, device.room_name_map = DSN, "Test Vac", "AABBCCDD", {}
    return device


async def ready(tmp_path, **kwargs):
    picker, api, mqtt, clock = make_picker(tmp_path, **kwargs)
    await picker.refresh(vac(), parse_floor_file(floor_file()))
    return picker, api, mqtt, clock


async def tap(picker, *points):
    for x, y in points:
        assert not await picker.handle(DSN, "area_corner", {"x": x, "y": y})


def last_picture(mqtt):
    _, svg, attributes, count = mqtt.publish_area.call_args.args
    xml.dom.minidom.parseString(svg)
    return svg, attributes, count


class TestShape:
    @pytest.mark.parametrize("shape", [SLANT, SQUARE])
    def test_shapes_that_worked_on_a_robot_are_kept(self, shape):
        assert ordered(shape) == shape

    @pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (0, 2, 1, 3), (1, 3, 2, 0)])
    def test_any_tap_order_gives_the_same_outline(self, order):
        assert ordered([SLANT[i] for i in order]) == SLANT

    def test_a_dented_shape_keeps_its_dent(self):
        dart = [(0.0, 2.0), (2.0, 1.0), (0.0, 0.0), (0.8, 1.0)]
        result = ordered(dart)
        assert sorted(result) == sorted(dart) and not area_picker._self_crossing(result)

    def test_text_is_what_the_robot_takes(self):
        assert coordinates(SLANT) == "1.0,-2.0,3.0,-2.0,1.6,-4.0,1.0,-4.0"
        assert coordinates([(0.12345678, -1.2345678)]) == "0.123,-1.235"


class TestSend:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("v3", [False, True])
    async def test_sends_the_marked_area(self, tmp_path, v3):
        picker, api, mqtt, _ = await ready(tmp_path, state=raw(v3=v3))
        await tap(picker, *[CORNERS[i] for i in (2, 0, 3, 1)])
        assert await picker.handle(DSN, "area_clean", {})
        api.clean_rooms.assert_awaited_once_with(
            DSN, rooms=[coordinates(ordered(CORNERS))], floor_id="AABBCCDD", clean_type="dry",
            clean_count=2, mode="PinDrop", use_v3=v3)
        svg, _, count = last_picture(mqtt)
        assert count == 0 and ">Sent to Test Vac<" in svg and 'class="area"' in svg
        # pressing again sends nothing more
        assert not await picker.handle(DSN, "area_clean", {})
        assert api.clean_rooms.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("v3, expected", [
        (False, [("Areas_To_Clean", {"floor_id": "F", "areas_to_clean": ["PinDrop:1.0,2.0"], "clean_count": 2}),
                 ("Operating_Mode", 2)]),
        (True, [("AreasToClean_V3", {"areas_to_clean": {"PinDrop": ["1.0,2.0"]}, "clean_count": 2,
                                     "floor_id": "F", "cleantype": "dry"})]),
    ])
    async def test_wire_format(self, v3, expected):
        api = SkegoxApi.__new__(SkegoxApi)
        api.set_desired_property = AsyncMock()
        await api.clean_rooms(DSN, ["1.0,2.0"], "F", clean_count=2, mode="PinDrop", use_v3=v3)
        sent = [(c.args[1], json.loads(c.args[2]) if isinstance(c.args[2], str) else c.args[2])
                for c in api.set_desired_property.await_args_list]
        assert sent == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state, reason", [
        (raw(docked=False), "must be on its dock"), (raw(online=False), "is offline"),
    ])
    async def test_refuses_a_robot_that_is_away_or_offline(self, tmp_path, state, reason):
        picker, api, mqtt, _ = await ready(tmp_path, state=state)
        await tap(picker, *CORNERS)
        assert not await picker.handle(DSN, "area_clean", {})
        api.clean_rooms.assert_not_awaited()
        svg, _, count = last_picture(mqtt)
        assert count == 4 and ">Not started<" in svg and reason in svg

    @pytest.mark.asyncio
    async def test_read_or_send_failures_send_nothing_and_keep_the_corners(self, tmp_path):
        picker, api, mqtt, _ = await ready(tmp_path)
        await tap(picker, *CORNERS)
        api.get_device.side_effect = OSError("no network")
        assert not await picker.handle(DSN, "area_clean", {})
        api.clean_rooms.assert_not_awaited()
        api.get_device.side_effect = None
        api.clean_rooms.side_effect = OSError("refused")
        assert not await picker.handle(DSN, "area_clean", {})
        assert len(picker.corners(DSN)) == 4

    @pytest.mark.asyncio
    async def test_needs_four_corners(self, tmp_path):
        picker, api, mqtt, _ = await ready(tmp_path)
        await tap(picker, *CORNERS[:3])
        assert not await picker.handle(DSN, "area_clean", {})
        api.get_device.assert_not_awaited()
        api.clean_rooms.assert_not_awaited()


class TestMarking:
    @pytest.mark.asyncio
    async def test_picture_follows_the_taps(self, tmp_path):
        picker, _, mqtt, _ = await ready(tmp_path)
        assert ">Tap 4 corners<" in last_picture(mqtt)[0]
        await tap(picker, *CORNERS[:2])
        svg, _, count = last_picture(mqtt)
        assert count == 2 and svg.count('class="corner"') == 2 and 'class="edges"' in svg
        assert 'class="area"' not in svg
        await tap(picker, *CORNERS[2:])
        svg, _, count = last_picture(mqtt)
        assert count == 4 and 'class="area"' in svg and ">Ready to clean<" in svg and "sq ft" in svg

    @pytest.mark.asyncio
    async def test_a_fifth_tap_starts_again_and_undo_and_clear(self, tmp_path):
        picker, _, _, _ = await ready(tmp_path)
        await tap(picker, *CORNERS, (1.524, 0.762))
        assert picker.corners(DSN) == [(1.524, 0.762)]
        await tap(picker, CORNERS[0])
        await picker.handle(DSN, "area_undo", {})
        assert picker.corners(DSN) == [(1.524, 0.762)]
        await picker.handle(DSN, "area_clear", {})
        await picker.handle(DSN, "area_undo", {})
        assert picker.corners(DSN) == []

    @pytest.mark.asyncio
    async def test_repeated_and_hopeless_corners_are_not_taken(self, tmp_path):
        picker, _, mqtt, _ = await ready(tmp_path)
        await tap(picker, (1.0, 1.0), (1.05, 1.0))
        assert len(picker.corners(DSN)) == 1 and ">Corner already marked<" in last_picture(mqtt)[0]
        await tap(picker, (1.3, 1.0), (1.6, 1.0), (1.9, 1.0))   # a straight line
        assert len(picker.corners(DSN)) == 3 and ">That area is too small<" in last_picture(mqtt)[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params", [{}, {"x": "a", "y": 1}, {"x": float("nan"), "y": 1}, {"x": 99, "y": 1}])
    async def test_bad_points_are_ignored(self, tmp_path, params):
        picker, _, _, _ = await ready(tmp_path)
        await picker.handle(DSN, "area_corner", params)
        assert picker.corners(DSN) == []

    @pytest.mark.asyncio
    async def test_an_old_shape_expires(self, tmp_path):
        picker, api, mqtt, clock = await ready(tmp_path)
        await tap(picker, *CORNERS)
        clock.now += area_picker.EXPIRES_SECONDS + 1
        assert not await picker.handle(DSN, "area_clean", {})
        api.clean_rooms.assert_not_awaited()
        assert picker.corners(DSN) == []

    @pytest.mark.asyncio
    async def test_only_named_robots_get_one(self, tmp_path):
        for robots in ("", "Another Vac"):
            picker, _, mqtt, _ = await ready(tmp_path, robots=robots)
            mqtt.publish_area_discovery.assert_not_awaited()
            mqtt.publish_area.assert_not_awaited()
            assert not await picker.handle(DSN, "area_clean", {})
        picker, _, mqtt, _ = await ready(tmp_path, robots="x, snd1")
        mqtt.publish_area_discovery.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nothing_is_republished_when_nothing_changed(self, tmp_path):
        picker, _, mqtt, _ = await ready(tmp_path)
        await picker.refresh(vac(), parse_floor_file(floor_file()))
        assert mqtt.publish_area.await_count == 1 and mqtt.publish_area_discovery.await_count == 1


def grown_file() -> bytes:
    """The same room in a grid with more unexplored cells around its far sides."""
    from .test_floor_map import _float, _int
    width, height = W + 15, H + 9
    old = parse_floor_file(floor_file())
    cells = bytearray([75]) * (width * height)
    for row in range(H):
        cells[row * width:row * width + W] = old.cells[row * W:(row + 1) * W]
    grid = _float(1, 0.1) + _bytes(2, b"") + _int(3, width) + _int(4, height) + _bytes(6, bytes(cells))
    return _bytes(3, b"AABBCCDD") + _bytes(5, grid)


class TestMapFrame:
    def test_tap_points_are_on_or_beside_the_floor(self):
        plan = parse_floor_file(floor_file())
        frame = floor_extent(plan)
        points = tap_points(plan, frame)
        assert (0.762, 0.762) in points and (3.81, 2.286) in points
        assert all(-0.2 <= x <= 4.2 and -0.2 <= y <= 3.2 for x, y in points)
        assert all(abs(x / 0.762 - round(x / 0.762)) < 1e-6 for x, _ in points)

    @pytest.mark.asyncio
    async def test_frame_and_positions_stay_when_the_grid_grows(self, tmp_path):
        picker, _, mqtt, _ = await ready(tmp_path)
        await tap(picker, (1.524, 0.762))
        before, attributes, _ = last_picture(mqtt)
        grown = parse_floor_file(grown_file())
        assert grown.width != parse_floor_file(floor_file()).width
        await picker.refresh(vac(), grown)
        after, later, _ = last_picture(mqtt)
        assert later["frame"] == attributes["frame"] and later["picture_size"] == attributes["picture_size"]
        corner = r'class="corner"><circle cx="([\d.]+)" cy="([\d.]+)"'
        assert re.search(corner, before).groups() == re.search(corner, after).groups()
        # and a restart reads the same frame back
        other, _, mqtt2, _ = make_picker(tmp_path)
        await other.refresh(vac(), grown)
        assert last_picture(mqtt2)[1]["frame"] == attributes["frame"]

    @pytest.mark.asyncio
    async def test_dashboard_positions_match_the_picture(self, tmp_path):
        picker, _, mqtt, _ = await ready(tmp_path)
        for point in [(0.762, 0.762), (3.048, 1.524), (1.524, 2.286)]:
            await picker.handle(DSN, "area_clear", {})
            await tap(picker, point)
            svg, attributes, _ = last_picture(mqtt)
            width, height = attributes["picture_size"]
            assert f'viewBox="0 0 {width} {height}"' in svg
            x, y = map(float, re.search(r'class="corner"><circle cx="([\d.]+)" cy="([\d.]+)"', svg).groups())
            left, top = position(tuple(attributes["frame"]), point)
            assert math.isclose(x, left / 100 * width, abs_tol=0.6)
            assert math.isclose(y, top / 100 * height, abs_tol=0.6)

    def test_other_maps_are_laid_out_as_before(self):
        scale, left, top, height = floor_style.layout(40, 30)
        assert (scale, left, top, height) == (pytest.approx(23.4), pytest.approx(32), 142, pytest.approx(942))


def job_v2(room):
    return json.dumps({"floor_id": "AABBCCDD", "areas_to_clean": [room], "clean_count": 2})


AREA_JOB = "PinDrop:0.5,0.5,1.5,0.5,1.5,1.5,0.5,1.5"


class TestStatusMap:
    def test_stored_area_is_not_a_room(self):
        plan = parse_floor_file(area_file())
        assert plan.area == [pytest.approx(p) for p in [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]]
        assert "PinDrop" not in [r.name for r in plan.rooms]
        assert rooms_on_path(plan) == ["Study"]
        svg = render_floor_svg(plan, Caption("docked", "x"))
        assert "PinDrop" not in svg and 'class="area"' not in svg
        assert path_share_in_area(plan) == 1.0

    @pytest.mark.parametrize("key, value", [
        ("AreasToClean_V2", job_v2(AREA_JOB)),
        ("AreasToClean_V3", json.dumps({"areas_to_clean": {"PinDrop": [AREA_JOB.split(":")[1]]}})),
    ])
    def test_job_is_named_not_spelled_out(self, key, value):
        reported = {key: {"value": value}}
        assert areas_being_cleaned(reported) == ["a marked area"]
        assert area_being_cleaned(reported) == [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
        assert area_being_cleaned({key: {"value": "*"}}) == []

    @pytest.mark.asyncio
    async def test_during_and_after_an_area_run(self):
        api = AsyncMock()
        api.fetch_property_file.return_value = area_file()
        tracker = FloorMapTracker(api)
        await tracker.update(robot("docked"), shadow())
        out = await tracker.update(robot("cleaning"), shadow(areas=job_v2(AREA_JOB)))
        assert ">Cleaning a marked area<" in out and 'class="area"' in out and "1.5,1.5" not in out
        done = await tracker.update(robot("docked"), shadow("2026-01-01T11:00:00.000Z"))
        assert ">Cleaned a marked area<" in done and 'class="area"' in done
        assert tracker.plan_for(DSN) is not None

    @pytest.mark.asyncio
    async def test_a_later_room_run_does_not_show_the_old_area(self):
        api = AsyncMock()
        api.fetch_property_file.return_value = area_file(box=(2.5, 1.5, 3.5, 2.5))
        tracker = FloorMapTracker(api)
        svg = await tracker.update(robot("docked"), shadow())     # after a restart
        assert ">Cleaned Study<" in svg and 'class="area"' not in svg

    @pytest.mark.asyncio
    async def test_an_area_run_is_recognised_after_a_restart(self):
        api = AsyncMock()
        api.fetch_property_file.return_value = area_file()
        svg = await FloorMapTracker(api).update(robot("docked"), shadow())
        assert ">Cleaned a marked area<" in svg and 'class="area"' in svg


class TestMqtt:
    @pytest.mark.asyncio
    async def test_entities(self):
        config = MagicMock()
        config.mqtt_prefix = "shark2mqtt"
        client = MqttClient(config)
        client._publish = AsyncMock()
        device = vac()
        device.device_info = {"identifiers": [DSN]}
        await client.publish_area_discovery(device)
        configs = {c.args[0]: c.args[1] for c in client._publish.await_args_list}
        assert sorted(t.split("/")[1] for t in configs) == ["button", "button", "button", "image", "sensor"]
        clean = configs["homeassistant/button/shark2mqtt_SND1_area_clean/config"]
        assert clean["command_topic"] == "shark2mqtt/SND1/send_command"
        assert MqttClient._parse_command(clean["payload_press"])[0] == "area_clean"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sent, polls", [(False, 0), (True, 1)])
    async def test_marking_does_not_poll_but_a_clean_does(self, sent, polls):
        config = MagicMock()
        config.mqtt_prefix = "shark2mqtt"
        client = MqttClient(config)
        message = MagicMock()
        message.topic.value = "shark2mqtt/SND1/send_command"
        message.payload = json.dumps({"command": "area_corner", "x": 1.0, "y": 2.0}).encode()

        async def messages():
            yield message

        client._client = AsyncMock()
        client._client.messages = messages()
        picker, handler, event = AsyncMock(), AsyncMock(), MagicMock()
        picker.handle.return_value = sent
        await client.command_listener(handler, {DSN: vac()}, event, picker)
        picker.handle.assert_awaited_once_with(DSN, "area_corner", {"x": 1.0, "y": 2.0})
        assert event.set.call_count == polls and not handler.mock_calls
