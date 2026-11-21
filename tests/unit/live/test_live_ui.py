import json
import urllib.request

from nicetoolbox.live.cues import CueEvent
from nicetoolbox.live.ui import UiServer, band_command


def _event(behaviour="body_crossing", part="arms", state="start"):
    return CueEvent(behaviour, 0, part, state, 3.0, 90, 1.0)


def test_arms_crossing_maps_to_band_commands():
    assert band_command(_event(state="start")) == "A"
    assert band_command(_event(state="end")) == "a"


def test_cues_the_band_does_not_take_map_to_none():
    assert band_command(_event(part="legs")) is None
    assert band_command(_event(behaviour="eye_closure", part="both_eyes")) is None


def _read_sse_message(stream) -> dict:
    data = None
    while True:
        line = stream.readline().decode().rstrip("\n")
        if line.startswith("data: "):
            data = line[len("data: ") :]
        elif line == "" and data is not None:
            return json.loads(data)


def test_server_serves_the_page_and_pushes_cues():
    ui = UiServer(port=0)
    try:
        assert ui.url.startswith("http://127.0.0.1:")
        page = urllib.request.urlopen(ui.url, timeout=5).read().decode()
        assert "navigator.bluetooth" in page

        stream = urllib.request.urlopen(ui.url + "events", timeout=5)
        assert _read_sse_message(stream) == {"type": "hello"}
        ui.publish_event(_event())
        message = _read_sse_message(stream)
        assert message["type"] == "cue"
        assert message["command"] == "A"
        assert message["part"] == "arms" and message["state"] == "start"
        stream.close()
    finally:
        ui.close()


def test_video_is_only_wanted_while_a_page_watches():
    ui = UiServer(port=0)
    try:
        assert not ui.wants_video
        ui.publish_frame(b"\xff\xd8fake-jpeg\xff\xd9")
        stream = urllib.request.urlopen(ui.url + "video.mjpg", timeout=5)
        assert stream.readline() == b"--frame\r\n"
        assert ui.wants_video
        stream.close()
    finally:
        ui.close()
