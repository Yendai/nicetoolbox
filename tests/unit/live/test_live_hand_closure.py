import numpy as np

from nicetoolbox.detectors.feature_detectors.hand_closure import utils as hand_utils
from nicetoolbox.live.backends import PersonHands, wholebody_to_hands
from nicetoolbox.live.hand_closure import LiveHandClosure
from nicetoolbox.live.latency import STAGES, LatencyTracker
from nicetoolbox.live.main import run
from nicetoolbox.live.sources import Frame

FPS = 10.0


def _hand(finger_scale: float) -> np.ndarray:
    """A synthetic 21-point hand whose four fingers all have extension ratio `finger_scale`."""
    coords = np.zeros((21, 2))
    for knuckle, tip in hand_utils.FINGERS.values():
        direction = np.array([(knuckle - 13) / 10, 1.0])
        for step, point in enumerate(range(knuckle, tip + 1)):
            coords[point] = direction * (1.0 + (finger_scale - 1.0) * step / 3)
    return coords


def _person(left: float = None, right: float = None) -> PersonHands:
    """One person; a hand given as None is not seen."""
    person = PersonHands.empty()
    for side, scale in (("left_hand", left), ("right_hand", right)):
        if scale is not None:
            person.coords[side] = _hand(scale)
            person.confidence[side] = np.ones(21)
    return person


def _feed(detector, sequence):
    """Run a list of per-frame people lists through the detector; return states and events."""
    states, events = [], []
    for i, people in enumerate(sequence):
        _, state, frame_events = detector.update(people, i / FPS, i)
        states.append(state)
        events.extend(frame_events)
    return states, events


def test_cue_starts_only_after_min_duration_and_ends_when_hand_opens():
    detector = LiveHandClosure(threshold=1.2, min_duration=0.5)
    # 2 open frames, 8 fist frames (0.2s - 0.9s), then open again.
    sequence = [[_person(1.9, 1.9)]] * 2 + [[_person(1.0, 1.9)]] * 8 + [[_person(1.9, 1.9)]] * 2
    states, events = _feed(detector, sequence)

    left = [s[0, 0] for s in states]
    # The fist starts at 0.2s, so the cue starts at 0.7s (frame 7) and stays until frame 10.
    assert left == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert [(e.side, e.state, e.frame) for e in events] == [("left_hand", "start", 7), ("left_hand", "end", 10)]
    assert events[0].onset == 0.2
    assert all(s[0, 1] == 0 for s in states)  # right hand stayed open


def test_short_fist_never_cues():
    detector = LiveHandClosure(threshold=1.2, min_duration=0.5)
    sequence = [[_person(1.0, 1.9)]] * 3 + [[_person(1.9, 1.9)]] * 3
    states, events = _feed(detector, sequence)
    assert events == []
    assert all(s[0, 0] == 0 for s in states)


def test_unseen_hand_is_nan_and_ends_active_cue():
    detector = LiveHandClosure(threshold=1.2, min_duration=0.0)
    sequence = [[_person(1.0, None)], [_person(None, None)]]
    states, events = _feed(detector, sequence)
    assert states[0][0, 0] == 1 and np.isnan(states[0][0, 1])
    assert np.isnan(states[1][0, 0])
    assert [e.state for e in events] == ["start", "end"]


def test_person_leaving_frame_ends_cue():
    detector = LiveHandClosure(threshold=1.2, min_duration=0.0)
    _, events = _feed(detector, [[_person(1.0, 1.9)], []])
    assert [(e.person, e.state) for e in events] == [(0, "start"), (0, "end")]


def test_wholebody_to_hands_uses_coco_wholebody_hand_ranges():
    keypoints = np.arange(133 * 2, dtype=float).reshape(1, 133, 2)
    scores = np.arange(133, dtype=float).reshape(1, 133) / 133
    (person,) = wholebody_to_hands(keypoints, scores)
    np.testing.assert_array_equal(person.coords["left_hand"][0], keypoints[0, 91])
    np.testing.assert_array_equal(person.coords["right_hand"][20], keypoints[0, 132])
    assert person.confidence["left_hand"].shape == (21,)


def test_run_records_latency_and_forwards_events(tmp_path):
    def frames():
        for i in range(6):
            yield Frame(index=i, image=np.zeros((4, 4, 3)), timestamp=i / FPS, captured_at=0.0, capture_ms=1.0)

    sequence = iter([[_person(1.0, 1.9)]] * 6)

    def backend(image, timestamp):  # noqa: ARG001
        return next(sequence)

    latency = LatencyTracker(str(tmp_path / "latency.csv"))
    events = []
    run(frames(), backend, LiveHandClosure(min_duration=0.2), latency, on_event=events.append)
    latency.close()

    summary = latency.summary()
    assert summary["frames"] == 6
    assert set(STAGES) <= set(summary)
    assert [(e.state, e.frame) for e in events] == [("start", 2)]
    assert len((tmp_path / "latency.csv").read_text().strip().splitlines()) == 7
