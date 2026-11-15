import numpy as np

from nicetoolbox.detectors.feature_detectors.body_crossing.utils import REQUIRED_POINTS
from nicetoolbox.live.backends import Observation, PersonBody, wholebody_to_observation
from nicetoolbox.live.body_crossing import LiveBodyCrossing
from nicetoolbox.live.cues import SustainedCue

FPS = 10.0

# A seated person facing the camera (image x right, y down; their left is on the image right).
SEATED = {
    "left_shoulder": (1.0, 0.0), "right_shoulder": (-1.0, 0.0), "left_wrist": (1.0, 2.5), "right_wrist": (-1.0, 2.5),
    "left_hip": (0.6, 2.5), "right_hip": (-0.6, 2.5), "left_knee": (0.8, 3.0), "right_knee": (-0.8, 3.0),
    "left_ankle": (0.8, 4.5), "right_ankle": (-0.8, 4.5),
}  # fmt: skip
FOLDED = {"left_wrist": (-0.9, 1.4), "right_wrist": (0.9, 1.5)}
CROSSED_ANKLES = {"left_ankle": (-0.3, 4.5), "right_ankle": (0.3, 4.5)}


def _body(moved=None, unseen=()) -> Observation:
    """One person as an observation: the seated pose with some points moved or not seen."""
    coords = {name: np.array((moved or {}).get(name, xy), dtype=float) for name, xy in SEATED.items()}
    confidence = {name: 0.0 if name in unseen else 1.0 for name in SEATED}
    return Observation(bodies=[PersonBody(coords, confidence)])


def _feed(detector, sequence):
    states, events = [], []
    for i, observation in enumerate(sequence):
        result, frame_events = detector.update(observation, i / FPS, i)
        states.append(result["crossing_state"])
        events.extend(frame_events)
    return states, events


def test_score_matches_the_batch_detector():
    result, _ = LiveBodyCrossing().update(_body(FOLDED), 0.0, 0)
    np.testing.assert_allclose(result["crossing_score"], [[0.9, -0.8]])


def test_cue_starts_after_min_duration_and_ends_when_arms_uncross():
    detector = LiveBodyCrossing(min_duration=0.5)
    # 2 open frames, 8 folded frames (0.2s - 0.9s), then open again.
    states, events = _feed(detector, [_body()] * 2 + [_body(FOLDED)] * 8 + [_body()] * 2)

    assert [s[0, 0] for s in states] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert [(e.behaviour, e.part, e.state, e.frame) for e in events] == [
        ("body_crossing", "arms", "start", 7),
        ("body_crossing", "arms", "end", 10),
    ]
    assert events[0].onset == 0.2


def test_hand_passing_across_the_body_never_cues():
    detector = LiveBodyCrossing(min_duration=0.5)
    _, events = _feed(detector, [_body(FOLDED)] * 3 + [_body()] * 3)
    assert events == []


def test_only_requested_parts_raise_cues():
    both = {**FOLDED, **CROSSED_ANKLES}
    states, events = _feed(LiveBodyCrossing(parts=("arms",), min_duration=0.0), [_body(both)])
    assert {e.part for e in events} == {"arms"}
    assert np.isnan(states[0][0, 1])  # legs are scored but raise no cue
    _, events = _feed(LiveBodyCrossing(parts=("arms", "legs"), min_duration=0.0), [_body(both)])
    assert {e.part for e in events} == {"arms", "legs"}


def test_hidden_wrist_is_bridged_up_to_max_gap():
    hidden = _body(FOLDED, unseen=("left_wrist",))
    # folded from frame 0; the wrist is hidden for 2 frames (0.2 s) mid-way
    sequence = [_body(FOLDED)] * 3 + [hidden] * 2 + [_body(FOLDED)] * 3

    bridged, events = _feed(LiveBodyCrossing(min_duration=0.5, max_gap_duration=0.3), sequence)
    assert [(e.state, e.frame) for e in events] == [("start", 5)]  # counted from frame 0, not restarted
    assert events[0].onset == 0.0
    assert np.isnan(bridged[3][0, 0])  # the hidden frames themselves stay unknown

    _, events = _feed(LiveBodyCrossing(min_duration=0.5, max_gap_duration=0.0), sequence)
    assert events == []  # without bridging the count restarts at frame 5 and never reaches 0.5 s


def test_gap_longer_than_max_gap_ends_the_cue():
    hidden = _body(FOLDED, unseen=("left_wrist",))
    detector = LiveBodyCrossing(min_duration=0.0, max_gap_duration=0.2)
    _, events = _feed(detector, [_body(FOLDED)] + [hidden] * 4)
    # seen at 0.0 s; unknown at 0.1 and 0.2 is tolerated, 0.3 is too long
    assert [(e.state, e.frame) for e in events] == [("start", 0), ("end", 3)]


def test_uncrossing_ends_the_cue_at_once_even_with_a_gap_allowance():
    detector = LiveBodyCrossing(min_duration=0.0, max_gap_duration=1.0)
    _, events = _feed(detector, [_body(FOLDED), _body()])
    assert [(e.state, e.frame) for e in events] == [("start", 0), ("end", 1)]


def test_side_on_person_is_unknown():
    side_on = _body({"left_shoulder": (0.2, 0.0), "right_shoulder": (-0.2, 0.0)})
    result, _ = LiveBodyCrossing().update(side_on, 0.0, 0)
    assert np.isnan(result["crossing_score"]).all()


def test_sustained_cue_default_still_ends_on_unknown():
    cue, events = SustainedCue("x", 0.0), []
    cue.step(0, "part", 1.0, 0.0, 0, events)
    cue.step(0, "part", np.nan, 0.1, 1, events)
    assert [e.state for e in events] == ["start", "end"]


def test_wholebody_to_observation_reads_coco_body_points():
    keypoints = np.arange(133 * 2, dtype=float).reshape(1, 133, 2)
    scores = np.linspace(0, 1, 133)[np.newaxis]
    observation = wholebody_to_observation(keypoints, scores, parts=("bodies",))

    assert observation.hands == [] and observation.faces == []
    body = observation.bodies[0]
    assert set(body.coords) == set(REQUIRED_POINTS)
    np.testing.assert_array_equal(body.coords["left_wrist"], keypoints[0, 9])
    np.testing.assert_array_equal(body.coords["right_ankle"], keypoints[0, 16])
    assert body.confidence["left_shoulder"] == scores[0, 5]
