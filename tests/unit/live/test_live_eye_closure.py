import numpy as np

from nicetoolbox.live.backends import Observation, PersonEyes
from nicetoolbox.live.eye_closure import LiveEyeClosure, OpenEyeBaseline

FPS = 10.0


def _eye(ear: float, x0: float = 0.0) -> np.ndarray:
    """Six points in the toolbox's eye order (corner, upper, upper, corner, lower, lower), 1 wide.

    Lids sit +-ear/2 from the corner line, so both vertical pairs span `ear` and the EAR is `ear`.
    """
    h = ear / 2
    return np.array([[x0, 0], [x0 + 0.33, h], [x0 + 0.66, h], [x0 + 1, 0], [x0 + 0.66, -h], [x0 + 0.33, -h]])


def _face(left: float = None, right: float = None) -> Observation:
    """One face; an eye given as None is not seen."""
    face = PersonEyes.empty()
    for side, ear, x0 in (("left_eye", left, 2.0), ("right_eye", right, 0.0)):
        if ear is not None:
            face.coords[side] = _eye(ear, x0)
            face.confidence[side] = np.ones(6)
    return Observation(faces=[face])


def _feed(detector, sequence):
    results, events = [], []
    for i, observation in enumerate(sequence):
        result, frame_events = detector.update(observation, i / FPS, i)
        results.append(result)
        events.extend(frame_events)
    return results, events


def test_ear_matches_eye_opening():
    (result,), _ = _feed(LiveEyeClosure(relative=0.0), [_face(0.3, 0.05)])
    np.testing.assert_allclose(result["ear"][0], [0.3, 0.05])
    np.testing.assert_array_equal(result["closed"][0], [0.0, 1.0, 0.0])  # only the right eye is closed


def test_cue_needs_both_eyes_closed_for_min_duration():
    detector = LiveEyeClosure(threshold=0.15, min_duration=1.0, relative=0.0)
    # open 0.0-0.1s, both closed 0.2-1.3s, open again from 1.4s
    sequence = [_face(0.3, 0.3)] * 2 + [_face(0.05, 0.05)] * 12 + [_face(0.3, 0.3)] * 2
    results, events = _feed(detector, sequence)
    cue = [r["cue_state"][0] for r in results]
    assert cue.index(1.0) == 12  # 0.2s onset + 1.0s
    assert [(e.part, e.state, e.frame) for e in events] == [("both_eyes", "start", 12), ("both_eyes", "end", 14)]


def test_blink_does_not_cue():
    detector = LiveEyeClosure(threshold=0.15, min_duration=1.0, relative=0.0)
    sequence = ([_face(0.3, 0.3)] * 5 + [_face(0.05, 0.05)] * 2) * 4
    _, events = _feed(detector, sequence)
    assert events == []


def test_unseen_eye_is_unknown_and_ends_cue():
    detector = LiveEyeClosure(min_duration=0.0, relative=0.0)
    results, events = _feed(detector, [_face(0.05, 0.05), _face(0.05, None), Observation()])
    assert np.isnan(results[1]["closed"][0, 2]) and np.isnan(results[1]["cue_state"][0])
    assert [e.state for e in events] == ["start", "end"]


def test_relative_threshold_adapts_to_narrow_open_eyes():
    # This person's open eyes measure 0.09, under the fixed 0.1 threshold.
    fixed = LiveEyeClosure(threshold=0.1, min_duration=0.0, relative=0.0)
    relative = LiveEyeClosure(threshold=0.1, min_duration=0.0, relative=0.6)
    sequence = [_face(0.09, 0.09)] * 40 + [_face(0.03, 0.03)] * 3

    fixed_results, _ = _feed(fixed, sequence)
    assert fixed_results[39]["closed"][0, 2] == 1.0  # the fixed threshold calls open eyes closed

    results, events = _feed(relative, sequence)
    assert np.isnan(results[0]["closed"][0, 2])  # unknown while the baseline warms up
    assert results[39]["closed"][0, 2] == 0.0
    np.testing.assert_allclose(results[39]["threshold"][0], [0.054, 0.054])
    assert results[40]["closed"][0, 2] == 1.0
    assert [(e.state, e.frame) for e in events] == [("start", 40)]


def test_baseline_forgets_history_outside_window():
    baseline = OpenEyeBaseline(window=1.0, min_samples=3)
    for i in range(10):
        baseline.update(0.3, i / FPS)
    for i in range(10, 25):
        open_ear = baseline.update(0.2, i / FPS)
    assert open_ear == 0.2
