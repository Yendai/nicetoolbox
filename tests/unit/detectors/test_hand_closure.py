import numpy as np
import pytest

from nicetoolbox.detectors.feature_detectors.hand_closure import utils as hand_utils


def _hand(finger_scale: float) -> np.ndarray:
    """A synthetic 21-point hand: wrist at the origin, every finger pointing up.

    Each finger's points lie on a ray from the wrist through its knuckle, spaced so the tip
    is `finger_scale` times as far from the wrist as the knuckle: the extension ratio.
    """
    coords = np.zeros((21, 2))
    for knuckle, tip in hand_utils.FINGERS.values():
        direction = np.array([(knuckle - 13) / 10, 1.0])
        for step, point in enumerate(range(knuckle, tip + 1)):
            coords[point] = direction * (1.0 + (finger_scale - 1.0) * step / 3)
    # thumb points, unused by the ratio
    coords[1:5] = [[-1.0, 0.5], [-1.5, 0.8], [-1.8, 1.0], [-2.0, 1.2]]
    return coords


def test_resolve_hand_indices_maps_labels_in_point_order():
    labels = [f"right_hand_{i}" for i in range(21)] + [f"left_hand_{i}" for i in range(21)]
    indices = hand_utils.resolve_hand_indices(labels)
    assert indices["right_hand"] == list(range(21))
    assert indices["left_hand"] == list(range(21, 42))


def test_resolve_hand_indices_raises_on_missing_labels():
    labels = [f"left_hand_{i}" for i in range(21)]
    with pytest.raises(ValueError, match="right_hand"):
        hand_utils.resolve_hand_indices(labels)


def test_open_hand_has_higher_ratio_than_fist():
    hands = np.stack([_hand(1.9), _hand(1.0)])  # (2, 21, 2): open, fist
    confidence = np.ones((2, 21))
    ratio = hand_utils.calculate_extension_ratio(hands, confidence, min_confidence=0.3)
    np.testing.assert_allclose(ratio, [1.9, 1.0])


def test_ratio_is_scale_invariant():
    hand = _hand(1.6)
    confidence = np.ones(21)
    small = hand_utils.calculate_extension_ratio(hand, confidence, 0.3)
    large = hand_utils.calculate_extension_ratio(hand * 7.5 + 100.0, confidence, 0.3)
    np.testing.assert_allclose(small, large)


def test_low_confidence_finger_is_skipped():
    hand = _hand(1.0)
    hand[8] = [5.0, 9.0]  # index tip wildly off, but low confidence
    confidence = np.ones(21)
    confidence[8] = 0.1
    ratio = hand_utils.calculate_extension_ratio(hand, confidence, 0.3)
    np.testing.assert_allclose(ratio, 1.0)


def test_missing_wrist_gives_nan():
    hand = _hand(1.9)
    confidence = np.ones(21)
    confidence[hand_utils.WRIST] = 0.0
    ratio = hand_utils.calculate_extension_ratio(hand, confidence, 0.3)
    assert np.isnan(ratio)


def test_nan_coordinates_give_nan():
    hand = np.full((21, 2), np.nan)
    ratio = hand_utils.calculate_extension_ratio(hand, np.ones(21), 0.3)
    assert np.isnan(ratio)


def _detector(hands: np.ndarray, threshold=1.2, min_duration=0.0, fps=10):
    """Build a HandClosure without the toolbox wiring: only what compute() reads."""
    from types import SimpleNamespace

    from nicetoolbox.detectors.feature_detectors.hand_closure.hand_closure import HandClosure
    from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes

    n_frames = hands.shape[2]
    labels = [f"left_hand_{i}" for i in range(21)] + [f"right_hand_{i}" for i in range(21)]
    axes = NpzArrayAxes(
        subjects=["client"],
        cameras=["view_front"],
        frames=[str(i) for i in range(n_frames)],
        labels=labels,
        data=["coordinate_x", "coordinate_y", "confidence_score"],
    )
    detector = object.__new__(HandClosure)
    detector.loaded_inputs = {"hand_joints_2d": SimpleNamespace(array=NpzArray(data=hands, axes=axes), axes=axes)}
    detector.camera_names = ["view_front"]
    detector.threshold = threshold
    detector.min_confidence = 0.3
    detector.min_duration = min_duration
    detector.hand_indices = hand_utils.resolve_hand_indices(labels)
    detector.data = SimpleNamespace(fps=fps)
    return detector


def _session(left_scales, right_scales):
    """(1 subject, 1 camera, frames, 42 points, [x, y, conf]) from per-frame finger scales."""
    frames = []
    for left, right in zip(left_scales, right_scales):
        points = np.concatenate([_hand(left), _hand(right)])
        frames.append(np.concatenate([points, np.ones((42, 1))], axis=-1))
    return np.stack(frames)[np.newaxis, np.newaxis]


def test_compute_marks_fist_frames():
    hands = _session(left_scales=[1.9, 1.0, 1.0], right_scales=[1.9, 1.9, 1.05])
    out = _detector(hands).compute()

    ratio = out.get("hand_closure", "extension_ratio")
    state = out.get("hand_closure", "fist_state")
    assert ratio.axes.labels == ["left_hand", "right_hand"]
    np.testing.assert_allclose(ratio.data[0, 0, :, 0], [1.9, 1.0, 1.0])
    np.testing.assert_array_equal(state.data[0, 0], [[0, 0], [1, 0], [1, 1]])


def test_compute_min_duration_drops_short_fists():
    # left: a 1-frame fist then a 3-frame fist; at 10 fps only runs >= 0.3 s survive
    hands = _session(left_scales=[1.0, 1.9, 1.0, 1.0, 1.0], right_scales=[1.9] * 5)
    out = _detector(hands, min_duration=0.3, fps=10).compute()
    np.testing.assert_array_equal(out.get("hand_closure", "fist_state").data[0, 0, :, 0], [0, 0, 1, 1, 1])


def test_compute_keeps_missing_hand_as_nan():
    hands = _session(left_scales=[1.0, 1.0], right_scales=[1.9, 1.9])
    hands[0, 0, 1, :21, 2] = 0.0  # left hand undetected in frame 1
    state = _detector(hands).compute().get("hand_closure", "fist_state").data
    assert state[0, 0, 0, 0] == 1.0
    assert np.isnan(state[0, 0, 1, 0])
