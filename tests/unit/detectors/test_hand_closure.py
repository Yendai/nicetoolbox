from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_instances_configs import HandClosureConfig
from nicetoolbox.detectors.feature_detectors.hand_closure import utils as hand_utils
from nicetoolbox.detectors.feature_detectors.hand_closure.hand_closure import HandClosure
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes

LABELS = [f"left_hand_{i}" for i in range(21)] + [f"right_hand_{i}" for i in range(21)]
DETECTORS_CONFIG = Path(__file__).parents[3] / "configs" / "detectors_config.toml"


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


def test_min_fingers_turns_pointing_hand_with_lost_index_into_nan():
    # pointing: index straight, the other three curled
    hand = _hand(1.0)
    index_knuckle, index_tip = hand_utils.FINGERS["index"]
    hand[index_knuckle : index_tip + 1] = _hand(1.9)[index_knuckle : index_tip + 1]
    confidence = np.ones(21)
    confidence[index_tip] = 0.1  # index tip lost

    # averaging the three seen fingers reads the pointing hand as a fist ...
    np.testing.assert_allclose(hand_utils.calculate_extension_ratio(hand, confidence, 0.3, min_fingers=3), 1.0)
    # ... requiring all four makes it unknown instead
    assert np.isnan(hand_utils.calculate_extension_ratio(hand, confidence, 0.3, min_fingers=4))


def test_fill_short_gaps_bridges_equal_neighbours():
    nan = np.nan
    state = np.array(
        [
            [1, nan, 1, 0, nan, 0],  # both gaps bridged
            [1, nan, 0, 0, 0, 0],  # neighbours disagree: stays NaN
            [nan, 1, 1, 1, 1, nan],  # gaps at the edges: stay NaN
            [1, nan, nan, 1, 1, 1],  # 2-frame gap, longer than the 1-frame limit: stays NaN
        ]
    )
    filled = hand_utils.fill_short_gaps(state, fps=10, max_gap_duration=0.1, axis=1)
    np.testing.assert_array_equal(
        filled,
        [
            [1, 1, 1, 0, 0, 0],
            [1, nan, 0, 0, 0, 0],
            [nan, 1, 1, 1, 1, nan],
            [1, nan, nan, 1, 1, 1],
        ],
    )
    assert np.isnan(state[0, 1])  # input untouched


def test_fill_short_gaps_uses_seconds_and_any_axis():
    state = np.array([1, np.nan, np.nan, 1.0])[np.newaxis, :, np.newaxis]  # frames on axis 1
    filled = hand_utils.fill_short_gaps(state, fps=20, max_gap_duration=0.1, axis=1)  # 0.1 s = 2 frames
    np.testing.assert_array_equal(filled[0, :, 0], [1, 1, 1, 1])
    unfilled = hand_utils.fill_short_gaps(state, fps=10, max_gap_duration=0.1, axis=1)  # 0.1 s = 1 frame
    assert np.isnan(unfilled[0, 1:3, 0]).all()


def _detector(hands: np.ndarray, threshold=1.2, min_duration=0.0, fps=10, min_fingers=4, max_gap_duration=0.0):
    """Build a HandClosure without the toolbox wiring: only what compute() reads."""
    detector = object.__new__(HandClosure)
    detector.loaded_inputs = _loaded_inputs(hands)
    detector.camera_names = ["view_front"]
    detector.threshold = threshold
    detector.min_confidence = 0.3
    detector.min_duration = min_duration
    detector.min_fingers = min_fingers
    detector.max_gap_duration = max_gap_duration
    detector.hand_indices = hand_utils.resolve_hand_indices(LABELS)
    detector.data = SimpleNamespace(fps=fps)
    return detector


def _loaded_inputs(hands: np.ndarray) -> dict:
    """The hand_joints_2d input as BaseDetector loads it: (1 subject, 1 camera, frames, 42, 3)."""
    axes = NpzArrayAxes(
        subjects=["client"],
        cameras=["view_front"],
        frames=[str(i) for i in range(hands.shape[2])],
        labels=LABELS,
        data=["coordinate_x", "coordinate_y", "confidence_score"],
    )
    return {"hand_joints_2d": SimpleNamespace(array=NpzArray(data=hands, axes=axes), axes=axes)}


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

    out.validate(HandClosure.outputs)  # what BaseFeature.run() checks before saving

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


def test_compute_fills_one_frame_dropout_inside_a_fist():
    # A fist from frame 1 to 9 with the hand lost in frame 4. Without gap filling the dropout
    # splits it into 0.3 s + 0.5 s runs and the first is dropped by min_duration.
    hands = _session(left_scales=[1.9] + [1.0] * 9, right_scales=[1.9] * 10)
    hands[0, 0, 4, :21, 2] = 0.0

    unfilled = _detector(hands, min_duration=0.5, fps=10).compute().get("hand_closure", "fist_state").data
    np.testing.assert_array_equal(unfilled[0, 0, :, 0], [0, 0, 0, 0, np.nan, 1, 1, 1, 1, 1])

    filled = _detector(hands, min_duration=0.5, fps=10, max_gap_duration=0.1).compute()
    np.testing.assert_array_equal(filled.get("hand_closure", "fist_state").data[0, 0, :, 0], [0] + [1] * 9)


def test_toml_section_parses_and_initializes_detector():
    """The shipped [algorithms.hand_closure] section reaches the detector's fields intact."""
    section = toml.load(DETECTORS_CONFIG)["algorithms"]["hand_closure"]
    config = DetectorsConfig.model_validate({"algorithms": {"hand_closure": section}}).algorithms["hand_closure"]
    assert isinstance(config, HandClosureConfig)

    detector = object.__new__(HandClosure)
    detector.detector_config = config
    detector.loaded_inputs = _loaded_inputs(_session([1.0], [1.0]))
    detector._initialize_detector()

    assert detector.camera_names == section["camera_names"]
    assert detector.threshold == section["threshold"]
    assert detector.min_confidence == section["min_confidence"]
    assert detector.min_duration == section["min_duration"]
    assert detector.min_fingers == section["min_fingers"]
    assert detector.max_gap_duration == section["max_gap_duration"]
    assert detector.hand_indices["right_hand"] == list(range(21, 42))
