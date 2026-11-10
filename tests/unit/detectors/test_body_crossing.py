from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_instances_configs import BodyCrossingConfig
from nicetoolbox.detectors.feature_detectors.body_crossing import utils as crossing_utils
from nicetoolbox.detectors.feature_detectors.body_crossing.body_crossing import BodyCrossing
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes

# The 23 body + foot labels rtmpose_l_wholebody writes for body_joints.
LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
]  # fmt: skip
DETECTORS_CONFIG = Path(__file__).parents[3] / "configs" / "detectors_config.toml"

# A seated person facing the camera, image x to the right, y down. The subject's left side is on
# the image right. Shoulder width 2, torso length 2.5.
SEATED = {
    "left_shoulder": (1.0, 0.0),
    "right_shoulder": (-1.0, 0.0),
    "left_elbow": (1.3, 1.3),
    "right_elbow": (-1.3, 1.3),
    "left_wrist": (1.0, 2.5),
    "right_wrist": (-1.0, 2.5),
    "left_hip": (0.6, 2.5),
    "right_hip": (-0.6, 2.5),
    "left_knee": (0.8, 3.0),
    "right_knee": (-0.8, 3.0),
    "left_ankle": (0.8, 4.5),
    "right_ankle": (-0.8, 4.5),
}


def _pose(**moved) -> dict[str, np.ndarray]:
    """The seated pose with some keypoints moved."""
    return {name: np.array(moved.get(name, xy), dtype=float) for name, xy in SEATED.items()}


FOLDED_ARMS = {"left_wrist": (-0.9, 1.4), "right_wrist": (0.9, 1.5)}
# Left knee over the right knee: the left ankle swings across to the right side.
CROSSED_LEGS = {"left_knee": (-0.3, 2.9), "left_ankle": (-1.2, 4.3)}
CROSSED_ANKLES = {"left_ankle": (-0.3, 4.5), "right_ankle": (0.3, 4.5)}


def test_resolve_body_indices_maps_names():
    indices = crossing_utils.resolve_body_indices(LABELS)
    assert indices["left_wrist"] == 9
    assert indices["right_ankle"] == 16


def test_resolve_body_indices_raises_on_missing_labels():
    with pytest.raises(ValueError, match="left_knee"):
        crossing_utils.resolve_body_indices([name for name in LABELS if name != "left_knee"])


def test_uncrossed_scores_are_negative():
    arm, leg = crossing_utils.calculate_crossing_scores(_pose())
    np.testing.assert_allclose(arm, -1.0)  # wrists two shoulder halves apart
    np.testing.assert_allclose(leg, -0.8)  # the knees and ankles are both 1.6 apart


def test_folded_arms_score_positive():
    arm, leg = crossing_utils.calculate_crossing_scores(_pose(**FOLDED_ARMS))
    np.testing.assert_allclose(arm, 0.9)
    assert leg < 0


def test_crossed_knees_and_crossed_ankles_score_positive():
    _, knee_over_knee = crossing_utils.calculate_crossing_scores(_pose(**CROSSED_LEGS))
    _, ankles = crossing_utils.calculate_crossing_scores(_pose(**CROSSED_ANKLES))
    assert knee_over_knee > 0.1  # the ankle carries it; the knees alone are still in order
    np.testing.assert_allclose(ankles, 0.3)


def test_hands_clasped_in_the_lap_stay_below_margin():
    arm, _ = crossing_utils.calculate_crossing_scores(_pose(left_wrist=(0.05, 2.5), right_wrist=(-0.05, 2.5)))
    assert arm < 0.1


def test_score_ignores_scale_shift_and_tilt():
    pose = _pose(**FOLDED_ARMS)
    angle = np.deg2rad(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = {name: 3.0 * rotation @ xy + 50.0 for name, xy in pose.items()}
    np.testing.assert_allclose(
        crossing_utils.calculate_crossing_scores(moved), crossing_utils.calculate_crossing_scores(pose)
    )


def test_side_on_body_gives_nan():
    side_on = _pose(left_shoulder=(0.2, 0.0), right_shoulder=(-0.2, 0.0))  # width 0.4, torso 2.5
    arm, leg = crossing_utils.calculate_crossing_scores(side_on, min_width_ratio=0.35)
    assert np.isnan(arm) and np.isnan(leg)
    # without hips the check is skipped rather than failing the frame
    side_on["left_hip"][:] = np.nan
    arm, _ = crossing_utils.calculate_crossing_scores(side_on, min_width_ratio=0.35)
    assert not np.isnan(arm)


def test_leg_score_uses_whichever_pair_was_seen():
    pose = _pose(**CROSSED_ANKLES)
    pose["left_knee"][:] = np.nan
    _, leg = crossing_utils.calculate_crossing_scores(pose)
    np.testing.assert_allclose(leg, 0.3)
    pose["left_ankle"][:] = np.nan
    _, leg = crossing_utils.calculate_crossing_scores(pose)
    assert np.isnan(leg)


def _session(*poses) -> np.ndarray:
    """(1 subject, 1 camera, frames, 23 labels, [x, y, conf]) from one pose dict per frame."""
    frames = []
    for pose in poses:
        data = np.zeros((len(LABELS), 3))
        data[:, 2] = 1.0
        for name, xy in pose.items():
            data[LABELS.index(name), :2] = xy
        frames.append(data)
    return np.stack(frames)[np.newaxis, np.newaxis]


def _loaded_inputs(body: np.ndarray) -> dict:
    axes = NpzArrayAxes(
        subjects=["client"],
        cameras=["view_front"],
        frames=[str(i) for i in range(body.shape[2])],
        labels=LABELS,
        data=["coordinate_x", "coordinate_y", "confidence_score"],
    )
    return {"body_joints_2d": SimpleNamespace(array=NpzArray(data=body, axes=axes), axes=axes)}


def _detector(body, margin=0.1, min_duration=0.0, fps=10, max_gap_duration=0.0, min_width_ratio=0.35):
    """Build a BodyCrossing without the toolbox wiring: only what compute() reads."""
    detector = object.__new__(BodyCrossing)
    detector.loaded_inputs = _loaded_inputs(body)
    detector.camera_names = ["view_front"]
    detector.margin = margin
    detector.min_confidence = 0.3
    detector.min_width_ratio = min_width_ratio
    detector.min_duration = min_duration
    detector.max_gap_duration = max_gap_duration
    detector.body_indices = crossing_utils.resolve_body_indices(LABELS)
    detector.data = SimpleNamespace(fps=fps)
    return detector


def test_compute_marks_crossed_frames():
    body = _session(_pose(), _pose(**FOLDED_ARMS), _pose(**FOLDED_ARMS, **CROSSED_ANKLES))
    out = _detector(body).compute()

    out.validate(BodyCrossing.outputs)  # what BaseFeature.run() checks before saving

    score = out.get("body_crossing", "crossing_score")
    state = out.get("body_crossing", "crossing_state")
    assert score.axes.labels == ["arms", "legs"]
    np.testing.assert_allclose(score.data[0, 0, :, 0], [-1.0, 0.9, 0.9])
    np.testing.assert_array_equal(state.data[0, 0], [[0, 0], [1, 0], [1, 1]])


def test_compute_min_duration_drops_passing_hands():
    # arms: crossed for 1 frame, then for 3; at 10 fps only runs >= 0.3 s survive
    folded = _pose(**FOLDED_ARMS)
    body = _session(folded, _pose(), folded, folded, folded)
    state = _detector(body, min_duration=0.3).compute().get("body_crossing", "crossing_state").data
    np.testing.assert_array_equal(state[0, 0, :, 0], [0, 0, 1, 1, 1])


def test_compute_keeps_missing_limbs_as_nan_and_fills_short_gaps():
    folded = _pose(**FOLDED_ARMS)
    body = _session(folded, folded, folded, folded)
    body[0, 0, 1, LABELS.index("left_wrist"), 2] = 0.0  # wrist lost in frame 1

    unfilled = _detector(body).compute().get("body_crossing", "crossing_state").data
    assert np.isnan(unfilled[0, 0, 1, 0])
    filled = _detector(body, max_gap_duration=0.1).compute().get("body_crossing", "crossing_state").data
    np.testing.assert_array_equal(filled[0, 0, :, 0], [1, 1, 1, 1])


def test_toml_section_parses_and_initializes_detector():
    """The shipped [algorithms.body_crossing] section reaches the detector's fields intact."""
    section = toml.load(DETECTORS_CONFIG)["algorithms"]["body_crossing"]
    config = DetectorsConfig.model_validate({"algorithms": {"body_crossing": section}}).algorithms["body_crossing"]
    assert isinstance(config, BodyCrossingConfig)

    detector = object.__new__(BodyCrossing)
    detector.detector_config = config
    detector.loaded_inputs = _loaded_inputs(_session(_pose()))
    detector._initialize_detector()

    assert detector.camera_names == section["camera_names"]
    assert detector.margin == section["margin"]
    assert detector.min_confidence == section["min_confidence"]
    assert detector.min_width_ratio == section["min_width_ratio"]
    assert detector.min_duration == section["min_duration"]
    assert detector.max_gap_duration == section["max_gap_duration"]
    assert detector.body_indices["left_shoulder"] == 5
