from pathlib import Path
from types import SimpleNamespace

import numpy as np
import toml

from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_instances_configs import HandFidgetingConfig
from nicetoolbox.detectors.feature_detectors.hand_closure import utils as hand_utils
from nicetoolbox.detectors.feature_detectors.hand_fidgeting import utils as fidget_utils
from nicetoolbox.detectors.feature_detectors.hand_fidgeting.hand_fidgeting import HandFidgeting
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes

LABELS = [f"left_hand_{i}" for i in range(21)] + [f"right_hand_{i}" for i in range(21)]
DETECTORS_CONFIG = Path(__file__).parents[3] / "configs" / "detectors_config.toml"
FPS = 10


def _hand(size: float = 10.0) -> np.ndarray:
    """A still 21-point hand: wrist at the origin, middle knuckle `size` above it."""
    coords = np.zeros((21, 2))
    for point in range(1, 21):
        finger, step = divmod(point - 1, 4)
        coords[point] = [(finger - 2) * 0.3 * size, -(1.0 + 0.4 * step) * size]
    coords[9] = [0.0, -size]  # middle knuckle
    return coords


def _track(n_frames: int, finger_wiggle=0.0, wrist_drift=0.0, size=10.0) -> np.ndarray:
    """(frames, 21, 2): fingertips wiggling sideways and/or the whole hand drifting right.

    finger_wiggle is the fingertips' peak sideways offset in hand lengths (alternating sign each
    frame); wrist_drift is the whole hand's speed in hand lengths per second.
    """
    frames = []
    for t in range(n_frames):
        hand = _hand(size)
        tips = [4, 8, 12, 16, 20]
        hand[tips, 0] += finger_wiggle * size * (1 if t % 2 else -1)
        hand += [wrist_drift * size * t / FPS, 0.0]
        frames.append(hand)
    return np.stack(frames)


def test_rolling_nanmean_trailing_and_centered():
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    trailing = fidget_utils.rolling_nanmean(values, 3, axis=0, min_fraction=0.5, centered=False)
    # frame 0 has only one valid frame of the three it needs half of
    np.testing.assert_allclose(trailing, [np.nan, 1.5, 1.5, 3.0, 4.5])
    centered = fidget_utils.rolling_nanmean(values, 3, axis=0, min_fraction=0.5, centered=True)
    np.testing.assert_allclose(centered, [1.5, 1.5, 3.0, 4.5, 4.5])


def test_rolling_nanmean_needs_enough_valid_frames():
    values = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    out = fidget_utils.rolling_nanmean(values, 4, axis=0, min_fraction=0.5, centered=False)
    np.testing.assert_allclose(out, [np.nan, np.nan, np.nan, np.nan, np.nan])


def _motion(track, smooth_frames=1, min_points=15):
    coords = track[np.newaxis, np.newaxis]  # one subject, one camera
    finger, hand = fidget_utils.calculate_hand_motion(coords, FPS, smooth_frames, min_points)
    return finger[0, 0], hand[0, 0]


def test_still_hand_has_no_motion():
    finger, hand = _motion(_track(5))
    assert np.isnan(finger[0]) and np.isnan(hand[0])  # no previous frame
    np.testing.assert_allclose(finger[1:], 0.0, atol=1e-9)
    np.testing.assert_allclose(hand[1:], 0.0, atol=1e-9)


def test_moving_the_whole_hand_is_hand_speed_not_finger_speed():
    finger, hand = _motion(_track(5, wrist_drift=3.0))
    np.testing.assert_allclose(finger[1:], 0.0, atol=1e-9)
    np.testing.assert_allclose(hand[1:], 3.0)


def test_wiggling_fingers_is_finger_speed_in_hand_lengths():
    finger, hand = _motion(_track(5, finger_wiggle=0.1))
    # 5 of 20 finger points move 0.2 hand lengths per frame: mean 0.05 * 10 fps
    np.testing.assert_allclose(finger[1:], 0.5)
    np.testing.assert_allclose(hand[1:], 0.0)


def test_motion_does_not_depend_on_camera_distance():
    near, _ = _motion(_track(5, finger_wiggle=0.1, size=40.0))
    far, _ = _motion(_track(5, finger_wiggle=0.1, size=8.0))
    np.testing.assert_allclose(near, far)


def test_smoothing_damps_frame_to_frame_jitter():
    raw, _ = _motion(_track(9, finger_wiggle=0.1))
    smoothed, _ = _motion(_track(9, finger_wiggle=0.1), smooth_frames=3)
    assert np.nanmean(smoothed) < np.nanmean(raw) / 2


def test_too_few_points_gives_nan():
    track = _track(3)
    track[1, 1:12] = np.nan  # 11 of 20 finger points lost
    finger, _ = _motion(track, min_points=15)
    assert np.isnan(finger[1]) and np.isnan(finger[2])


def _session(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """(1 subject, 1 camera, frames, 42, [x, y, conf]) from a left and a right hand track."""
    points = np.concatenate([left, right], axis=1)
    return np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)[np.newaxis, np.newaxis]


def _loaded_inputs(hands: np.ndarray) -> dict:
    axes = NpzArrayAxes(
        subjects=["client"],
        cameras=["view_front"],
        frames=[str(i) for i in range(hands.shape[2])],
        labels=LABELS,
        data=["coordinate_x", "coordinate_y", "confidence_score"],
    )
    return {"hand_joints_2d": SimpleNamespace(array=NpzArray(data=hands, axes=axes), axes=axes)}


def _detector(hands, threshold=0.3, max_hand_speed=2.0, window_duration=0.4, min_duration=0.0):
    """Build a HandFidgeting without the toolbox wiring: only what compute() reads."""
    detector = object.__new__(HandFidgeting)
    detector.loaded_inputs = _loaded_inputs(hands)
    detector.camera_names = ["view_front"]
    detector.threshold = threshold
    detector.max_hand_speed = max_hand_speed
    detector.window_duration = window_duration
    detector.smooth_duration = 0.0
    detector.min_confidence = 0.3
    detector.min_points = 15
    detector.min_duration = min_duration
    detector.max_gap_duration = 0.0
    detector.hand_indices = hand_utils.resolve_hand_indices(LABELS)
    detector.data = SimpleNamespace(fps=FPS)
    return detector


def test_compute_marks_fidgeting_hand_only():
    still = _track(10)
    fidget = np.concatenate([_track(4), _track(6, finger_wiggle=0.1)])
    out = _detector(_session(fidget, still)).compute()

    out.validate(HandFidgeting.outputs)  # what BaseFeature.run() checks before saving

    score = out.get("hand_fidgeting", "fidget_score")
    state = out.get("hand_fidgeting", "fidget_state")
    assert score.axes.labels == ["left_hand", "right_hand"]
    # frames 0 and 1 have too little history for the 4-frame window. Finger speed is 0.25 at
    # frame 4 (half a wiggle) and 0.5 after, so the window mean first reaches 0.3 at frame 6.
    np.testing.assert_array_equal(state.data[0, 0, :, 0], [np.nan, np.nan, 0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(state.data[0, 0, 2:, 1], [0] * 8)


def test_compute_gesturing_hand_is_not_fidgeting():
    gesture = _track(10, finger_wiggle=0.1, wrist_drift=5.0)
    still = _track(10)
    state = _detector(_session(gesture, still)).compute().get("hand_fidgeting", "fidget_state").data
    np.testing.assert_array_equal(state[0, 0, 2:, 0], [0] * 8)
    # with the wrist check off it counts
    state = _detector(_session(gesture, still), max_hand_speed=0.0).compute()
    assert np.nansum(state.get("hand_fidgeting", "fidget_state").data[0, 0, :, 0]) > 0


def test_compute_min_duration_drops_short_bursts():
    burst = np.concatenate([_track(3, finger_wiggle=0.1), _track(7)])
    hands = _session(burst, _track(10))
    short = _detector(hands, window_duration=0.1).compute().get("hand_fidgeting", "fidget_state").data
    assert np.nansum(short[0, 0, :, 0]) > 0
    kept = _detector(hands, window_duration=0.1, min_duration=0.5).compute()
    assert np.nansum(kept.get("hand_fidgeting", "fidget_state").data[0, 0, :, 0]) == 0


def test_toml_section_parses_and_initializes_detector():
    """The shipped [algorithms.hand_fidgeting] section reaches the detector's fields intact."""
    section = toml.load(DETECTORS_CONFIG)["algorithms"]["hand_fidgeting"]
    config = DetectorsConfig.model_validate({"algorithms": {"hand_fidgeting": section}}).algorithms["hand_fidgeting"]
    assert isinstance(config, HandFidgetingConfig)

    detector = object.__new__(HandFidgeting)
    detector.detector_config = config
    detector.loaded_inputs = _loaded_inputs(_session(_track(1), _track(1)))
    detector._initialize_detector()

    for field in (
        "camera_names", "threshold", "max_hand_speed", "window_duration", "smooth_duration",
        "min_confidence", "min_points", "min_duration", "max_gap_duration",
    ):  # fmt: skip
        assert getattr(detector, field) == section[field], field
    assert detector.hand_indices["right_hand"] == list(range(21, 42))
