"""
Utility functions for calculating and visualizing arm and leg crossing.
"""

import os

import numpy as np

# Body keypoints the detector reads, by their label in the body_joints axis (COCO-WholeBody names).
SHOULDERS = ("left_shoulder", "right_shoulder")
HIPS = ("left_hip", "right_hip")
# Left/right keypoint pairs whose left-right order flips when a limb crosses the body.
ARM_PAIRS = (("left_wrist", "right_wrist"),)
LEG_PAIRS = (("left_knee", "right_knee"), ("left_ankle", "right_ankle"))
LIMBS = ("arms", "legs")
REQUIRED_POINTS = SHOULDERS + HIPS + tuple(name for pair in ARM_PAIRS + LEG_PAIRS for name in pair)


def resolve_body_indices(labels: list[str]) -> dict[str, int]:
    """
    Find the position of every keypoint the detector reads in a body_joints label axis.

    Returns:
        dict: keypoint name -> position into the label axis.

    Raises:
        ValueError: If any of the required keypoints is missing.
    """
    position = {name: i for i, name in enumerate(labels)}
    missing = [name for name in REQUIRED_POINTS if name not in position]
    if missing:
        raise ValueError(
            f"body_joints input is missing {missing}. Expected COCO-style body keypoint names, "
            "such as rtmpose_l_wholebody produces."
        )
    return {name: position[name] for name in REQUIRED_POINTS}


def calculate_crossing_scores(
    points: dict[str, np.ndarray], min_width_ratio: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate how far the arms and the legs are crossed over the body.

    The body's left-right axis is the line from the right shoulder to the left shoulder. For a
    left/right pair of points (two wrists, two knees, two ankles) the score is how far the right
    point lies to the left of the left point along that axis, in shoulder widths:

        score = dot(right_point - left_point, unit(left_shoulder - right_shoulder)) / shoulder_width

    Uncrossed limbs keep each point on its own side, so the score is negative (wrists resting on
    the thighs are about -1). When the limbs cross the body's midline, the order flips and the
    score turns positive. Folded arms put each wrist by the opposite elbow, about +0.5 to +1.
    Projecting on the shoulder axis means a tilted or leaning body is still measured across
    itself, and dividing by the shoulder width makes the score independent of camera distance.

    The leg score is the larger of the knee and the ankle score, so it catches both a knee
    crossed over the other knee (the ankle swings across) and ankles crossed under the chair.

    Args:
        points (dict): keypoint name -> (..., 2) coordinates, NaN where the point was not seen.
        min_width_ratio (float): Shoulder width divided by torso length (shoulder midpoint to
            hip midpoint) below which the frame gives NaN. A person turned sideways has almost
            no shoulder width, which would blow up the score. Frames without hips are not checked.

    Returns:
        tuple: (arm_score, leg_score), each of shape points[...].shape[:-1].
    """
    across = points["left_shoulder"] - points["right_shoulder"]
    width = np.linalg.norm(across, axis=-1)
    width = np.where(width == 0, np.nan, width)
    unit = across / width[..., np.newaxis]

    def pair_score(left_name, right_name):
        offset = points[right_name] - points[left_name]
        return np.sum(offset * unit, axis=-1) / width

    arm_score = pair_score(*ARM_PAIRS[0])
    leg_scores = np.stack([pair_score(left, right) for left, right in LEG_PAIRS], axis=-1)
    # Larger of knee and ankle; NaN only when neither pair was seen.
    seen = ~np.isnan(leg_scores)
    leg_score = np.where(seen.any(axis=-1), np.max(np.where(seen, leg_scores, -np.inf), axis=-1), np.nan)

    if min_width_ratio > 0.0:
        shoulder_mid = (points["left_shoulder"] + points["right_shoulder"]) / 2
        hip_mid = (points["left_hip"] + points["right_hip"]) / 2
        torso = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
        sideways = width / torso < min_width_ratio  # False where the hips are missing (NaN)
        arm_score = np.where(sideways, np.nan, arm_score)
        leg_score = np.where(sideways, np.nan, leg_score)

    return arm_score, leg_score


def plot_body_crossing(viz_folder, crossing_score, crossing_state, subjects_descr, camera_names, margin) -> None:
    """Plot the per-frame arm and leg crossing scores with crossed intervals shaded, per camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_subj = len(subjects_descr)
    frames = np.arange(crossing_score.shape[2])
    for cam_idx, camera_name in enumerate(camera_names):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)
        for subject_idx, subject_name in enumerate(subjects_descr):
            ax = axes[subject_idx, 0]
            for limb_idx, (limb, color) in enumerate(zip(LIMBS, ("purple", "orange"))):
                ax.plot(frames, crossing_score[subject_idx, cam_idx, :, limb_idx], color=color, label=limb)
                crossed = crossing_state[subject_idx, cam_idx, :, limb_idx] == 1.0
                ax.fill_between(
                    frames, 0, 1, where=crossed, color=color, alpha=0.15, transform=ax.get_xaxis_transform()
                )
            ax.axhline(margin, color="red", linestyle="--", label=f"margin {margin}")
            ax.set_ylabel("Crossing score (shoulder widths)")
            ax.set_title(f"Arm and leg crossing - {subject_name} ({camera_name}), shaded = crossed")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(os.path.join(viz_folder, f"body_crossing_{camera_name}.png"))
        plt.close(fig)
