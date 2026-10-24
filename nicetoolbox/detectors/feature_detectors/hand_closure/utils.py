"""
Utility functions for calculating and visualizing hand closure (fist) scores.
"""

import os

import numpy as np

# Point order of one 21-point hand (COCO-WholeBody / OpenPose hand layout):
# 0 = wrist, then four points per finger from the knuckle (MCP) outwards to the tip.
WRIST = 0
# (knuckle, tip) per finger. The thumb is left out: it folds across the palm rather than
# curling towards the wrist, so its tip-to-wrist distance barely changes in a fist.
FINGERS = {
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}
HAND_SIDES = ("left_hand", "right_hand")
POINTS_PER_HAND = 21


def resolve_hand_indices(labels: list[str]) -> dict[str, list[int]]:
    """
    Find the positions of each hand's 21 points in a hand_joints label axis.

    Labels follow the toolbox convention `<side>_<index>`, e.g. `left_hand_0` ... `right_hand_20`.

    Returns:
        dict: side name -> 21 positions into the label axis, ordered by point index.

    Raises:
        ValueError: If any of the 42 expected labels is missing.
    """
    position = {name: i for i, name in enumerate(labels)}
    indices = {}
    for side in HAND_SIDES:
        expected = [f"{side}_{i}" for i in range(POINTS_PER_HAND)]
        missing = [name for name in expected if name not in position]
        if missing:
            raise ValueError(
                f"hand_joints input is missing {len(missing)} '{side}' labels (e.g. {missing[:3]}). "
                "Expected a 21-point hand layout per side, such as rtmpose_l_wholebody produces."
            )
        indices[side] = [position[name] for name in expected]
    return indices


def calculate_extension_ratio(hand_coords: np.ndarray, confidence: np.ndarray, min_confidence: float) -> np.ndarray:
    """
    Calculate how extended the four fingers of a hand are.

    For each finger the ratio is distance(tip, wrist) / distance(knuckle, wrist). An open hand
    gives roughly 1.7 to 2.0; a clenched fist brings the tips back to about the knuckles, near 1.0.
    The ratio is scale invariant, so it does not depend on how far the person sits from the camera.

    Args:
        hand_coords (np.ndarray): (..., 21, 2) point coordinates of one hand.
        confidence (np.ndarray): (..., 21) keypoint confidence scores.
        min_confidence (float): Points below this confidence count as missing.

    Returns:
        np.ndarray: Mean ratio over the four fingers, shape hand_coords.shape[:-2]. NaN where
        the wrist is missing, or every finger is missing its knuckle or tip.
    """
    coords = np.where((confidence >= min_confidence)[..., np.newaxis], hand_coords, np.nan)
    wrist = coords[..., WRIST, :]

    ratios = []
    for knuckle_idx, tip_idx in FINGERS.values():
        knuckle_dist = np.linalg.norm(coords[..., knuckle_idx, :] - wrist, axis=-1)
        tip_dist = np.linalg.norm(coords[..., tip_idx, :] - wrist, axis=-1)
        # A zero knuckle distance would divide to inf; treat it as missing instead.
        knuckle_dist = np.where(knuckle_dist == 0, np.nan, knuckle_dist)
        ratios.append(tip_dist / knuckle_dist)

    ratios = np.stack(ratios, axis=-1)
    # Average the fingers that were seen; a frame with no usable finger stays NaN.
    counts = np.sum(~np.isnan(ratios), axis=-1)
    sums = np.nansum(ratios, axis=-1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def plot_hand_closure(viz_folder, extension_ratio, fist_state, subjects_descr, camera_names, threshold) -> None:
    """Plot the per-frame extension ratio and fist state for each camera and subject."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_subj = len(subjects_descr)
    for cam_idx, camera_name in enumerate(camera_names):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)
        for subject_idx, subject_name in enumerate(subjects_descr):
            ax = axes[subject_idx, 0]
            for side_idx, (side, color) in enumerate(zip(HAND_SIDES, ("blue", "green"))):
                frames = np.arange(extension_ratio.shape[2])
                ax.plot(frames, extension_ratio[subject_idx, cam_idx, :, side_idx], color=color, label=side)
                fist = fist_state[subject_idx, cam_idx, :, side_idx] == 1.0
                ax.fill_between(frames, 0, 1, where=fist, color=color, alpha=0.15, transform=ax.get_xaxis_transform())
            ax.axhline(threshold, color="red", linestyle="--", label=f"threshold {threshold}")
            ax.set_ylabel("Finger extension ratio")
            ax.set_title(f"Hand closure - {subject_name} ({camera_name}), shaded = fist")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(os.path.join(viz_folder, f"hand_closure_{camera_name}.png"))
        plt.close(fig)
