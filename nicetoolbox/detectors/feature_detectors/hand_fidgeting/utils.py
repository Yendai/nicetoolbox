"""
Utility functions for calculating and visualizing hand and finger fidgeting.
"""

import os

import numpy as np

from ..hand_closure.utils import HAND_SIDES, WRIST

# The middle finger's knuckle: wrist-to-knuckle distance is the hand's size unit ("hand length").
# Unlike the fingertips, it barely moves when the fingers curl, so it stays steady while fidgeting.
MIDDLE_KNUCKLE = 9


def rolling_nanmean(values: np.ndarray, window: int, axis: int, min_fraction: float, centered: bool) -> np.ndarray:
    """
    Mean over a sliding window of frames, ignoring NaN.

    Args:
        values (np.ndarray): Any shape; NaN marks a missing frame.
        window (int): Window length in frames (at least 1).
        axis (int): The frame axis.
        min_fraction (float): Share of the window that must be valid, or the result is NaN.
        centered (bool): Centre the window on each frame (for smoothing a recording). Otherwise
            the window ends at each frame, so the value only depends on the past, as in a live loop.
            Windows are cut short at the start and end of the recording.

    Returns:
        np.ndarray: Same shape as `values`.
    """
    window = max(int(window), 1)
    moved = np.moveaxis(values, axis, -1)
    valid = ~np.isnan(moved)
    zeros = np.zeros(moved.shape[:-1] + (1,))
    # Cumulative sums with a leading 0, so sum(t0..t1) = csum[t1 + 1] - csum[t0].
    csum = np.concatenate([zeros, np.cumsum(np.where(valid, moved, 0.0), axis=-1)], axis=-1)
    ccount = np.concatenate([zeros, np.cumsum(valid, axis=-1)], axis=-1)

    n_frames = moved.shape[-1]
    frames = np.arange(n_frames)
    if centered:
        start = np.clip(frames - (window - 1) // 2, 0, n_frames)
        end = np.clip(frames + window // 2 + 1, 0, n_frames)
    else:
        start = np.clip(frames - window + 1, 0, n_frames)
        end = frames + 1

    sums = csum[..., end] - csum[..., start]
    counts = ccount[..., end] - ccount[..., start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    mean = np.where(counts >= max(min_fraction * window, 1), mean, np.nan)
    return np.moveaxis(mean, -1, axis)


def calculate_hand_motion(
    hand_coords: np.ndarray, fps: float, smooth_frames: int, min_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-frame finger motion and whole-hand motion for one hand.

    Coordinates are first divided by the hand's median size over the recording (wrist to middle
    knuckle), so speeds are in hand lengths per second and do not depend on camera distance.
    They are then smoothed over `smooth_frames`, because keypoint jitter alone would otherwise
    read as motion.

    - Finger speed: how fast the 20 finger points move relative to the wrist, averaged over the
      points. Moving the whole hand without moving the fingers gives zero.
    - Hand speed: how fast the wrist itself moves. It is high when the person gestures or reaches.

    Args:
        hand_coords (np.ndarray): (subjects, cameras, frames, 21, 2), NaN where a point was not seen.
        fps (float): Frames per second.
        smooth_frames (int): Centered moving-average length applied to the coordinates.
        min_points (int): Finger points (of 20) that must be seen in two consecutive frames for a
            finger speed; fewer gives NaN.

    Returns:
        tuple: (finger_speed, hand_speed), each (subjects, cameras, frames). The first frame is NaN.
    """
    size = np.linalg.norm(hand_coords[..., MIDDLE_KNUCKLE, :] - hand_coords[..., WRIST, :], axis=-1)
    size = np.where(size == 0, np.nan, size)
    with np.errstate(all="ignore"):
        scale = np.nanmedian(size, axis=2, keepdims=True)  # per subject and camera
    coords = hand_coords / scale[..., np.newaxis, np.newaxis]

    coords = rolling_nanmean(coords, smooth_frames, axis=2, min_fraction=0.5, centered=True)
    wrist = coords[..., WRIST, :]
    fingers = np.delete(coords, WRIST, axis=-2) - wrist[..., np.newaxis, :]

    finger_step = np.full(fingers.shape[:-1], np.nan)
    finger_step[:, :, 1:] = np.linalg.norm(fingers[:, :, 1:] - fingers[:, :, :-1], axis=-1)
    counts = np.sum(~np.isnan(finger_step), axis=-1)
    finger_speed = np.nansum(finger_step, axis=-1) / np.maximum(counts, 1) * fps
    finger_speed = np.where(counts >= max(min_points, 1), finger_speed, np.nan)

    hand_speed = np.full(wrist.shape[:-1], np.nan)
    hand_speed[:, :, 1:] = np.linalg.norm(wrist[:, :, 1:] - wrist[:, :, :-1], axis=-1) * fps
    return finger_speed, hand_speed


def plot_hand_fidgeting(viz_folder, fidget_score, fidget_state, subjects_descr, camera_names, threshold) -> None:
    """Plot the windowed finger motion with the fidgeting intervals shaded, one figure per camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_subj = len(subjects_descr)
    frames = np.arange(fidget_score.shape[2])
    for cam_idx, camera_name in enumerate(camera_names):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)
        for subject_idx, subject_name in enumerate(subjects_descr):
            ax = axes[subject_idx, 0]
            for side_idx, (side, color) in enumerate(zip(HAND_SIDES, ("blue", "green"))):
                ax.plot(frames, fidget_score[subject_idx, cam_idx, :, side_idx], color=color, label=side)
                fidgeting = fidget_state[subject_idx, cam_idx, :, side_idx] == 1.0
                ax.fill_between(
                    frames, 0, 1, where=fidgeting, color=color, alpha=0.15, transform=ax.get_xaxis_transform()
                )
            ax.axhline(threshold, color="red", linestyle="--", label=f"threshold {threshold}")
            ax.set_ylabel("Finger motion (hand lengths/s)")
            ax.set_title(f"Hand fidgeting - {subject_name} ({camera_name}), shaded = fidgeting")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(os.path.join(viz_folder, f"hand_fidgeting_{camera_name}.png"))
        plt.close(fig)
