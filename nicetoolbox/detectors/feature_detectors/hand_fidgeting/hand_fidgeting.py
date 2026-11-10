"""
Hand and finger fidgeting feature detector class.
"""

import logging

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, ArraySchema, BooleanSchema
from nicetoolbox_core.data.loaded_array import select_array

from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from ..eye_closure_threshold.utils import filter_duration
from ..hand_closure import utils as hand_utils
from . import utils as fidget_utils

# One value per hand, per subject/camera/frame (axis3 = hand side, no data axis).
PER_HAND = ArraySchema(labels_columns=hand_utils.HAND_SIDES)
# Fidgeting state per hand: 1 = fidgeting, 0 = still, NaN where the hand was not seen.
FIDGET_STATE = BooleanSchema(labels_columns=hand_utils.HAND_SIDES)


class HandFidgeting(BaseFeature):
    """
    Detects sustained small hand and finger movements (fidgeting) from upstream 2D hand keypoints.

    Per hand and frame it measures how fast the fingers move relative to the wrist, and how fast
    the wrist itself moves (see utils.calculate_hand_motion). Both are averaged over a trailing
    window of `window_duration` seconds, because fidgeting is a pattern over time rather than a
    single movement. A hand is fidgeting where the windowed finger motion reaches `threshold`
    while the windowed wrist motion stays at or below `max_hand_speed`: busy fingers on a hand
    that stays roughly in place. The wrist limit keeps gesturing and reaching from counting.

    Gaps of at most max_gap_duration between two equal states are filled, and with min_duration
    set only fidgeting that lasts at least that long is kept.

    The default thresholds are starting guesses. They should be tuned against annotated sessions
    (still hands set the noise floor) with the evaluation module before the state is trusted.

    Output is (subjects, cameras, frames, [left_hand, right_hand]) for the windowed finger motion
    (fidget_score), the windowed wrist motion (hand_speed) and the state (fidget_state).
    """

    components = ["hand_fidgeting"]  # TODO: delete me
    algorithm_type = "hand_fidgeting"

    inputs = [
        NpzDetectorInput("hand_joints", "hand_joints_2d", schema=VECTOR_2D_CONF_PER_LABEL),
    ]
    outputs = [
        NpzDetectorOutput("hand_fidgeting", "fidget_score", schema=PER_HAND),
        NpzDetectorOutput("hand_fidgeting", "hand_speed", schema=PER_HAND),
        NpzDetectorOutput("hand_fidgeting", "fidget_state", schema=FIDGET_STATE),
    ]

    def _initialize_detector(self) -> None:
        self.camera_names = self.detector_config.camera_names
        self.threshold = self.detector_config.threshold
        self.max_hand_speed = self.detector_config.max_hand_speed
        self.window_duration = self.detector_config.window_duration
        self.smooth_duration = self.detector_config.smooth_duration
        self.min_confidence = self.detector_config.min_confidence
        self.min_points = self.detector_config.min_points
        self.min_duration = self.detector_config.min_duration
        self.max_gap_duration = self.detector_config.max_gap_duration

        labels = self.loaded_inputs["hand_joints_2d"].axes.labels
        self.hand_indices = hand_utils.resolve_hand_indices(labels)

    def compute(self) -> DetectorOutput:
        """Compute the fidget score, hand speed and fidget state for every subject/camera/frame and hand.

        Returns a DetectorOutput carrying the arrays under the hand_fidgeting component;
        validation and saving are handled by BaseFeature.run().
        """
        # Restrict to the configured cameras; raises if the upstream lacks any of them.
        hands = select_array(self.loaded_inputs["hand_joints_2d"].array, cameras=self.camera_names)
        axes = hands.axes
        fps = self.data.fps
        seen = hands.data[..., 2] >= self.min_confidence
        coords = np.where(seen[..., np.newaxis], hands.data[..., :2], np.nan)

        smooth_frames = max(int(round(self.smooth_duration * fps)), 1)
        window_frames = max(int(round(self.window_duration * fps)), 1)

        scores, speeds = [], []
        for side in hand_utils.HAND_SIDES:
            side_coords = coords[:, :, :, self.hand_indices[side], :]
            finger_speed, hand_speed = fidget_utils.calculate_hand_motion(
                side_coords, fps, smooth_frames, self.min_points
            )
            # Trailing windows: the value at a frame only depends on the past, as it would live.
            scores.append(fidget_utils.rolling_nanmean(finger_speed, window_frames, 2, 0.5, centered=False))
            speeds.append(fidget_utils.rolling_nanmean(hand_speed, window_frames, 2, 0.5, centered=False))
        fidget_score = np.stack(scores, axis=-1)
        hand_speed = np.stack(speeds, axis=-1)

        fidgeting = fidget_score >= self.threshold
        if self.max_hand_speed > 0.0:
            # A missing wrist speed (NaN) does not veto: the comparison is only False when it is too high.
            fidgeting &= ~(hand_speed > self.max_hand_speed)
        # Float state so a missing hand stays NaN rather than collapsing into "still".
        fidget_state = np.where(fidgeting, 1.0, 0.0)
        fidget_state[np.isnan(fidget_score)] = np.nan

        # Bridge brief dropouts (1, NaN, 1 -> 1, 1, 1) so they don't split a fidgeting run in two.
        if self.max_gap_duration > 0.0:
            fidget_state = hand_utils.fill_short_gaps(fidget_state, fps, self.max_gap_duration, axis=2)

        # Duration filtering needs a clean 0/1 signal: treat the remaining gaps as 0, then restore them.
        missing = np.isnan(fidget_state)
        if self.min_duration > 0.0:
            fidget_state = filter_duration(np.nan_to_num(fidget_state, nan=0.0), fps, self.min_duration, axis=2)
        fidget_state[missing] = np.nan

        out_axes = PER_HAND.make_axes(axes.subjects, axes.cameras, axes.frames)
        out = DetectorOutput()
        out.add("hand_fidgeting", "fidget_score", data=fidget_score, axes=out_axes)
        out.add("hand_fidgeting", "hand_speed", data=hand_speed, axes=out_axes)
        out.add("hand_fidgeting", "fidget_state", data=fidget_state, axes=out_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Plot the windowed finger motion with the fidgeting intervals shaded, one figure per camera."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        score = out.get("hand_fidgeting", "fidget_score")
        state = out.get("hand_fidgeting", "fidget_state")
        fidget_utils.plot_hand_fidgeting(
            viz_folder=self.viz_folder,
            fidget_score=score.data,
            fidget_state=state.data,
            subjects_descr=self.subjects_descr,
            camera_names=score.axes.cameras,
            threshold=self.threshold,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")
