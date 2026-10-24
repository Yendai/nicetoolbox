"""
Hand Closure (fist) feature detector class.
"""

import logging

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, ArraySchema, BooleanSchema
from nicetoolbox_core.data.loaded_array import select_array

from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from ..eye_closure_threshold.utils import filter_duration
from . import utils as hand_utils

# One finger extension ratio per hand, per subject/camera/frame (axis3 = hand side, no data axis).
HAND_EXTENSION_RATIO = ArraySchema(labels_columns=hand_utils.HAND_SIDES)
# Fist state per hand: 1 = fist, 0 = open, NaN where the hand was not seen.
FIST_STATE = BooleanSchema(labels_columns=hand_utils.HAND_SIDES)


class HandClosure(BaseFeature):
    """
    Detects a closed hand (fist) from upstream 2D hand keypoints.

    For each hand it computes a finger extension ratio (see utils.calculate_extension_ratio)
    and marks a fist where the ratio falls below `threshold`. With min_duration set, only fists
    held at least that long are kept, which filters out brief grasping or tracking noise.

    The default threshold is a starting guess. It should be tuned against annotated sessions
    with the evaluation module before the state is trusted.

    Output is (subjects, cameras, frames, [left_hand, right_hand]) for both the ratio and the state.
    """

    components = ["hand_closure"]  # TODO: delete me
    algorithm_type = "hand_closure"

    inputs = [
        NpzDetectorInput("hand_joints", "hand_joints_2d", schema=VECTOR_2D_CONF_PER_LABEL),
    ]
    outputs = [
        NpzDetectorOutput("hand_closure", "extension_ratio", schema=HAND_EXTENSION_RATIO),
        NpzDetectorOutput("hand_closure", "fist_state", schema=FIST_STATE),
    ]

    def _initialize_detector(self) -> None:
        self.camera_names = self.detector_config.camera_names
        self.threshold = self.detector_config.threshold
        self.min_confidence = self.detector_config.min_confidence
        self.min_duration = self.detector_config.min_duration

        labels = self.loaded_inputs["hand_joints_2d"].axes.labels
        self.hand_indices = hand_utils.resolve_hand_indices(labels)

    def compute(self) -> DetectorOutput:
        """Compute the extension ratio and fist state for every subject/camera/frame and hand.

        Returns a DetectorOutput carrying both arrays under the hand_closure component;
        validation and saving are handled by BaseFeature.run().
        """
        # Restrict to the configured cameras; raises if the upstream lacks any of them.
        hands = select_array(self.loaded_inputs["hand_joints_2d"].array, cameras=self.camera_names)
        axes = hands.axes
        coords = hands.data[..., :2]
        confidence = hands.data[..., 2]

        ratios = []
        for side in hand_utils.HAND_SIDES:
            idx = self.hand_indices[side]
            side_coords, side_confidence = coords[:, :, :, idx, :], confidence[:, :, :, idx]
            ratios.append(hand_utils.calculate_extension_ratio(side_coords, side_confidence, self.min_confidence))
        extension_ratio = np.stack(ratios, axis=-1)

        # Float state so a missing hand stays NaN rather than collapsing into "open".
        missing = np.isnan(extension_ratio)
        fist_state = np.where(extension_ratio < self.threshold, 1.0, 0.0)
        if self.min_duration > 0.0:
            fist_state = filter_duration(fist_state, self.data.fps, self.min_duration, axis=2)
        fist_state[missing] = np.nan

        out_axes = HAND_EXTENSION_RATIO.make_axes(axes.subjects, axes.cameras, axes.frames)
        out = DetectorOutput()
        out.add("hand_closure", "extension_ratio", data=extension_ratio, axes=out_axes)
        out.add("hand_closure", "fist_state", data=fist_state, axes=out_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Plot the extension ratio with the fist intervals shaded, one figure per camera."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        ratio = out.get("hand_closure", "extension_ratio")
        state = out.get("hand_closure", "fist_state")
        hand_utils.plot_hand_closure(
            viz_folder=self.viz_folder,
            extension_ratio=ratio.data,
            fist_state=state.data,
            subjects_descr=self.subjects_descr,
            camera_names=ratio.axes.cameras,
            threshold=self.threshold,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")
