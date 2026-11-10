"""
Arm and leg crossing feature detector class.
"""

import logging

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, ArraySchema, BooleanSchema
from nicetoolbox_core.data.loaded_array import select_array

from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from ..eye_closure_threshold.utils import filter_duration
from ..hand_closure.utils import fill_short_gaps
from . import utils as crossing_utils

# One crossing score per limb pair, per subject/camera/frame (axis3 = arms, legs; no data axis).
CROSSING_SCORE = ArraySchema(labels_columns=crossing_utils.LIMBS)
# Crossed state per limb pair: 1 = crossed, 0 = not crossed, NaN where the limbs were not seen.
CROSSING_STATE = BooleanSchema(labels_columns=crossing_utils.LIMBS)


class BodyCrossing(BaseFeature):
    """
    Detects crossed arms and crossed legs from upstream 2D body keypoints.

    For the arms (wrists) and the legs (knees, ankles) it computes how far the left and the right
    point have swapped sides across the body (see utils.calculate_crossing_scores) and marks the
    limbs crossed where the score is above `margin`. The margin keeps hands clasped in the lap or
    knees pressed together, which sit right at the midline, from flickering in and out. Gaps of
    at most max_gap_duration between two equal states are filled, and with min_duration set only
    crossings held at least that long are kept, so passing a hand across the body does not count.

    The default margin and durations are starting guesses. They should be tuned against
    annotated sessions with the evaluation module before the state is trusted.

    Output is (subjects, cameras, frames, [arms, legs]) for both the score and the state.
    """

    components = ["body_crossing"]  # TODO: delete me
    algorithm_type = "body_crossing"

    inputs = [
        NpzDetectorInput("body_joints", "body_joints_2d", schema=VECTOR_2D_CONF_PER_LABEL),
    ]
    outputs = [
        NpzDetectorOutput("body_crossing", "crossing_score", schema=CROSSING_SCORE),
        NpzDetectorOutput("body_crossing", "crossing_state", schema=CROSSING_STATE),
    ]

    def _initialize_detector(self) -> None:
        self.camera_names = self.detector_config.camera_names
        self.margin = self.detector_config.margin
        self.min_confidence = self.detector_config.min_confidence
        self.min_width_ratio = self.detector_config.min_width_ratio
        self.min_duration = self.detector_config.min_duration
        self.max_gap_duration = self.detector_config.max_gap_duration

        labels = self.loaded_inputs["body_joints_2d"].axes.labels
        self.body_indices = crossing_utils.resolve_body_indices(labels)

    def compute(self) -> DetectorOutput:
        """Compute the crossing score and state for every subject/camera/frame and limb pair.

        Returns a DetectorOutput carrying both arrays under the body_crossing component;
        validation and saving are handled by BaseFeature.run().
        """
        # Restrict to the configured cameras; raises if the upstream lacks any of them.
        body = select_array(self.loaded_inputs["body_joints_2d"].array, cameras=self.camera_names)
        axes = body.axes
        seen = body.data[..., 2] >= self.min_confidence
        coords = np.where(seen[..., np.newaxis], body.data[..., :2], np.nan)
        points = {name: coords[:, :, :, idx, :] for name, idx in self.body_indices.items()}

        arm_score, leg_score = crossing_utils.calculate_crossing_scores(points, self.min_width_ratio)
        crossing_score = np.stack([arm_score, leg_score], axis=-1)

        # Float state so a missing limb stays NaN rather than collapsing into "not crossed".
        crossing_state = np.where(crossing_score > self.margin, 1.0, 0.0)
        crossing_state[np.isnan(crossing_score)] = np.nan

        # Bridge brief dropouts (1, NaN, 1 -> 1, 1, 1) so they don't split a crossing in two.
        if self.max_gap_duration > 0.0:
            crossing_state = fill_short_gaps(crossing_state, self.data.fps, self.max_gap_duration, axis=2)

        # Duration filtering needs a clean 0/1 signal: treat the remaining gaps as 0, then restore them.
        missing = np.isnan(crossing_state)
        if self.min_duration > 0.0:
            crossing_state = filter_duration(
                np.nan_to_num(crossing_state, nan=0.0), self.data.fps, self.min_duration, axis=2
            )
        crossing_state[missing] = np.nan

        out_axes = CROSSING_SCORE.make_axes(axes.subjects, axes.cameras, axes.frames)
        out = DetectorOutput()
        out.add("body_crossing", "crossing_score", data=crossing_score, axes=out_axes)
        out.add("body_crossing", "crossing_state", data=crossing_state, axes=out_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Plot the crossing scores with the crossed intervals shaded, one figure per camera."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        score = out.get("body_crossing", "crossing_score")
        state = out.get("body_crossing", "crossing_state")
        crossing_utils.plot_body_crossing(
            viz_folder=self.viz_folder,
            crossing_score=score.data,
            crossing_state=state.data,
            subjects_descr=self.subjects_descr,
            camera_names=score.axes.cameras,
            margin=self.margin,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")
