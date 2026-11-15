"""
Streaming version of the arm and leg crossing feature detector.

The per-frame score is the batch detector's crossing score, computed with the same utility
function. A cue starts once the limbs have stayed crossed for `min_duration` (see cues.py).
"""

import numpy as np

from ..detectors.feature_detectors.body_crossing import utils as crossing_utils
from .backends import Observation
from .cues import SustainedCue


class LiveBodyCrossing:
    """
    Frame-by-frame crossed arms / crossed legs detection with start/end cues per limb pair.

    Args:
        parts (tuple): Which limb pairs raise cues: "arms", "legs", or both. Scores are computed
            for both either way.
        margin (float): Crossing score (shoulder widths) above which limbs count as crossed
            (uncrossed about -1, folded arms about +0.5 to +1).
        min_confidence (float): Points below this confidence count as missing.
        min_width_ratio (float): Shoulder width / torso length below which the person is too
            side-on to judge and the frame is unknown.
        min_duration (float): Seconds the limbs must stay crossed before a cue starts.
        max_gap_duration (float): Seconds the limbs may go unseen inside a crossing without
            ending it. Folded arms often hide a wrist, so some tolerance is needed.
    """

    behaviour = "body_crossing"
    needs = "bodies"

    def __init__(
        self,
        parts=("arms",),
        margin: float = 0.1,
        min_confidence: float = 0.3,
        min_width_ratio: float = 0.35,
        min_duration: float = 2.0,
        max_gap_duration: float = 0.5,
    ):
        self.parts = tuple(parts)
        self.margin = margin
        self.min_confidence = min_confidence
        self.min_width_ratio = min_width_ratio
        self.cues = SustainedCue(self.behaviour, min_duration, max_gap_duration)

    def update(self, observation: Observation, timestamp: float, frame: int):
        """
        Process one frame.

        Returns:
            tuple: (result, events). result has "crossing_score" and "crossing_state", both
            (persons, 2) in LIMBS order (arms, legs); crossing_state is 1 while a cue is active,
            0 otherwise, NaN where the limbs were not seen or raise no cues. events lists this
            frame's CueEvents.
        """
        people = observation.bodies
        score = np.full((len(people), len(crossing_utils.LIMBS)), np.nan)
        state = np.full_like(score, np.nan)
        events = []

        for p, person in enumerate(people):
            points = {
                name: np.asarray(person.coords[name], dtype=float)
                if person.confidence[name] >= self.min_confidence
                else np.full(2, np.nan)
                for name in crossing_utils.REQUIRED_POINTS
            }
            score[p] = crossing_utils.calculate_crossing_scores(points, self.min_width_ratio)
            for i, limb in enumerate(crossing_utils.LIMBS):
                if limb not in self.parts:
                    continue
                crossed = np.nan if np.isnan(score[p, i]) else float(score[p, i] > self.margin)
                state[p, i] = self.cues.step(p, limb, crossed, timestamp, frame, events)

        self.cues.end_frame(timestamp, frame, events)
        return {"crossing_score": score, "crossing_state": state}, events
