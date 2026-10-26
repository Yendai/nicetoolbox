"""
Streaming version of the hand closure (fist) feature detector.

The per-frame score is the batch detector's finger extension ratio, computed with the same
utility function. A cue starts once a fist has been held for `min_duration` (see cues.py).
"""

import numpy as np

from ..detectors.feature_detectors.hand_closure import utils as hand_utils
from .backends import Observation
from .cues import SustainedCue


class LiveHandClosure:
    """
    Frame-by-frame fist detection with start/end cues per hand.

    Args:
        threshold (float): Max finger extension ratio that counts as a fist (open ~1.7-2.0, fist ~1.0).
        min_confidence (float): Points below this confidence count as missing.
        min_duration (float): Seconds a fist must be held before a cue starts (0 = immediately).
    """

    behaviour = "hand_closure"
    needs = "hands"

    def __init__(self, threshold: float = 1.2, min_confidence: float = 0.3, min_duration: float = 0.5):
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.cues = SustainedCue(self.behaviour, min_duration)

    def update(self, observation: Observation, timestamp: float, frame: int):
        """
        Process one frame.

        Returns:
            tuple: (result, events). result has "extension_ratio" and "fist_state", both
            (persons, 2) with the hand sides in HAND_SIDES order; fist_state is 1 while a cue is
            active, 0 otherwise, NaN where the hand was not seen. events lists this frame's CueEvents.
        """
        people = observation.hands
        ratio = np.full((len(people), len(hand_utils.HAND_SIDES)), np.nan)
        state = np.full_like(ratio, np.nan)
        events = []

        for p, person in enumerate(people):
            for s, side in enumerate(hand_utils.HAND_SIDES):
                r = float(
                    hand_utils.calculate_extension_ratio(
                        person.coords[side], person.confidence[side], self.min_confidence
                    )
                )
                ratio[p, s] = r
                fist = np.nan if np.isnan(r) else float(r < self.threshold)
                state[p, s] = self.cues.step(p, side, fist, timestamp, frame, events)

        self.cues.end_frame(timestamp, frame, events)
        return {"extension_ratio": ratio, "fist_state": state}, events
