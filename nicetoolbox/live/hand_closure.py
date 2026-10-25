"""
Streaming version of the hand closure (fist) feature detector.

The per-frame score is the batch detector's finger extension ratio, computed with the same
utility function. What changes is the duration filter: the batch detector looks at whole runs
after the fact, while a live cue can only fire once a fist has already been held for
`min_duration`. So a cue starts min_duration after the fist starts, and ends on the first
frame the hand is open again or not seen.
"""

from dataclasses import dataclass

import numpy as np

from ..detectors.feature_detectors.hand_closure import utils as hand_utils
from .backends import PersonHands


@dataclass
class CueEvent:
    """A behaviour starting or stopping, as it would be relayed to the therapist.

    Attributes:
        behaviour (str): Name of the behaviour, e.g. "hand_closure".
        person (int): Person position in the backend's output.
        side (str): "left_hand" or "right_hand".
        state (str): "start" or "end".
        timestamp (float): Stream time of the frame that triggered the event, in seconds.
        frame (int): Index of that frame.
        onset (float): Stream time the behaviour itself began (the first closed frame).
    """

    behaviour: str
    person: int
    side: str
    state: str
    timestamp: float
    frame: int
    onset: float


class _RunState:
    """Tracks one hand's current run of closed frames."""

    def __init__(self):
        self.run_start = None  # stream time of the first closed frame in the current run
        self.active = False  # True once the run has lasted min_duration and a cue started


class LiveHandClosure:
    """
    Frame-by-frame fist detection with start/end cues.

    Args:
        threshold (float): Max finger extension ratio that counts as a fist (open ~1.7-2.0, fist ~1.0).
        min_confidence (float): Points below this confidence count as missing.
        min_duration (float): Seconds a fist must be held before a cue starts (0 = immediately).
    """

    behaviour = "hand_closure"

    def __init__(self, threshold: float = 1.2, min_confidence: float = 0.3, min_duration: float = 0.5):
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.min_duration = min_duration
        self._states: dict[tuple[int, str], _RunState] = {}

    def update(self, people: list[PersonHands], timestamp: float, frame: int):
        """
        Process one frame.

        Returns:
            tuple: (extension_ratio, fist_state, events). The arrays are (persons, 2) with the
            hand sides in HAND_SIDES order; fist_state is 1 while a cue is active, 0 otherwise,
            NaN where the hand was not seen. events lists the CueEvents this frame triggered.
        """
        n = len(people)
        ratio = np.full((n, len(hand_utils.HAND_SIDES)), np.nan)
        state = np.full_like(ratio, np.nan)
        events = []
        seen = set()

        for person_idx, person in enumerate(people):
            for side_idx, side in enumerate(hand_utils.HAND_SIDES):
                key = (person_idx, side)
                seen.add(key)
                r = float(
                    hand_utils.calculate_extension_ratio(
                        person.coords[side], person.confidence[side], self.min_confidence
                    )
                )
                ratio[person_idx, side_idx] = r
                run = self._states.setdefault(key, _RunState())

                if np.isnan(r) or r >= self.threshold:
                    # Open or not seen: the run ends, as in the batch detector.
                    if run.active:
                        events.append(self._event(key, "end", timestamp, frame, run.run_start))
                    run.run_start, run.active = None, False
                    state[person_idx, side_idx] = np.nan if np.isnan(r) else 0.0
                    continue

                if run.run_start is None:
                    run.run_start = timestamp
                # The tolerance keeps float timestamps (e.g. 0.7 - 0.2) from missing the boundary.
                if not run.active and timestamp - run.run_start >= self.min_duration - 1e-6:
                    run.active = True
                    events.append(self._event(key, "start", timestamp, frame, run.run_start))
                state[person_idx, side_idx] = 1.0 if run.active else 0.0

        # A person who left the frame ends their active cues.
        for key in [k for k in self._states if k not in seen]:
            run = self._states.pop(key)
            if run.active:
                events.append(self._event(key, "end", timestamp, frame, run.run_start))

        return ratio, state, events

    def _event(self, key, state, timestamp, frame, onset) -> CueEvent:
        person, side = key
        return CueEvent(self.behaviour, person, side, state, timestamp, frame, onset)
