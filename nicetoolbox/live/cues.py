"""
Cue events and the causal duration filter shared by the live feature detectors.

The batch detectors keep runs that last at least `min_duration`, judged after the fact. A live
cue can only fire once a behaviour has already lasted `min_duration`, so a cue starts that long
after the behaviour starts, and ends on the first frame the behaviour stops or is not seen.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CueEvent:
    """A behaviour starting or stopping, as it would be relayed to the therapist.

    Attributes:
        behaviour (str): Name of the behaviour, e.g. "hand_closure" or "eye_closure".
        person (int): Person position in the backend's output.
        part (str): Body part, e.g. "left_hand" or "both_eyes".
        state (str): "start" or "end".
        timestamp (float): Stream time of the frame that triggered the event, in seconds.
        frame (int): Index of that frame.
        onset (float): Stream time the behaviour itself began (its first frame).
    """

    behaviour: str
    person: int
    part: str
    state: str
    timestamp: float
    frame: int
    onset: float


class _Run:
    def __init__(self):
        self.start = None  # stream time of the first frame in the current run
        self.active = False  # True once the run has lasted min_duration and a cue started


class SustainedCue:
    """
    Turns a per-frame yes/no/unknown signal into start and end cues, per (person, part).

    Args:
        behaviour (str): Name put on the events.
        min_duration (float): Seconds the signal must hold before a cue starts (0 = immediately).
    """

    def __init__(self, behaviour: str, min_duration: float):
        self.behaviour = behaviour
        self.min_duration = min_duration
        self._runs: dict[tuple[int, str], _Run] = {}
        self._seen: set = set()

    def step(self, person: int, part: str, present: float, timestamp: float, frame: int, events: list) -> float:
        """
        Feed one frame's signal for one (person, part).

        Args:
            present (float): 1 if the behaviour is seen this frame, 0 if not, NaN if unknown
                (e.g. the hand or face was not detected). Unknown ends a run, as in the batch
                detectors.
            events (list): Start/end CueEvents are appended here.

        Returns:
            float: 1 while a cue is active, 0 otherwise, NaN where the signal was unknown.
        """
        key = (person, part)
        self._seen.add(key)
        run = self._runs.setdefault(key, _Run())
        if np.isnan(present) or present == 0:
            if run.active:
                events.append(CueEvent(self.behaviour, person, part, "end", timestamp, frame, run.start))
            run.start, run.active = None, False
            return np.nan if np.isnan(present) else 0.0

        if run.start is None:
            run.start = timestamp
        # The tolerance keeps float timestamps (e.g. 0.7 - 0.2) from missing the boundary.
        if not run.active and timestamp - run.start >= self.min_duration - 1e-6:
            run.active = True
            events.append(CueEvent(self.behaviour, person, part, "start", timestamp, frame, run.start))
        return 1.0 if run.active else 0.0

    def end_frame(self, timestamp: float, frame: int, events: list) -> None:
        """Call after every frame: ends the cues of any (person, part) not fed this frame."""
        for key in [k for k in self._runs if k not in self._seen]:
            run = self._runs.pop(key)
            if run.active:
                events.append(CueEvent(self.behaviour, *key, "end", timestamp, frame, run.start))
        self._seen = set()
