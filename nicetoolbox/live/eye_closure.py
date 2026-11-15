"""
Streaming version of the eye closure (EAR) feature detectors.

The per-frame score is the batch eye_closure_ear detector's Eye Aspect Ratio, computed with the
same calculate_ear function on six points per eye. A cue starts once both eyes have stayed
closed for `min_duration`, which also filters out ordinary blinks (well under half a second).

An eye counts as closed below a threshold. A fixed threshold, as in the batch
eye_closure_threshold detector, depends on the person, the camera and the landmark model: on the
toolbox's seated-subject image, MediaPipe measured open eyes at EAR 0.10-0.19, under the 0.15 a
fixed threshold might use. So by default the threshold is relative: a fraction of that eye's own
open-eye EAR, estimated from the recent past. Until about a second of history has built up the
eye state is unknown, so no cue fires on a guess.
"""

from collections import deque

import numpy as np

from ..configs.schemas.predictions_mapping import EyeLayout
from ..detectors.feature_detectors.eye_closure_ear.utils import calculate_ear
from .backends import EYE_SIDES, Observation
from .cues import SustainedCue

# Six points per eye: outer/inner corner at 0 and 3, upper lid 1-2, lower lid 5-4. Same as the
# toolbox's coco_wholebody eye layout, so the backends hand over their points in this order.
SIX_POINT_EYE = EyeLayout(
    upper_eyelid=[0, 1, 2, 3], lower_eyelid=[0, 5, 4, 3], vertical_pairs=[[1, 5], [2, 4]], horizontal_pair=[0, 3]
)


class OpenEyeBaseline:
    """
    Running estimate of one eye's open EAR: a high percentile of its recent scores, so blinks
    and closures pull it down only a little.

    Args:
        window (float): Seconds of history to keep.
        percentile (float): Which percentile of the history counts as "open".
        min_samples (int): History needed before the estimate is used.
    """

    def __init__(self, window: float = 60.0, percentile: float = 90.0, min_samples: int = 30):
        self.window = window
        self.percentile = percentile
        self.min_samples = min_samples
        self._history: deque = deque()

    def update(self, ear: float, timestamp: float) -> float:
        """Add this frame's EAR (NaN is skipped) and return the open estimate, or NaN if not ready."""
        if not np.isnan(ear):
            self._history.append((timestamp, ear))
        while self._history and self._history[0][0] < timestamp - self.window:
            self._history.popleft()
        if len(self._history) < self.min_samples:
            return np.nan
        return float(np.percentile([e for _, e in self._history], self.percentile))


class LiveEyeClosure:
    """
    Frame-by-frame eye closure with start/end cues when both eyes stay closed.

    Args:
        threshold (float): Fixed max EAR for a closed eye, as in the batch detector. Used only
            when relative is 0.
        min_confidence (float): Eye points below this confidence count as missing.
        min_duration (float): Seconds both eyes must stay closed before a cue starts.
        relative (float): If above 0, an eye is closed below this fraction of its open-eye
            baseline (e.g. 0.6 = the eye has lost 40% of its usual opening).
        baseline_window (float): Seconds of history the open-eye baseline looks back over.
    """

    behaviour = "eye_closure"
    needs = "faces"

    def __init__(
        self,
        threshold: float = 0.1,
        min_confidence: float = 0.3,
        min_duration: float = 1.0,
        relative: float = 0.6,
        baseline_window: float = 60.0,
    ):
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.relative = relative
        self.baseline_window = baseline_window
        self.cues = SustainedCue(self.behaviour, min_duration)
        self._baselines: dict[tuple[int, str], OpenEyeBaseline] = {}

    def _threshold(self, person: int, side: str, ear: float, timestamp: float) -> float:
        if self.relative <= 0:
            return self.threshold
        baseline = self._baselines.setdefault((person, side), OpenEyeBaseline(self.baseline_window))
        open_ear = baseline.update(ear, timestamp)
        return self.relative * open_ear  # NaN while warming up, so the eye state is unknown

    def update(self, observation: Observation, timestamp: float, frame: int):
        """
        Process one frame.

        Returns:
            tuple: (result, events). result["ear"] is (persons, 2) in EYE_SIDES order;
            result["threshold"] is (persons, 2), the threshold each eye was judged against (NaN
            while the baseline warms up);
            result["closed"] is (persons, 3) per-frame closed state for left, right and both eyes
            (NaN where unseen); result["cue_state"] is (persons,) 1 while a both-eyes cue is active.
        """
        faces = observation.faces
        ear = np.full((len(faces), len(EYE_SIDES)), np.nan)
        threshold = np.full_like(ear, np.nan)
        closed = np.full((len(faces), len(EYE_SIDES) + 1), np.nan)
        cue_state = np.full(len(faces), np.nan)
        events = []

        for p, face in enumerate(faces):
            for s, side in enumerate(EYE_SIDES):
                coords = np.where((face.confidence[side] >= self.min_confidence)[:, None], face.coords[side], np.nan)
                ear[p, s] = float(calculate_ear(coords, SIX_POINT_EYE))
                threshold[p, s] = self._threshold(p, side, ear[p, s], timestamp)
                unknown = np.isnan(ear[p, s]) or np.isnan(threshold[p, s])
                closed[p, s] = np.nan if unknown else float(ear[p, s] < threshold[p, s])
            # Both closed needs both eyes seen; one unseen eye makes it unknown.
            closed[p, 2] = np.nan if np.isnan(closed[p, :2]).any() else float(closed[p, :2].all())
            cue_state[p] = self.cues.step(p, "both_eyes", closed[p, 2], timestamp, frame, events)

        self.cues.end_frame(timestamp, frame, events)
        return {"ear": ear, "threshold": threshold, "closed": closed, "cue_state": cue_state}, events
