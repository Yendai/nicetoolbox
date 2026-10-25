"""
Keypoint backends for the live loop.

Each backend turns one BGR frame into 21-point hands per person, in the same point order the
batch toolbox uses (wrist, then four points per finger from the knuckle out to the tip), so the
batch feature utilities apply unchanged.

- MediapipeHands: MediaPipe Hand Landmarker. Runs in real time on a CPU; default.
- RtmlibWholebody: RTMW whole-body keypoints through rtmlib and ONNX Runtime. Same 133-point
  COCO-WholeBody layout as the batch rtmpose_l_wholebody detector; wants a GPU.
"""

import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from ..detectors.feature_detectors.hand_closure.utils import HAND_SIDES, POINTS_PER_HAND

MEDIAPIPE_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_ASSETS_DIR = os.path.join("assets", "live")

# Point ranges of each hand in the 133-point COCO-WholeBody layout (0-indexed).
WHOLEBODY_HAND_SLICES = {"left_hand": slice(91, 112), "right_hand": slice(112, 133)}


@dataclass
class PersonHands:
    """Hands of one person in one frame, in pixel coordinates.

    Attributes:
        coords (dict): side ("left_hand" / "right_hand") -> (21, 2) point coordinates.
        confidence (dict): side -> (21,) confidence per point; all zeros where the hand was not seen.
    """

    coords: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PersonHands":
        hands = cls()
        for side in HAND_SIDES:
            hands.coords[side] = np.zeros((POINTS_PER_HAND, 2))
            hands.confidence[side] = np.zeros(POINTS_PER_HAND)
        return hands


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


class MediapipeHands:
    """
    MediaPipe Hand Landmarker in video mode (it tracks hands between frames).

    MediaPipe does not give a confidence per point, so every point of a detected hand carries
    the hand's handedness score; an undetected hand has zero confidence. MediaPipe sees one set
    of hands, not people, so it always reports a single person.

    Args:
        model_path (str): Path to hand_landmarker.task; downloaded on first use if missing.
        num_hands (int): Maximum hands to track. 2 covers one client.
        mirrored (bool): True if frames are mirrored (selfie view); hand labels are then swapped so
            they still name the person's own left and right. On unmirrored frames MediaPipe's labels
            already match the person's sides (checked on the toolbox's hand and seated-pose images).
        device (str): "cpu" or "gpu" (GPU delegate; not available on every platform).
    """

    name = "mediapipe"

    def __init__(self, model_path: str = None, num_hands: int = 2, mirrored: bool = False, device: str = "cpu"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        model_path = model_path or os.path.join(DEFAULT_ASSETS_DIR, "hand_landmarker.task")
        _download(MEDIAPIPE_HAND_MODEL_URL, model_path)
        delegate = BaseOptions.Delegate.GPU if device == "gpu" else BaseOptions.Delegate.CPU
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.mirrored = mirrored
        self._last_ms = -1

    def __call__(self, image: np.ndarray, timestamp: float) -> list[PersonHands]:
        import cv2

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Video mode needs strictly increasing timestamps in whole milliseconds.
        ts_ms = max(int(timestamp * 1000), self._last_ms + 1)
        self._last_ms = ts_ms
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        person = PersonHands.empty()
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name.lower()  # "left" / "right"
            if self.mirrored:
                label = "right" if label == "left" else "left"
            side = f"{label}_hand"
            score = handedness[0].score
            # Keep the more confident detection if both hands got the same label.
            if person.confidence[side][0] >= score:
                continue
            # Scale to pixels so the extension ratio is not distorted by the image's aspect ratio.
            person.coords[side] = np.array([[lm.x * width, lm.y * height] for lm in landmarks])
            person.confidence[side] = np.full(POINTS_PER_HAND, score)
        return [person]

    def close(self) -> None:
        self.landmarker.close()


def wholebody_to_hands(keypoints: np.ndarray, scores: np.ndarray) -> list[PersonHands]:
    """
    Split 133-point COCO-WholeBody results into hands per person.

    Args:
        keypoints (np.ndarray): (persons, 133, 2) coordinates.
        scores (np.ndarray): (persons, 133) confidence scores.
    """
    people = []
    for person_kpts, person_scores in zip(keypoints, scores):
        hands = PersonHands()
        for side, points in WHOLEBODY_HAND_SLICES.items():
            hands.coords[side] = np.asarray(person_kpts[points], dtype=float)
            hands.confidence[side] = np.asarray(person_scores[points], dtype=float)
        people.append(hands)
    return people


class RtmlibWholebody:
    """
    RTMW whole-body keypoints (body, face, hands) through rtmlib's PoseTracker.

    The person detector runs every `det_frequency` frames and boxes are tracked in between,
    which is most of the speed-up over running detection on every frame. IoU tracking keeps
    each person at the same list position from frame to frame while they stay in view.

    Args:
        mode (str): rtmlib preset: "lightweight", "balanced" or "performance".
        device (str): "cpu" or "cuda" (needs onnxruntime-gpu).
        det_frequency (int): Run the person detector once every this many frames.
    """

    name = "rtmlib"

    def __init__(self, mode: str = "balanced", device: str = "cuda", det_frequency: int = 10):
        from rtmlib import PoseTracker, Wholebody

        self.tracker = PoseTracker(
            Wholebody,
            det_frequency=det_frequency,
            mode=mode,
            backend="onnxruntime",
            device=device,
            to_openpose=False,
        )

    def __call__(self, image: np.ndarray, timestamp: float) -> list[PersonHands]:  # noqa: ARG002
        keypoints, scores = self.tracker(image)
        if keypoints is None or len(keypoints) == 0:
            return []
        return wholebody_to_hands(np.asarray(keypoints), np.asarray(scores))

    def close(self) -> None:
        pass


def create_backend(name: str, **kwargs):
    """Build a backend by name ("mediapipe" or "rtmlib")."""
    if name == "mediapipe":
        return MediapipeHands(**kwargs)
    if name == "rtmlib":
        return RtmlibWholebody(**kwargs)
    raise ValueError(f"Unknown live backend '{name}'. Choose 'mediapipe' or 'rtmlib'.")
