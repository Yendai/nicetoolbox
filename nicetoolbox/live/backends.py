"""
Keypoint backends for the live loop.

Each backend turns one BGR frame into an Observation: 21-point hands and six-point eyes per
person, in the point orders the batch toolbox uses, so the batch feature utilities apply unchanged.

- MediapipeBackend: MediaPipe Face Landmarker and/or Hand Landmarker. Real time on a CPU; default.
- RtmlibWholebody: RTMW whole-body keypoints through rtmlib and ONNX Runtime. Same 133-point
  COCO-WholeBody layout as the batch rtmpose_l_wholebody detector; wants a GPU.
"""

import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from ..detectors.feature_detectors.hand_closure.utils import HAND_SIDES, POINTS_PER_HAND

MEDIAPIPE_MODEL_URLS = {
    "hands": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "faces": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
DEFAULT_ASSETS_DIR = os.path.join("assets", "live")

EYE_SIDES = ("left_eye", "right_eye")
POINTS_PER_EYE = 6

# Point ranges in the 133-point COCO-WholeBody layout (0-indexed), as in predictions_mapping.toml.
WHOLEBODY_HAND_SLICES = {"left_hand": slice(91, 112), "right_hand": slice(112, 133)}
WHOLEBODY_EYE_SLICES = {"left_eye": slice(65, 71), "right_eye": slice(59, 65)}

# MediaPipe face mesh points for each of the person's eyes, in the six-point order the batch
# toolbox uses: corner, two upper-lid points, the other corner, two lower-lid points (the lower
# points run back, so pairs 1-5 and 2-4 face each other across the eye).
MEDIAPIPE_EYE_POINTS = {
    "left_eye": [362, 385, 387, 263, 373, 380],
    "right_eye": [33, 160, 158, 133, 153, 144],
}


def _empty_parts(sides, n_points) -> tuple[dict, dict]:
    return (
        {side: np.zeros((n_points, 2)) for side in sides},
        {side: np.zeros(n_points) for side in sides},
    )


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
        return cls(*_empty_parts(HAND_SIDES, POINTS_PER_HAND))


@dataclass
class PersonEyes:
    """Eyes of one person in one frame, in pixel coordinates.

    Attributes:
        coords (dict): side ("left_eye" / "right_eye") -> (6, 2) point coordinates.
        confidence (dict): side -> (6,) confidence per point; all zeros where the eye was not seen.
    """

    coords: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PersonEyes":
        return cls(*_empty_parts(EYE_SIDES, POINTS_PER_EYE))


@dataclass
class Observation:
    """Everything the backend found in one frame. A list is empty when it was not asked for."""

    hands: list[PersonHands] = field(default_factory=list)
    faces: list[PersonEyes] = field(default_factory=list)


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


class MediapipeBackend:
    """
    MediaPipe Face Landmarker and/or Hand Landmarker in video mode (they track between frames).

    MediaPipe does not give a confidence per point, so every point of a detected hand carries
    the hand's handedness score, and every eye point of a detected face is 1; anything undetected
    has zero confidence. The hand model sees one set of hands, not people, so hands are always
    reported as a single person.

    The face detector is a short-range model: in testing it missed a face that filled about 7%
    of the frame width and found it at 16%. Point a camera at the client's face, or pass
    face_roi, a region around where they sit, so the face model only looks there.

    Args:
        parts (tuple): Which models to run: "faces", "hands", or both.
        assets_dir (str): Where the .task model files live; they are downloaded on first use.
        num_hands (int): Maximum hands to track. 2 covers one client.
        num_faces (int): Maximum faces to track.
        mirrored (bool): True if frames are mirrored (selfie view); hand labels are then swapped so
            they still name the person's own left and right. On unmirrored frames MediaPipe's labels
            already match the person's sides (checked on the toolbox's hand and seated-pose images).
        device (str): "cpu" or "gpu" (GPU delegate; not available on every platform).
        face_roi (tuple): Optional (x, y, width, height) in pixels for the face model; hands still
            use the whole frame. Results are in full-frame coordinates either way.
    """

    name = "mediapipe"

    def __init__(
        self,
        parts=("faces",),
        assets_dir: str = DEFAULT_ASSETS_DIR,
        num_hands: int = 2,
        num_faces: int = 1,
        mirrored: bool = False,
        device: str = "cpu",
        face_roi=None,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.mirrored = mirrored
        self.face_roi = tuple(int(v) for v in face_roi) if face_roi else None
        delegate = BaseOptions.Delegate.GPU if device == "gpu" else BaseOptions.Delegate.CPU

        def base_options(part):
            path = _download(MEDIAPIPE_MODEL_URLS[part], os.path.join(assets_dir, f"{part[:-1]}_landmarker.task"))
            return BaseOptions(model_asset_path=path, delegate=delegate)

        self.hand_landmarker = self.face_landmarker = None
        if "hands" in parts:
            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base_options("hands"), running_mode=vision.RunningMode.VIDEO, num_hands=num_hands
                )
            )
        if "faces" in parts:
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base_options("faces"), running_mode=vision.RunningMode.VIDEO, num_faces=num_faces
                )
            )
        self._last_ms = -1

    def __call__(self, image: np.ndarray, timestamp: float) -> Observation:
        import cv2

        # Video mode needs strictly increasing timestamps in whole milliseconds.
        ts_ms = max(int(timestamp * 1000), self._last_ms + 1)
        self._last_ms = ts_ms
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def to_mp_image(array):
            return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(array))

        # Scale to pixels so ratios are not distorted by the image's aspect ratio; offset is where
        # the analysed region starts in the full frame.
        def to_pixels(landmarks, size, offset=(0, 0), indices=None):
            points = landmarks if indices is None else [landmarks[i] for i in indices]
            (width, height), (dx, dy) = size, offset
            return np.array([[lm.x * width + dx, lm.y * height + dy] for lm in points])

        observation = Observation()
        if self.hand_landmarker:
            result = self.hand_landmarker.detect_for_video(to_mp_image(rgb), ts_ms)
            size = (image.shape[1], image.shape[0])
            person = PersonHands.empty()
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name.lower()  # "left" / "right"
                if self.mirrored:
                    label = "right" if label == "left" else "left"
                side, score = f"{label}_hand", handedness[0].score
                # Keep the more confident detection if both hands got the same label.
                if person.confidence[side][0] >= score:
                    continue
                person.coords[side] = to_pixels(landmarks, size)
                person.confidence[side] = np.full(POINTS_PER_HAND, score)
            observation.hands.append(person)

        if self.face_landmarker:
            x, y, w, h = self.face_roi or (0, 0, image.shape[1], image.shape[0])
            region = rgb[y : y + h, x : x + w]
            result = self.face_landmarker.detect_for_video(to_mp_image(region), ts_ms)
            size = (region.shape[1], region.shape[0])
            for landmarks in result.face_landmarks:
                eyes = PersonEyes()
                for side, indices in MEDIAPIPE_EYE_POINTS.items():
                    eyes.coords[side] = to_pixels(landmarks, size, (x, y), indices)
                    eyes.confidence[side] = np.ones(POINTS_PER_EYE)
                observation.faces.append(eyes)
        return observation

    def close(self) -> None:
        for landmarker in (self.hand_landmarker, self.face_landmarker):
            if landmarker:
                landmarker.close()


def wholebody_to_observation(keypoints: np.ndarray, scores: np.ndarray, parts=("faces", "hands")) -> Observation:
    """
    Split 133-point COCO-WholeBody results into hands and eyes per person.

    Args:
        keypoints (np.ndarray): (persons, 133, 2) coordinates.
        scores (np.ndarray): (persons, 133) confidence scores.
    """
    observation = Observation()
    for person_kpts, person_scores in zip(keypoints, scores):
        for part, cls, slices, target in (
            ("hands", PersonHands, WHOLEBODY_HAND_SLICES, observation.hands),
            ("faces", PersonEyes, WHOLEBODY_EYE_SLICES, observation.faces),
        ):
            if part not in parts:
                continue
            person = cls()
            for side, points in slices.items():
                person.coords[side] = np.asarray(person_kpts[points], dtype=float)
                person.confidence[side] = np.asarray(person_scores[points], dtype=float)
            target.append(person)
    return observation


class RtmlibWholebody:
    """
    RTMW whole-body keypoints (body, face, hands) through rtmlib's PoseTracker.

    One model gives both hands and eyes, for every person in view. The person detector runs
    every `det_frequency` frames and boxes are tracked in between, which is most of the speed-up
    over running detection on every frame. IoU tracking keeps each person at the same list
    position from frame to frame while they stay in view. Eye points come from the whole-body
    crop, so they are coarser than a dedicated face model's when the person is far away.

    Args:
        parts (tuple): Which parts to report: "faces", "hands", or both.
        mode (str): rtmlib preset: "lightweight", "balanced" or "performance".
        device (str): "cpu" or "cuda" (needs onnxruntime-gpu).
        det_frequency (int): Run the person detector once every this many frames.
    """

    name = "rtmlib"

    def __init__(self, parts=("faces", "hands"), mode: str = "balanced", device: str = "cuda", det_frequency: int = 10):
        from rtmlib import PoseTracker, Wholebody

        self.parts = parts
        self.tracker = PoseTracker(
            Wholebody,
            det_frequency=det_frequency,
            mode=mode,
            backend="onnxruntime",
            device=device,
            to_openpose=False,
        )

    def __call__(self, image: np.ndarray, timestamp: float) -> Observation:  # noqa: ARG002
        keypoints, scores = self.tracker(image)
        if keypoints is None or len(keypoints) == 0:
            return Observation()
        return wholebody_to_observation(np.asarray(keypoints), np.asarray(scores), self.parts)

    def close(self) -> None:
        pass


def create_backend(name: str, **kwargs):
    """Build a backend by name ("mediapipe" or "rtmlib")."""
    if name == "mediapipe":
        return MediapipeBackend(**kwargs)
    if name == "rtmlib":
        return RtmlibWholebody(**kwargs)
    raise ValueError(f"Unknown live backend '{name}'. Choose 'mediapipe' or 'rtmlib'.")
