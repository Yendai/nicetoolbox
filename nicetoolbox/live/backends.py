"""
Keypoint backends for the live loop.

Each backend turns one BGR frame into an Observation: 21-point hands, six-point eyes and named
body points per person, in the point orders and names the batch toolbox uses, so the batch
feature utilities apply unchanged.

- MediapipeBackend: MediaPipe Face, Hand and/or Pose Landmarker. Real time on a CPU; default.
- RtmlibWholebody: RTMW whole-body keypoints through rtmlib and ONNX Runtime. Same 133-point
  COCO-WholeBody layout as the batch rtmpose_l_wholebody detector; wants a GPU.
"""

import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from ..detectors.feature_detectors.body_crossing.utils import REQUIRED_POINTS as BODY_POINTS
from ..detectors.feature_detectors.hand_closure.utils import HAND_SIDES, POINTS_PER_HAND

MEDIAPIPE_MODEL_URLS = {
    "hands": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "faces": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "bodies": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task",
}
MEDIAPIPE_MODEL_FILES = {
    "hands": "hand_landmarker.task",
    "faces": "face_landmarker.task",
    "bodies": "pose_landmarker_{variant}.task",
}
DEFAULT_ASSETS_DIR = os.path.join("assets", "live")

EYE_SIDES = ("left_eye", "right_eye")
POINTS_PER_EYE = 6

# Point ranges in the 133-point COCO-WholeBody layout (0-indexed), as in predictions_mapping.toml.
WHOLEBODY_HAND_SLICES = {"left_hand": slice(91, 112), "right_hand": slice(112, 133)}
WHOLEBODY_EYE_SLICES = {"left_eye": slice(65, 71), "right_eye": slice(59, 65)}

# Where the body points the crossing detector reads (shoulders, hips, wrists, knees, ankles) sit
# in each model's output. Both models name sides from the person's own point of view.
WHOLEBODY_BODY_INDEX = {
    "left_shoulder": 5, "right_shoulder": 6, "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}  # fmt: skip
MEDIAPIPE_POSE_INDEX = {
    "left_shoulder": 11, "right_shoulder": 12, "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}  # fmt: skip

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
class PersonBody:
    """Body points of one person in one frame, in pixel coordinates.

    Attributes:
        coords (dict): point name (e.g. "left_wrist", see BODY_POINTS) -> (2,) coordinates.
        confidence (dict): point name -> confidence; zero where the point was not seen.
    """

    coords: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PersonBody":
        return cls({name: np.zeros(2) for name in BODY_POINTS}, {name: 0.0 for name in BODY_POINTS})


@dataclass
class Observation:
    """Everything the backend found in one frame. A list is empty when it was not asked for."""

    hands: list[PersonHands] = field(default_factory=list)
    faces: list[PersonEyes] = field(default_factory=list)
    bodies: list[PersonBody] = field(default_factory=list)


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


class MediapipeBackend:
    """
    MediaPipe Face, Hand and/or Pose Landmarker in video mode (they track between frames).

    MediaPipe does not give a confidence per point, so every point of a detected hand carries
    the hand's handedness score, and every eye point of a detected face is 1; anything undetected
    has zero confidence. The hand model sees one set of hands, not people, so hands are always
    reported as a single person. Body points carry the pose model's own per-point visibility,
    which drops when a point is hidden, for example a wrist tucked under the other arm.

    The face detector is a short-range model: in testing it missed a face that filled about 7%
    of the frame width and found it at 16%. Point a camera at the client's face, or pass
    face_roi, a region around where they sit, so the face model only looks there.

    Args:
        parts (tuple): Which models to run: any of "faces", "hands", "bodies".
        assets_dir (str): Where the .task model files live; they are downloaded on first use.
        num_hands (int): Maximum hands to track. 2 covers one client.
        num_faces (int): Maximum faces to track.
        num_bodies (int): Maximum bodies to track.
        pose_variant (str): Pose model size: "lite", "full" or "heavy" (slower, steadier points).
        mirrored (bool): True if frames are mirrored (selfie view); hand and body side labels are
            then swapped so they still name the person's own left and right. On unmirrored frames MediaPipe's labels
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
        num_bodies: int = 1,
        pose_variant: str = "lite",
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
            url = MEDIAPIPE_MODEL_URLS[part].format(variant=pose_variant)
            file_name = MEDIAPIPE_MODEL_FILES[part].format(variant=pose_variant)
            return BaseOptions(model_asset_path=_download(url, os.path.join(assets_dir, file_name)), delegate=delegate)

        self.hand_landmarker = self.face_landmarker = self.pose_landmarker = None
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
        if "bodies" in parts:
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base_options("bodies"), running_mode=vision.RunningMode.VIDEO, num_poses=num_bodies
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

        if self.pose_landmarker:
            result = self.pose_landmarker.detect_for_video(to_mp_image(rgb), ts_ms)
            width, height = image.shape[1], image.shape[0]
            for landmarks in result.pose_landmarks:
                body = PersonBody()
                for name, index in MEDIAPIPE_POSE_INDEX.items():
                    if self.mirrored:
                        name = name.replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                    point = landmarks[index]
                    body.coords[name] = np.array([point.x * width, point.y * height])
                    body.confidence[name] = float(point.visibility)
                observation.bodies.append(body)
        return observation

    def close(self) -> None:
        for landmarker in (self.hand_landmarker, self.face_landmarker, self.pose_landmarker):
            if landmarker:
                landmarker.close()


def wholebody_to_observation(keypoints: np.ndarray, scores: np.ndarray, parts=("faces", "hands")) -> Observation:
    """
    Split 133-point COCO-WholeBody results into hands, eyes and body points per person.

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
        if "bodies" in parts:
            observation.bodies.append(
                PersonBody(
                    {name: np.asarray(person_kpts[i], dtype=float) for name, i in WHOLEBODY_BODY_INDEX.items()},
                    {name: float(person_scores[i]) for name, i in WHOLEBODY_BODY_INDEX.items()},
                )
            )
    return observation


class RtmlibWholebody:
    """
    RTMW whole-body keypoints (body, face, hands) through rtmlib's PoseTracker.

    One model gives hands, eyes and body points, for every person in view. The person detector runs
    every `det_frequency` frames and boxes are tracked in between, which is most of the speed-up
    over running detection on every frame. IoU tracking keeps each person at the same list
    position from frame to frame while they stay in view. Eye points come from the whole-body
    crop, so they are coarser than a dedicated face model's when the person is far away.

    Args:
        parts (tuple): Which parts to report: any of "faces", "hands", "bodies".
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
