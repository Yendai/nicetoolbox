# Live loop (proof of concept)

Runs feature detectors frame by frame on a webcam or video, prints a cue when a behaviour starts
and stops, and measures how long each frame takes. The first detector is hand closure (fist),
using the same finger extension ratio as the batch `hand_closure` detector.

The loop flags observable behaviour only (a hand held closed). It does not infer emotions.

## Setup

The live loop has its own small environment, separate from the batch toolbox's pinned one:

```bash
python -m venv envs/live
envs/live/bin/pip install -r nicetoolbox/live/requirements.txt
```

On a headless Linux machine MediaPipe also needs `libegl1` and `libgles2` from the system packages.

## Run

From the repository root:

```bash
envs/live/bin/python -m nicetoolbox.live --source 0 --show              # webcam 0, overlay window (q quits)
envs/live/bin/python -m nicetoolbox.live --source clip.mp4 --out-dir out  # video, writes logs
envs/live/bin/python -m nicetoolbox.live --backend rtmlib --device cuda --source 0
```

The MediaPipe hand model (`hand_landmarker.task`, about 8 MB) is downloaded to `assets/live/` on
first run. Useful options: `--threshold` (default 1.2), `--min-duration` seconds a fist must be held
before a cue (default 0.5), `--mirrored` for selfie-view cameras, `--no-realtime` to process a
video as fast as possible. See `--help` for the rest.

With `--out-dir` the loop writes `latency.csv` (one row per frame), `cues.jsonl` (one line per
cue start or end) and `summary.json`.

## What the latency numbers mean

| Column | Meaning |
|---|---|
| `capture_ms` | Reading the frame from the camera or file |
| `keypoints_ms` | Hand keypoint model |
| `features_ms` | Extension ratio and cue logic |
| `total_ms` | From frame captured to cue ready: the delay the loop adds |

`total_ms` leaves out the camera's own exposure and transfer time and the time to deliver a cue
to the therapist. A cue also waits `--min-duration` by design: a fist is only reported once it has
been held that long, so it reaches the therapist about `min_duration + total_ms` after it starts.

## Backends

| Backend | Model | Hardware | Notes |
|---|---|---|---|
| `mediapipe` (default) | MediaPipe Hand Landmarker, Apache-2.0 | CPU real time | Crops each hand at high resolution and tracks between frames. Up to two hands, no notion of separate people. |
| `rtmlib` | RTMW whole-body via rtmlib + ONNX Runtime, Apache-2.0 | GPU (`onnxruntime-gpu`) | Same 133-point layout as the batch `rtmpose_l_wholebody`, so live and batch features match. Several people, with body and face points too. |

Both give 21-point hands in the toolbox's point order, so `calculate_extension_ratio` from the
batch detector is reused unchanged.

## Differences from the batch detector

The batch detector keeps fist runs that last at least `min_duration`, judged after the fact, so a
kept run is marked from its first frame. Live, the state turns on once the fist has already lasted
`min_duration`, and turns off on the first frame the hand is open or not seen. A hand that drops
out of tracking for one frame therefore ends the cue, as it ends a run in the batch detector.
