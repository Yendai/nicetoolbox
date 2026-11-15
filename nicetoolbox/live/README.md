# Live loop (proof of concept)

Runs feature detectors frame by frame on a webcam or video, prints a cue when a behaviour starts
and stops, and measures how long each frame takes. Two detectors so far, each reusing the batch
toolbox's own scoring function:

- **Eye closure** (default): the batch `eye_closure_ear` Eye Aspect Ratio on six points per eye.
  A cue starts when both eyes have stayed closed for `--eye-min-duration` (default 1 s), which
  ignores ordinary blinks.
- **Hand closure**: the batch `hand_closure` finger extension ratio. A cue starts when a fist has
  been held for `--hand-min-duration` (default 0.5 s).

The loop flags observable behaviour only (eyes or a hand held closed). It does not infer emotions.

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
envs/live/bin/python -m nicetoolbox.live --source 0 --show                     # webcam 0, eyes, overlay (q quits)
envs/live/bin/python -m nicetoolbox.live --source 0 --detect eyes,hands        # both detectors
envs/live/bin/python -m nicetoolbox.live --source clip.mp4 --out-dir out       # video, writes logs
envs/live/bin/python -m nicetoolbox.live --source 0 --face-roi 400,100,500,500 # wide room camera
envs/live/bin/python -m nicetoolbox.live --backend rtmlib --device cuda --source 0
```

MediaPipe model files (`face_landmarker.task` about 4 MB, `hand_landmarker.task` about 8 MB) are
downloaded to `assets/live/` on first run. See `--help` for all options.

With `--out-dir` the loop writes `latency.csv` (one row per frame), `cues.jsonl` (one line per
cue start or end) and `summary.json`.

## Camera framing matters for eyes

MediaPipe's face detector is a short-range model. On the toolbox's seated-subject photo it missed
the face when the face filled about 7% of the frame width and found it at 16%. With a wide room
camera, either point a second camera at the client's face, or pass `--face-roi x,y,width,height`
around where the client sits so the face model only looks there (hands still use the whole frame).

## Eye threshold

Open-eye EAR depends on the person, the camera and the landmark model: on that same photo
MediaPipe measured open eyes at 0.10-0.19, so a fixed threshold such as 0.15 would call them
closed. By default each eye is judged against its own recent past instead: closed below
`--eye-relative` (default 0.6) times the 90th percentile of its EAR over the last 60 s. The state is
unknown for the first second, while that history builds up. Pass `--eye-relative 0` to use the
fixed `--eye-threshold` (default 0.1, as in the batch config). If someone keeps their eyes closed
for most of a minute, the baseline drifts down with them and the cue ends; tune on annotated clips.

## What the latency numbers mean

| Column | Meaning |
|---|---|
| `capture_ms` | Reading the frame from the camera or file |
| `keypoints_ms` | Landmark models (face and/or hands) |
| `features_ms` | Scores and cue logic |
| `total_ms` | From frame captured to cue ready: the delay the loop adds |

`total_ms` leaves out the camera's own exposure and transfer time and the time to deliver a cue
to the therapist. A cue also waits its min duration by design, so it reaches the therapist about
`min_duration + total_ms` after the behaviour starts.

Measured on a 4-core cloud CPU (no GPU), 30 fps clips made from the toolbox's docs images:

| Run | total p50 | total p95 |
|---|---|---|
| Eyes, face filling the frame (640x640) | 11.7 ms | 14.1 ms |
| Eyes, room view (720x960) with `--face-roi` | 11.7 ms | 14.0 ms |
| Eyes and hands, room view with `--face-roi` | 32.2 ms | 38.5 ms |

## Backends

| Backend | Model | Hardware | Notes |
|---|---|---|---|
| `mediapipe` (default) | MediaPipe Face Landmarker (478 points) and Hand Landmarker, Apache-2.0 | CPU real time | Dedicated face and hand models at high resolution, tracked between frames. Short-range face detector (see above). |
| `rtmlib` | RTMW whole-body via rtmlib + ONNX Runtime, Apache-2.0 | GPU (`onnxruntime-gpu`) | One model for body, face (68 points) and hands, for several people. Same 133-point layout as the batch `rtmpose_l_wholebody`, so live and batch features match. Eye points are coarser when people are far away. |

## Differences from the batch detectors

The batch detectors keep runs that last at least their min duration, judged after the fact, so a
kept run is marked from its first frame. Live, the state turns on once the behaviour has already
lasted that long, and turns off on the first frame it stops or is not seen. A face or hand that
drops out of tracking for one frame therefore ends the cue, as it ends a run in the batch detectors.
