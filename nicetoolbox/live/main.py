"""
Live loop: camera or video -> keypoints -> eye / hand closure -> cues, with per-frame latency.

Run from the repository root:

    python -m nicetoolbox.live --source 0                          # webcam 0, eye closure, MediaPipe on CPU
    python -m nicetoolbox.live --source 0 --detect eyes,hands      # eye and hand closure together
    python -m nicetoolbox.live --source clip.mp4 --show            # video file with an overlay window
    python -m nicetoolbox.live --backend rtmlib --device cuda --source 0
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np

from ..detectors.feature_detectors.hand_closure.utils import FINGERS, HAND_SIDES, WRIST
from .backends import EYE_SIDES, create_backend
from .eye_closure import LiveEyeClosure
from .hand_closure import LiveHandClosure
from .latency import LatencyTracker
from .sources import open_source

DETECTORS = {"eyes": "faces", "hands": "hands"}  # --detect name -> backend part it needs


def run(frames, backend, detectors, latency, on_event=None, on_frame=None, max_frames=None) -> None:
    """
    Drive the loop over an iterable of sources.Frame.

    Args:
        detectors (list): Live feature detectors; each has update(observation, timestamp, frame).
        on_event (callable): Called with each CueEvent as soon as its frame is processed.
        on_frame (callable): Called with (frame, observation, results) after each frame, where
            results maps detector behaviour -> its result dict; e.g. to draw an overlay. Its time
            is not counted in the latency. Returning False stops the loop.
        max_frames (int): Stop after this many frames.
    """
    for frame in frames:
        t_kpts = time.perf_counter()
        observation = backend(frame.image, frame.timestamp)
        t_feat = time.perf_counter()
        results, events = {}, []
        for detector in detectors:
            results[detector.behaviour], detector_events = detector.update(observation, frame.timestamp, frame.index)
            events.extend(detector_events)
        t_done = time.perf_counter()

        latency.record(
            frame.index,
            frame.timestamp,
            capture_ms=frame.capture_ms,
            keypoints_ms=(t_feat - t_kpts) * 1000,
            features_ms=(t_done - t_feat) * 1000,
            total_ms=(t_done - frame.captured_at) * 1000,
        )
        if on_event:
            for event in events:
                on_event(event)
        if on_frame and on_frame(frame, observation, results) is False:
            break
        if max_frames and frame.index + 1 >= max_frames:
            break


def print_event(event) -> None:
    held = event.timestamp - event.onset
    print(
        f"[{event.timestamp:8.2f}s] CUE {event.state.upper():5s} {event.behaviour} "
        f"person {event.person} {event.part} (began {event.onset:.2f}s, {held:.2f}s ago)",
        flush=True,
    )


def make_overlay(face_roi=None):
    """Return an on_frame callback that shows keypoints, scores and states in a window."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    red, green = (0, 0, 255), (0, 200, 0)

    def on_frame(frame, observation, results):
        image = frame.image.copy()
        if face_roi:
            x, y, w, h = face_roi
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

        hands = results.get("hand_closure")
        for p, person in enumerate(observation.hands if hands else []):
            for s, side in enumerate(HAND_SIDES):
                if person.confidence[side].max(initial=0) <= 0:
                    continue
                coords = person.coords[side].astype(int)
                fist = hands["fist_state"][p, s] == 1.0
                color = red if fist else green
                for knuckle, tip in FINGERS.values():
                    cv2.line(image, tuple(coords[WRIST]), tuple(coords[tip]), color, 1)
                    cv2.circle(image, tuple(coords[knuckle]), 3, color, -1)
                label = f"{side} {hands['extension_ratio'][p, s]:.2f}{' FIST' if fist else ''}"
                cv2.putText(image, label, tuple(coords[WRIST] + [0, 20]), font, 0.5, color, 1)

        eyes = results.get("eye_closure")
        for p, face in enumerate(observation.faces if eyes else []):
            for s, side in enumerate(EYE_SIDES):
                color = red if eyes["closed"][p, s] == 1.0 else green
                cv2.polylines(image, [face.coords[side].astype(np.int32)], True, color, 1)
            cue = eyes["cue_state"][p] == 1.0
            left, right = eyes["ear"][p]
            anchor = face.coords["right_eye"][0].astype(int) + [0, -15]
            cv2.putText(
                image, f"EAR L {left:.2f} R {right:.2f}{' EYES CLOSED' if cue else ''}", tuple(anchor), font, 0.5,
                red if cue else green, 1,
            )  # fmt: skip

        cv2.imshow("nicetoolbox live", image)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    return on_frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run eye / hand closure detectors live and report per-frame latency.")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or a video file path.")
    parser.add_argument("--detect", default="eyes", help="Comma-separated detectors: eyes, hands.")
    parser.add_argument("--backend", choices=["mediapipe", "rtmlib"], default="mediapipe")
    parser.add_argument(
        "--device",
        default=None,
        help="mediapipe: cpu|gpu (default cpu); rtmlib: cpu|cuda (default cuda).",
    )
    parser.add_argument("--rtmlib-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument(
        "--face-roi", default=None, help="x,y,width,height: MediaPipe face model only looks at this region."
    )
    parser.add_argument("--mirrored", action="store_true", help="Frames are mirrored (selfie view).")
    parser.add_argument(
        "--eye-threshold", type=float, default=0.1, help="Fixed max EAR for a closed eye, used with --eye-relative 0."
    )
    parser.add_argument(
        "--eye-relative",
        type=float,
        default=0.6,
        help="Closed below this fraction of the eye's recent open EAR; 0 = use only --eye-threshold.",
    )
    parser.add_argument("--eye-min-duration", type=float, default=1.0, help="Seconds both eyes stay closed per cue.")
    parser.add_argument("--hand-threshold", type=float, default=1.2, help="Max finger extension ratio for a fist.")
    parser.add_argument("--hand-min-duration", type=float, default=0.5, help="Seconds a fist is held per cue.")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true", help="Process a video file as fast as possible.")
    parser.add_argument("--show", action="store_true", help="Show an overlay window (press q to quit).")
    parser.add_argument("--out-dir", default=None, help="Write latency.csv, cues.jsonl and summary.json here.")
    parser.add_argument("--report-every", type=float, default=5.0, help="Seconds between latency printouts (0 = off).")
    args = parser.parse_args(argv)

    args.detect = [d.strip() for d in args.detect.split(",") if d.strip()]
    unknown = [d for d in args.detect if d not in DETECTORS]
    if unknown or not args.detect:
        parser.error(f"--detect takes a comma-separated list of {sorted(DETECTORS)}, got {args.detect}")
    if args.face_roi:
        args.face_roi = tuple(int(v) for v in args.face_roi.split(","))
        if len(args.face_roi) != 4:
            parser.error("--face-roi takes x,y,width,height")
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)

    parts = tuple(DETECTORS[d] for d in args.detect)
    if args.backend == "mediapipe":
        backend = create_backend(
            "mediapipe", parts=parts, mirrored=args.mirrored, device=args.device or "cpu", face_roi=args.face_roi
        )
    else:
        backend = create_backend("rtmlib", parts=parts, mode=args.rtmlib_mode, device=args.device or "cuda")
    detectors = []
    if "eyes" in args.detect:
        detectors.append(
            LiveEyeClosure(args.eye_threshold, args.min_confidence, args.eye_min_duration, relative=args.eye_relative)
        )
    if "hands" in args.detect:
        detectors.append(LiveHandClosure(args.hand_threshold, args.min_confidence, args.hand_min_duration))

    cue_file = None
    csv_path = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, "latency.csv")
        cue_file = open(os.path.join(args.out_dir, "cues.jsonl"), "w")  # noqa: SIM115
    latency = LatencyTracker(csv_path)

    def on_event(event):
        print_event(event)
        if cue_file:
            cue_file.write(json.dumps(dataclasses.asdict(event)) + "\n")
            cue_file.flush()

    overlay = make_overlay(args.face_roi) if args.show else None
    last_report = [time.perf_counter()]

    def on_frame(frame, observation, results):
        if args.report_every and time.perf_counter() - last_report[0] >= args.report_every:
            last_report[0] = time.perf_counter()
            print(LatencyTracker.format(latency.summary()), flush=True)
        return overlay(frame, observation, results) if overlay else True

    try:
        run(
            open_source(args.source, realtime=not args.no_realtime),
            backend,
            detectors,
            latency,
            on_event=on_event,
            on_frame=on_frame,
            max_frames=args.max_frames,
        )
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        latency.close()
        if cue_file:
            cue_file.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    summary = latency.summary()
    summary["backend"] = backend.name
    summary["detectors"] = args.detect
    print("Final:", LatencyTracker.format(summary))
    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    return summary


if __name__ == "__main__":
    main()
