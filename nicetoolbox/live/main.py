"""
Live loop: camera or video -> keypoints -> hand closure -> cues, with per-frame latency.

Run from the repository root:

    python -m nicetoolbox.live --source 0                  # webcam 0, MediaPipe on CPU
    python -m nicetoolbox.live --source clip.mp4 --show    # video file with an overlay window
    python -m nicetoolbox.live --backend rtmlib --device cuda --source 0
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np

from ..detectors.feature_detectors.hand_closure.utils import FINGERS, HAND_SIDES, WRIST
from .backends import create_backend
from .hand_closure import LiveHandClosure
from .latency import LatencyTracker
from .sources import open_source


def run(frames, backend, detector, latency, on_event=None, on_frame=None, max_frames=None) -> None:
    """
    Drive the loop over an iterable of sources.Frame.

    Args:
        on_event (callable): Called with each CueEvent as soon as its frame is processed.
        on_frame (callable): Called with (frame, people, ratio, state) after each frame, e.g. to
            draw an overlay. Its time is not counted in the latency.
        max_frames (int): Stop after this many frames.
    """
    for frame in frames:
        t_kpts = time.perf_counter()
        people = backend(frame.image, frame.timestamp)
        t_feat = time.perf_counter()
        ratio, state, events = detector.update(people, frame.timestamp, frame.index)
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
        if on_frame and on_frame(frame, people, ratio, state) is False:
            break
        if max_frames and frame.index + 1 >= max_frames:
            break


def print_event(event) -> None:
    held = event.timestamp - event.onset
    print(
        f"[{event.timestamp:8.2f}s] CUE {event.state.upper():5s} {event.behaviour} "
        f"person {event.person} {event.side} (began {event.onset:.2f}s, {held:.2f}s ago)",
        flush=True,
    )


def make_overlay(threshold: float):
    """Return an on_frame callback that shows keypoints, ratios and fist state in a window."""
    import cv2

    def on_frame(frame, people, ratio, state):
        image = frame.image.copy()
        for p, person in enumerate(people):
            for s, side in enumerate(HAND_SIDES):
                conf = person.confidence[side]
                if conf.max(initial=0) <= 0:
                    continue
                coords = person.coords[side].astype(int)
                fist = state[p, s] == 1.0
                color = (0, 0, 255) if fist else (0, 200, 0)
                for knuckle, tip in FINGERS.values():
                    cv2.line(image, tuple(coords[WRIST]), tuple(coords[tip]), color, 1)
                    cv2.circle(image, tuple(coords[knuckle]), 3, color, -1)
                label = f"{side} {ratio[p, s]:.2f}{' FIST' if fist else ''}"
                cv2.putText(image, label, tuple(coords[WRIST] + [0, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, f"threshold {threshold}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("nicetoolbox live", image)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    return on_frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the hand closure detector live and report per-frame latency.")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or a video file path.")
    parser.add_argument("--backend", choices=["mediapipe", "rtmlib"], default="mediapipe")
    parser.add_argument(
        "--device",
        default=None,
        help="mediapipe: cpu|gpu (default cpu); rtmlib: cpu|cuda (default cuda).",
    )
    parser.add_argument("--rtmlib-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--mirrored", action="store_true", help="Frames are mirrored (selfie view).")
    parser.add_argument("--threshold", type=float, default=1.2)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true", help="Process a video file as fast as possible.")
    parser.add_argument("--show", action="store_true", help="Show an overlay window (press q to quit).")
    parser.add_argument("--out-dir", default=None, help="Write latency.csv, cues.jsonl and summary.json here.")
    parser.add_argument("--report-every", type=float, default=5.0, help="Seconds between latency printouts (0 = off).")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)

    if args.backend == "mediapipe":
        backend = create_backend("mediapipe", mirrored=args.mirrored, device=args.device or "cpu")
    else:
        backend = create_backend("rtmlib", mode=args.rtmlib_mode, device=args.device or "cuda")
    detector = LiveHandClosure(args.threshold, args.min_confidence, args.min_duration)

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

    overlay = make_overlay(args.threshold) if args.show else None
    last_report = [time.perf_counter()]

    def on_frame(frame, people, ratio, state):
        if args.report_every and time.perf_counter() - last_report[0] >= args.report_every:
            last_report[0] = time.perf_counter()
            print(LatencyTracker.format(latency.summary()), flush=True)
        return overlay(frame, people, ratio, state) if overlay else True

    try:
        run(
            open_source(args.source, realtime=not args.no_realtime),
            backend,
            detector,
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
    summary["backend"] = args.backend
    print("Final:", LatencyTracker.format(summary))
    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    return summary


if __name__ == "__main__":
    main()
