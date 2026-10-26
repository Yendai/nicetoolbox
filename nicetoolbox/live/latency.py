"""
Per-frame latency bookkeeping for the live loop.
"""

import csv
import time

import numpy as np

STAGES = ("capture_ms", "keypoints_ms", "features_ms", "total_ms")


class LatencyTracker:
    """
    Records how long each stage took per frame and summarises it.

    total_ms runs from the moment the frame was captured until its features (and any cues) are
    ready, so it is the delay the loop adds before a cue could be sent. It does not include the
    camera's own exposure and transfer time, or the time to deliver the cue.

    Args:
        csv_path (str): Optional path to write one row per frame.
    """

    def __init__(self, csv_path: str = None):
        self.rows: list[dict] = []
        self._file = None
        self._writer = None
        if csv_path:
            self._file = open(csv_path, "w", newline="")  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=["frame", "timestamp", *STAGES])
            self._writer.writeheader()
        self._started = time.perf_counter()

    def record(self, frame: int, timestamp: float, **stages_ms: float) -> dict:
        row = {"frame": frame, "timestamp": round(timestamp, 4)}
        row.update({k: round(stages_ms[k], 3) for k in STAGES})
        self.rows.append(row)
        if self._writer:
            self._writer.writerow(row)
        return row

    def summary(self) -> dict:
        """Median, 95th percentile and max per stage, plus processed frames per second."""
        rows = self.rows
        if not rows:
            return {}
        out = {"frames": len(rows)}
        for stage in STAGES:
            values = np.array([r[stage] for r in rows])
            out[stage] = {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
            }
        elapsed = time.perf_counter() - self._started
        out["fps"] = len(self.rows) / elapsed if elapsed > 0 else float("nan")
        return out

    @staticmethod
    def format(summary: dict) -> str:
        if not summary:
            return "no frames processed"
        parts = [f"{summary['frames']} frames, {summary['fps']:.1f} fps"]
        for stage in STAGES:
            s = summary[stage]
            parts.append(f"{stage[:-3]} p50 {s['p50']:.1f} / p95 {s['p95']:.1f} / max {s['max']:.1f} ms")
        return " | ".join(parts)

    def close(self) -> None:
        if self._file:
            self._file.close()
