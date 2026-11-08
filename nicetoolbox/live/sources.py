"""
Frame sources for the live loop: a webcam by index, or a video file played back in real time.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """One captured frame.

    Attributes:
        index (int): Frame number since the source was opened, from 0.
        image (np.ndarray): BGR image, (height, width, 3).
        timestamp (float): Seconds since the source was opened, on the stream's clock. For a
            webcam this is wall time; for a video file it is the frame's position in the video.
        captured_at (float): time.perf_counter() when the frame was handed to the loop. All
            latency is measured from here.
        capture_ms (float): Time spent reading the frame (not the real-time playback wait).
    """

    index: int
    image: np.ndarray
    timestamp: float
    captured_at: float
    capture_ms: float


def open_source(source: str, realtime: bool = True) -> Iterator[Frame]:
    """
    Yield frames from a webcam or a video file.

    Args:
        source (str): A webcam index such as "0", or a path to a video file.
        realtime (bool): For a video file, wait between frames so playback runs at the video's
            frame rate, as a camera would deliver it. Set False to process as fast as possible.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    import cv2

    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source)) if is_camera else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'.")
    if is_camera:
        # Keep only the newest frame in the driver queue, so a slow frame does not make every
        # later frame stale.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start = time.perf_counter()
    index = 0
    try:
        while True:
            t0 = time.perf_counter()
            ok, image = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            read_ms = (t1 - t0) * 1000
            if is_camera:
                timestamp = t1 - start
            else:
                timestamp = index / fps
                if realtime:
                    # Sleep until this frame's due time; a frame that is already late is not delayed.
                    delay = timestamp - (t1 - start)
                    if delay > 0:
                        time.sleep(delay)
                        t1 = time.perf_counter()
            yield Frame(index=index, image=image, timestamp=timestamp, captured_at=t1, capture_ms=read_ms)
            index += 1
    finally:
        cap.release()
