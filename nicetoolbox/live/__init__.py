"""
Live module. Runs eye and hand closure detectors frame by frame on a camera or video stream,
emits cues as behaviours start and stop, and measures per-frame latency.

Only light helpers (numpy scoring functions, a pydantic schema) are imported from the batch
toolbox, so this module runs in its own environment (see requirements.txt) without the batch
pipeline's pins.
"""
