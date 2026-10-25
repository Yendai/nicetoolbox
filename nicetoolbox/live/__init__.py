"""
Live module. Runs feature detectors frame by frame on a camera or video stream, emits cues
as behaviours start and stop, and measures per-frame latency.

Only numpy-based helpers are imported from the batch toolbox, so this module runs in its
own lightweight environment (see requirements.txt) without the batch pipeline's pins.
"""
