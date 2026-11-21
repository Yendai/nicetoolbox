"""
A small local web page for the proof of concept: live camera view, cue state, and the band.

The live loop runs as before. With --ui it also starts this server on the same machine:

    GET /            the page (ui/index.html)
    GET /events      Server-Sent Events: each cue as it fires, plus a status line about once a second
    GET /video.mjpg  the overlay frames as a motion-JPEG stream (only encoded while a page watches)

The page connects to the wristband itself with Web Bluetooth (Chrome or Edge) and forwards each
cue's band command over the Nordic UART service. Python never talks Bluetooth: it only says
which command a cue maps to (BAND_COMMANDS below), so the mapping lives in one tested place.

Only the standard library is used. The server listens on 127.0.0.1 only, because the video shows
therapy clients and must not be reachable from the network.
"""

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(__file__), "ui", "index.html")

# (behaviour, part, state) -> one-character command the band understands. See
# nicetoolbox/live/band/band_receiver/band_receiver.ino for what the band does with each one.
BAND_COMMANDS = {
    ("body_crossing", "arms", "start"): "A",
    ("body_crossing", "arms", "end"): "a",
}


def band_command(event):
    """The band command for a CueEvent, or None if the band does not take this cue (yet)."""
    return BAND_COMMANDS.get((event.behaviour, event.part, event.state))


class UiServer:
    """
    Serves the page, pushes cue events to it, and streams the latest overlay frame.

    Args:
        port (int): Port on 127.0.0.1. 0 picks a free one (see .url).
    """

    def __init__(self, port: int = 8765):
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._frame = None  # latest JPEG bytes
        self._frame_id = 0
        self._frame_ready = threading.Condition()
        self._viewers = 0
        self._closed = False
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler_class())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/"

    @property
    def wants_video(self) -> bool:
        """True while at least one page shows the video, so frames are worth encoding."""
        return self._viewers > 0

    def publish(self, message: dict) -> None:
        """Send one JSON message to every open page."""
        data = json.dumps(message)
        with self._lock:
            for client in self._clients:
                client.put(data)

    def publish_event(self, event) -> None:
        """Send a CueEvent, with the band command it maps to (or null)."""
        message = {"type": "cue", "command": band_command(event)}
        message.update(vars(event))
        self.publish(message)

    def publish_frame(self, jpeg: bytes) -> None:
        with self._frame_ready:
            self._frame = jpeg
            self._frame_id += 1
            self._frame_ready.notify_all()

    def close(self) -> None:
        self._closed = True
        self.publish({"type": "bye"})
        with self._lock:
            for client in self._clients:
                client.put(None)
        with self._frame_ready:
            self._frame_ready.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for cues and latency
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send_page()
                elif path == "/events":
                    self._send_events()
                elif path == "/video.mjpg":
                    self._send_video()
                else:
                    self.send_error(404)

            def _send_page(self):
                with open(PAGE, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                client = queue.Queue()
                with server._lock:
                    server._clients.append(client)
                try:
                    self.wfile.write(b'data: {"type": "hello"}\n\n')
                    self.wfile.flush()
                    while True:
                        try:
                            data = client.get(timeout=15)
                        except queue.Empty:
                            data = ""  # a comment line keeps idle connections open
                        if data is None:
                            break
                        self.wfile.write(f"data: {data}\n\n".encode() if data else b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with server._lock:
                        server._clients.remove(client)

            def _send_video(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with server._lock:
                    server._viewers += 1
                last = 0  # frame ids start at 1, so this waits for the first frame
                try:
                    while not server._closed:
                        with server._frame_ready:
                            fresh = lambda seen=last: server._frame_id != seen or server._closed  # noqa: E731
                            server._frame_ready.wait_for(fresh, 2)
                            if server._closed or server._frame_id == last:
                                continue
                            jpeg, last = server._frame, server._frame_id
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with server._lock:
                        server._viewers -= 1

        return Handler
