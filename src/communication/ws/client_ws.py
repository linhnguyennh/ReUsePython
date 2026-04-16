import cv2
import numpy as np
import threading
import logging
import sys
from pathlib import Path
from queue import Queue, Empty
from websockets.sync.client import connect
from typing import Callable, Optional

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

try:
    from src.communication.ws.ws_color_depth_helper import encode_frame
except ImportError:
    from ws_color_depth_helper import encode_frame
from src.vision.realsense_stream import RealSenseStream

logger = logging.getLogger(__name__)


class StreamClientWebSocket:
    def __init__(
        self,
        ws_url: str,
        frame_queue: Queue,
        encoder: Optional[Callable[..., bytes]] = None,
    ):
        self.ws_url      = ws_url
        self.frame_queue = frame_queue
        self.encoder     = encoder or self._default_encoder
        self.ws          = None
        self._running    = False

    # ---------- default encoder ----------
    @staticmethod
    def _default_encoder(rgb, depth=None) -> bytes:
        return encode_frame(rgb, depth)

    # ---------- setup ----------
    def _connect_ws(self):
        self.ws = connect(self.ws_url)

    # ---------- send loop ----------
    def _send_loop(self):
        logger.info("Send loop started")
        while self._running:
            try:
                color_frame, depth_frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            rgb   = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            try:
                packet = self.encoder(rgb, depth)
                self.ws.send(packet)
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

        logger.info("Send loop stopped")

    # ---------- run ----------
    def run(self):
        self._connect_ws()
        self._running = True

        send_thread = threading.Thread(target=self._send_loop, daemon=True)
        send_thread.start()

        try:
            while self._running:
                if cv2.waitKey(1) & 0xFF == 27:
                    self._running = False
        finally:
            send_thread.join(timeout=2)
            self.close()

    # ---------- cleanup ----------
    def close(self):
        self._running = False
        if self.ws:
            self.ws.close()
        logger.info("Client closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rs_stream = RealSenseStream(
        width=640, height=480, fps=30,
        enable_decimation=True,
        enable_spatial=True,
        enable_temporal=True,
    )
    rs_stream.start()

    client = StreamClientWebSocket(
        ws_url="ws://localhost:8000/ws",
        frame_queue=rs_stream.frame_queue,
    )

    try:
        client.run()
    finally:
        rs_stream.stop()