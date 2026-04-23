import cv2
import numpy as np
import threading
import logging
import sys
from pathlib import Path
from queue import Queue, Empty
from websockets.sync.client import connect
from typing import Callable, Optional
from ultralytics import YOLO
import time

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

try:
    from src.communication.ws.ws_color_depth_helper import encode_frame
except ImportError:
    from ws_color_depth_helper import encode_frame
from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import SegmentationWorker

logger = logging.getLogger(__name__)


class StreamClientWebSocket:
    def __init__(
        self,
        ws_url: str,
        frame_and_mask_queue: Queue,
        encoder: Optional[Callable[..., bytes]] = None,
    ):
        self.ws_url      = ws_url
        self._frame_and_mask_queue = frame_and_mask_queue
        self.encoder     = encoder or self._default_encoder
        self.ws          = None
        self._running    = False

    # ---------- default encoder ----------
    @staticmethod
    def _default_encoder(rgb, depth=None, mask=None) -> bytes:
        return encode_frame(rgb, depth, mask)

    # ---------- setup ----------
    def _connect_ws(self):
        self.ws = connect(self.ws_url)

    # ---------- send loop ----------
    def _send_loop(self):
        logger.info("Send loop started")
        while self._running:
            try:
                rgb_frame, depth_frame, mask = self._frame_and_mask_queue.get(timeout=1.0)
            except Empty:
                continue
            except ValueError:
                logger.error("Expected (rgb, depth, mask) tuple in queue")
                continue
            
            rgb_data   = np.asanyarray(rgb_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            if mask is not None:
                mask = mask.astype(np.uint8)
            try:
                packet = self.encoder(rgb_data, depth_data, mask)
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
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("KeyboardInterrupt received → stopping client")
            self._running = False

        finally:
            send_thread.join(timeout=2)
            self.close()
    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
        logger.info("Client closed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rs_stream = RealSenseStream(
        width=640, height=480, fps=30,
        enable_decimation=False,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=False
    )
    rs_stream.start()
    frame_queue = rs_stream.frame_queue
    seg_model = YOLO(r"models\focus1\260421\train\weights\best.pt")

    segmentor = SegmentationWorker(
        model = seg_model,
        frame_queue = frame_queue,
        max_queue_size=1,
        conf=0.6,
        device='cuda'
    )
    
    segmentor_out_queue = segmentor.mask_queue

    
    segmentor.start()


    client = StreamClientWebSocket(
        ws_url="ws://localhost:8000/ws", #port 1000 for local windows cross-script, port 8000 for script-to-container
        frame_and_mask_queue=segmentor_out_queue,
    )

    try:
        client.run()
    finally:
        segmentor.stop()
        rs_stream.stop()
        