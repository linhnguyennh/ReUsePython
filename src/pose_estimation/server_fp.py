import cv2
import numpy as np
import threading
import logging
from queue import Queue, Empty
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
from typing import Callable, Any, Optional
from ws_color_depth_helper import decode_frame, encode_pose
from foundationpose_class import FoundationPoseEstimator  # your class from earlier
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StreamServerWebSocket:
    def __init__(
        self,
        decoder: Optional[Callable[[bytes], Any]] = None,
        K: np.ndarray = None,
        mesh_file: str = None,
    ):
        self.decoder       = decoder or self._default_decoder
        self.K             = K
        self.display_queue = Queue(maxsize=1)
        self.frame_queue   = Queue(maxsize=1)
        self.pose_queue = Queue(maxsize=1)

        self._websocket = None

        @asynccontextmanager
        async def lifespan(app):
            asyncio.ensure_future(self._pose_sender())
            yield

        self.app           = FastAPI(lifespan=lifespan)
        self.setup_routes()

        # FoundationPose — only created if mesh_file is provided
        self.estimator = FoundationPoseEstimator(mesh_file=mesh_file) if mesh_file else None

    # ---------- route setup ----------
    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_client(websocket)

    # ---------- websocket handler — just receives and queues ----------
    async def handle_client(self, websocket: WebSocket):
        await websocket.accept()
        self._websocket = websocket
        logger.info("Client connected")
        try:
            while True:
                packet  = await websocket.receive_bytes()
                decoded = self.decoder(packet)
                if decoded is None:
                    continue

                # Only feed frame_queue — display_queue is pose loop's responsibility
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put_nowait(decoded)

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            self._websocket = None
    # ---------- default decoder ----------
    @staticmethod
    def _default_decoder(buffer: bytes):
        try:
            return decode_frame(buffer)
        except Exception as e:
            logger.error(f"Decode error: {e}")
            return None

    # ---------- pose estimation loop (worker thread) ----------
    def _pose_loop(self):
        if self.estimator is None or self.K is None:
            logger.warning("No estimator or K — pose loop not running")
            return

        logger.info("Pose loop started")

        while True:
            try:
                decoded = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            rgb = decoded.get("rgb")
            depth = decoded.get("depth")
            mask = decoded.get("mask")

            if rgb is None or depth is None:
                continue

            pose = None

            # -------------------------
            # Registration
            # -------------------------
            if not self.estimator._initialized:
                if mask is None or np.sum(mask > 0) < 500:
                    continue

                try:
                    logger.info("Registering object...")
                    pose = self.estimator.register(rgb, depth, mask, self.K)

                    if pose is None or np.isnan(pose).any():
                        continue

                    logger.info("Registration done")

                except Exception as e:
                    logger.error(f"Registration failed: {e}")
                    continue

            # -------------------------
            # Tracking
            # -------------------------
            else:
                try:
                    pose = self.estimator.track(rgb, depth, self.K)

                except Exception as e:
                    logger.error(f"Tracking error: {e}")
                    # self.estimator._initialized = False
                    continue

            # -------------------------
            # Push to display (RAW DATA ONLY)
            # -------------------------
            self._push_display(rgb, depth, mask, pose)

            if pose is not None:
                # CODE FOR PUSHING POSE DATA BACK TO CLIENT HERE
                # QUEUE
                #
                if self.pose_queue.full():
                    try:
                        self.pose_queue.get_nowait()
                    except Empty:
                        pass

                self.pose_queue.put_nowait(pose)
                

                t = pose[:3, 3]
                logger.info(f"t (m): x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f}")
                logger.info(f"Pose matrix: {pose}")

    async def _pose_sender(self):
        while True:
            await asyncio.sleep(0.001)
            if self._websocket is None:
                continue
            
            #Get pose from queue
            try:
                pose = self.pose_queue.get_nowait()
            except Empty:
                continue
            
            #Send pose back to client
            try:
                await self._websocket.send_bytes(encode_pose(pose))
                logger.info(f"Pose sent!")
            except Exception as e:
                logger.error(f"Pose send error: {e}")

    # -------------------------
    # Safe queue push
    # -------------------------
    def _push_display(self, rgb, depth, mask, pose):
        if self.display_queue.full():
            try:
                self.display_queue.get_nowait()
            except Empty:
                pass

        self.display_queue.put_nowait({
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "pose": pose
        })

    def run_display(self):
        while True:
            try:
                display = self.display_queue.get(timeout=1.0)
            except Empty:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            rgb   = display.get("rgb")
            depth = display.get("depth")
            mask  = display.get("mask")
            pose  = display.get("pose")

            if rgb is None:
                continue

            h, w = rgb.shape[:2]

            # =========================================================
            # 1. MAIN VIEW — stay in RGB until pose overlay
            # =========================================================
            show = rgb.copy()

            # Mask overlay (RGB-safe: green is (0,255,0) in both spaces)
            if mask is not None:
                mask_arr = np.asarray(mask)
                mask_bin = (mask_arr > 0).astype(np.uint8)
                if mask_bin.shape != (h, w):
                    mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                green_layer = np.zeros_like(show)
                green_layer[mask_bin > 0] = (0, 255, 0)
                alpha = 0.4
                show = np.where(
                    mask_bin[:, :, np.newaxis] > 0,
                    cv2.addWeighted(show, 1 - alpha, green_layer, alpha, 0),
                    show
                ).astype(np.uint8)

            # Pose overlay — visualize() expects RGB in, returns BGR out
            if (
                pose is not None and
                isinstance(pose, np.ndarray) and
                pose.shape == (4, 4) and
                not np.isnan(pose).any()
            ):
                try:
                    show = self.estimator.visualize(show, pose, self.K)
                    # show is now BGR
                except Exception as e:
                    logger.error(f"Visualization error: {e}")
                    show = show[..., ::-1]  # still convert so everything below is BGR
            else:
                show = show[..., ::-1]  # RGB → BGR for cv2

            # =========================================================
            # 2. DEPTH PANEL — applyColorMap returns BGR, keep it BGR
            # =========================================================
            if depth is not None:
                depth_arr = np.asarray(depth).astype(np.float32)
                depth_vis = np.clip(depth_arr, 0, 2000)
                depth_vis = (depth_vis / 2000.0 * 255).astype(np.uint8)
                depth_panel = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                if depth_panel.shape[:2] != (h, w):
                    depth_panel = cv2.resize(depth_panel, (w, h))
            else:
                depth_panel = np.zeros((h, w, 3), dtype=np.uint8)

            # =========================================================
            # 3. MASK PANEL
            # =========================================================
            if mask is not None:
                mask_panel = (mask_bin * 255)[..., np.newaxis].repeat(3, axis=2)
            else:
                mask_panel = np.zeros((h, w, 3), dtype=np.uint8)

            # =========================================================
            # 4. STACK AND DISPLAY — all panels are BGR at this point
            # =========================================================
            # Add labels
            def label(img, text):
                out = img.copy()
                cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                return out

            row = np.concatenate([
                label(show,        "RGB + Mask + Pose"),
                label(depth_panel, "Depth"),
                label(mask_panel,  "Mask"),
            ], axis=1)

            cv2.imshow("Realsense D435i -> FoundationPose + YOLOv11-seg -> 6D pose", row)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

# ---------- run ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    K = np.array([
        [607.747,   0,      315.685],
        [0,        607.918, 252.557],
        [0,         0,            1]
        ], dtype=np.float64)# your RealSense color intrinsics as (3,3) numpy array

    server = StreamServerWebSocket(
        K=K,
        mesh_file="mesh/morrow.obj",
    )

    # uvicorn on daemon thread
    uvicorn_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": server.app, "host": "0.0.0.0", "port": 8000},
        daemon=True,
    )
    uvicorn_thread.start()

    # pose estimation on worker thread
    pose_thread = threading.Thread(target=server._pose_loop, daemon=True)
    pose_thread.start()

    # display on main thread
    server.run_display()