import cv2
import numpy as np
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Callable, Any, Optional
from src.communication.ws.ws_helper import decode_frame


class StreamServerWebSocket:
    def __init__(self, decoder: Optional[Callable[[bytes], Any]] = None):
        self.decoder = decoder or self._default_decoder
        self.app = FastAPI()
        self.setup_routes()

    # ---------- route setup ----------
    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_client(websocket)

    # ---------- main handler ----------
    # async def handle_client(self, websocket: WebSocket):
    #     await websocket.accept()
    #     print("Client connected")

    #     try:
    #         while True:
    #             packet = await websocket.receive_bytes()

    #             decoded = self.decoder(packet)

    #             if decoded is None:
    #                 print("Decode failed")
    #                 continue

    #             if isinstance(decoded, dict):
    #                 rgb = decoded.get("rgb")
    #                 depth = decoded.get("depth")
    #                 mask = decoded.get("mask")

    #             if rgb is None:
    #                 print("Decode failed")
    #                 continue
    #             cv2.imshow("Server - Live Stream", rgb)
    #             img = rgb.copy()
    #             if depth is not None:
    #                 depth_display = cv2.convertScaleAbs(depth, alpha=0.03)
    #                 cv2.imshow("Server - Depth Stream", depth_display)

    #             if mask is not None:
    #                 binary_mask = (mask > 0).astype(np.uint8)
    #                 colored = np.zeros_like(img)
    #                 colored[:, :, 1] = binary_mask * 255

    #                 img = cv2.addWeighted(img, 1.0, colored, 0.5, 0)
    #                 cv2.imshow("Server - Mask Stream", img)

    #             if cv2.waitKey(1) & 0xFF == 27:
    #                 break

    #     except WebSocketDisconnect:
    #         print("Client disconnected")

    #     finally:
    #         cv2.destroyAllWindows()
    async def handle_client(self, websocket: WebSocket):
        await websocket.accept()
        print("Client connected")

        try:
            while True:
                packet = await websocket.receive_bytes()
                decoded = self.decoder(packet)

                if decoded is None:
                    continue

                rgb = decoded.get("rgb")
                depth = decoded.get("depth")
                mask = decoded.get("mask")

                if rgb is None or depth is None:
                    print("Missing RGB or Depth for overlay")
                    continue

                # 1. Prepare Depth for Visualization
                # Normalize 16-bit depth to 8-bit (0-255)
                # We clip at ~2 meters (2000mm) to get better contrast on the battery
                depth_clipped = np.clip(depth, 0, 2000) / 2000 * 255
                depth_8bit = depth_clipped.astype(np.uint8)
                
                # Apply a colormap (JET: Blue is far, Red is close)
                depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # 2. Ensure dimensions match (D435i usually does this, but good to safety check)
                if rgb.shape[:2] != depth_color.shape[:2]:
                    depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]))

                # 3. Create the Overlay (50% RGB, 50% Depth)
                overlay = cv2.addWeighted(rgb, 0.6, depth_color, 0.4, 0)

                # 4. Optional: Draw the Mask Contour to check alignment of all three
                if mask is not None:
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
                    # Find the outlines
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw the green outline on your RGB/Depth overlay
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                    if mask.dtype != np.uint8:
                        mask_vis = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                    else:
                        mask_vis = mask

                    # Convert to clean binary (0 or 255)
                    mask_black_bg = np.where(mask_vis > 127, 255, 0).astype(np.uint8)

                    # Resize if needed (important for alignment debugging)
                    if mask_black_bg.shape[:2] != rgb.shape[:2]:
                        mask_black_bg = cv2.resize(mask_black_bg, (rgb.shape[1], rgb.shape[0]))

                    cv2.imshow("Mask (Black BG)", mask_black_bg)

                # Display
                cv2.imshow("Validation - RGB/Depth Alignment", overlay)
                cv2.imshow("Depth Only", depth_color)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except WebSocketDisconnect:
            print("Client disconnected")
        finally:
            cv2.destroyAllWindows()

    # ---------- default decoder ----------
    @staticmethod
    def _default_decoder(buffer: bytes):
        return decode_frame(buffer)

    # ---------- decoding ----------


# ---------- run ----------
if __name__ == "__main__":
    # Use default decoder for encode_frame packets from the new client
    server = StreamServerWebSocket()
    
    # Or inject custom decoder:
    # from .ws_color_depth_helper import decode_rgb
    # custom_decoder = lambda buffer: decode_rgb(buffer)  # Add custom processing here
    # server = StreamServerWebSocket(decoder=custom_decoder)
    
    uvicorn.run(server.app, host="localhost", port=1000)