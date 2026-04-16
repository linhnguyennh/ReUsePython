import cv2
import numpy as np
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Callable, Any, Optional

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

try:
    from src.communication.ws.ws_color_depth_helper import decode_frame
except ImportError:
    from ws_color_depth_helper import decode_frame


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
    async def handle_client(self, websocket: WebSocket):
        await websocket.accept()
        print("Client connected")

        try:
            while True:
                packet = await websocket.receive_bytes()

                decoded = self.decoder(packet)

                if decoded is None:
                    print("Decode failed")
                    continue

                if isinstance(decoded, dict):
                    rgb = decoded.get("rgb")
                    depth = decoded.get("depth")
                else:
                    rgb = decoded
                    depth = None

                if rgb is None:
                    print("Decode failed")
                    continue

                cv2.imshow("Server - Live Stream", rgb)
                if depth is not None:
                    depth_display = cv2.convertScaleAbs(depth, alpha=0.03)
                    cv2.imshow("Server - Depth Stream", depth_display)

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
    
    uvicorn.run(server.app, host="0.0.0.0", port=8000)