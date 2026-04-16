import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

def decode_image(b64_string: str):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_text()

            img = decode_image(data)

            if img is None:
                print("Decode failed")
                continue

            cv2.imshow("Server - Live Stream", img)
            cv2.waitKey(1)

    except Exception as e:
        print("Connection closed:", e)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)