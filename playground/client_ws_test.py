import cv2
import base64
import numpy as np
from websockets.sync.client import connect
import json

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def main():
    with connect("ws://localhost:8000/ws") as ws:
        
        # simple wrapper to send frames
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            data = encode_image(frame)

            ws.send(data)

            # optional local view
            cv2.imshow("Client - Sending", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        ws.close()


if __name__ == "__main__":
    main()