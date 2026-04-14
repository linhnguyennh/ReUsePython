import base64
import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel


# uvicorn src.communication.http.server_test:app --host localhost --port 8000


app = FastAPI()


# ===== Request schema =====
class ImageRequest(BaseModel):
    image: str  # base64 encoded JPEG


# ===== Decode function =====
def decode_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


# ===== Endpoint =====
@app.post("/test")
def receive_image(req: ImageRequest):
    img = decode_image(req.image)

    print("Received image shape:", img.shape)
    print("Dtype:", img.dtype)

    # simple sanity check
    mean_val = float(np.mean(img))

    # 👇 Display the image
    cv2.imwrite("data/debug.png", img)
    return {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "mean_pixel_value": mean_val
    }