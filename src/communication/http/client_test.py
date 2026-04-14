import requests
import base64
import cv2

# ===== Encode function =====
def encode_image(img):
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


# ===== Load test image (simulate RealSense RGB) =====
img = cv2.imread("data/handeye_images/pose_000.png")  # replace with your frame
cv2.imshow("Input image",img)
cv2.waitKey(0)
if img is None:
    raise RuntimeError("Image not found")

print("Original shape:", img.shape)
print("Original dtype:", img.dtype)

# ===== Send request =====
url = "http://localhost:8000/test"

payload = {
    "image": encode_image(img)
}

response = requests.post(url, json=payload)

print("Server response:")
print(response.json())