# ws_color_depth_helper.py
import struct
import json
import numpy as np
import cv2
from typing import Any

# Header: magic(H) + rgb_size(I) + depth_size(I) + mask_size(I)
HEADER_FORMAT = '>HIIII'
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)
MAGIC         = 0xAB01

# ── RGB only ──────────────────────────────────────────────────────────────────

def encode_rgb(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def decode_rgb(buffer: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)


# def encode_depth(depth):
#     return depth.tobytes()

# def decode_depth(buffer, shape):
#     depth = np.frombuffer(buffer, dtype=np.uint16).reshape(shape)
#     return depth


# ── RGB + Depth + optional metadata ──────────────────────────────────────────

def encode_frame(
    rgb: np.ndarray,
    depth: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rgb_quality: int = 85,
) -> bytes:
    _, rgb_buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, rgb_quality])
    rgb_bytes = rgb_buf.tobytes()

    depth_bytes = cv2.imencode('.png', depth.astype(np.uint16))[1].tobytes() if depth is not None else b''

    # Mask is binary (0/1 or 0/255) — PNG is lossless and compact for sparse data
    mask_bytes = cv2.imencode('.png', mask.astype(np.uint8))[1].tobytes() if mask is not None else b''

    header = struct.pack(HEADER_FORMAT, MAGIC, len(rgb_bytes), len(depth_bytes), len(mask_bytes))
    return header + rgb_bytes + depth_bytes + mask_bytes


def decode_frame(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(data)} bytes")

    magic, rs, ds, ms = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic:#06x}")

    offset      = HEADER_SIZE
    rgb_bytes   = data[offset : offset + rs]; offset += rs
    depth_bytes = data[offset : offset + ds]; offset += ds
    mask_bytes  = data[offset : offset + ms]

    rgb   = cv2.imdecode(np.frombuffer(rgb_bytes,   np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED) if ds > 0 else None
    mask  = cv2.imdecode(np.frombuffer(mask_bytes,  np.uint8), cv2.IMREAD_GRAYSCALE) if ms > 0 else None

    return {"rgb": rgb, "depth": depth, "mask": mask}