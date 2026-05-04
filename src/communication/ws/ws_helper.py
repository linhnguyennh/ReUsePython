# ws_color_depth_helper.py
import struct
import json
import numpy as np
import cv2
from typing import Any

# Header: magic(H) + rgb_size(I) + depth_size(I) + mask_size(I)
FRAME_HEADER_FORMAT = '>HIII'
FRAME_HEADER_SIZE   = struct.calcsize(FRAME_HEADER_FORMAT)
FRAME_MAGIC         = 0xAB01

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

    header = struct.pack(FRAME_HEADER_FORMAT, FRAME_MAGIC, len(rgb_bytes), len(depth_bytes), len(mask_bytes))
    return header + rgb_bytes + depth_bytes + mask_bytes


def decode_frame(data: bytes) -> dict:
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(data)} bytes")

    magic, rs, ds, ms = struct.unpack(FRAME_HEADER_FORMAT, data[:FRAME_HEADER_SIZE])

    if magic != FRAME_MAGIC:
        raise ValueError(f"Bad magic: {magic:#06x}")

    offset      = FRAME_HEADER_SIZE
    rgb_bytes   = data[offset : offset + rs]; offset += rs
    depth_bytes = data[offset : offset + ds]; offset += ds
    mask_bytes  = data[offset : offset + ms]

    rgb   = cv2.imdecode(np.frombuffer(rgb_bytes,   np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED) if ds > 0 else None
    mask  = cv2.imdecode(np.frombuffer(mask_bytes,  np.uint8), cv2.IMREAD_GRAYSCALE) if ms > 0 else None

    return {"rgb": rgb, "depth": depth, "mask": mask}


# Header: magic(H) + type(H) + payload_size(I)
POSE_HEADER_FORMAT = '>HHI'
POSE_HEADER_SIZE   = struct.calcsize(POSE_HEADER_FORMAT)
POSE_MAGIC         = 0xAB02  # different from image magic 0xAB01

# Message types
TYPE_POSE = 0x0001

def encode_pose(pose: np.ndarray) -> bytes:
    assert pose.shape == (4, 4), f"Expected (4,4), got {pose.shape}"
    payload = pose.astype(np.float32).tobytes()          # 16 × 4 = 64 bytes, row-major
    header  = struct.pack(POSE_HEADER_FORMAT, POSE_MAGIC, TYPE_POSE, len(payload))
    return header + payload                              # 8 + 64 = 72 bytes total

def decode_pose(data: bytes) -> dict:
    if len(data) < POSE_HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(data)} bytes")

    magic, msg_type, payload_size = struct.unpack(POSE_HEADER_FORMAT, data[:POSE_HEADER_SIZE])

    if magic != POSE_MAGIC:
        raise ValueError(f"Bad magic: {magic:#06x}")

    payload = data[POSE_HEADER_SIZE : POSE_HEADER_SIZE + payload_size]

    if msg_type == TYPE_POSE:
        if payload_size != 64:
            raise ValueError(f"Expected 64 bytes for pose, got {payload_size}")
        pose = np.frombuffer(payload, dtype=np.float32).reshape(4, 4)
        return {"type": TYPE_POSE, "pose": pose}

    raise ValueError(f"Unknown message type: {msg_type:#06x}")