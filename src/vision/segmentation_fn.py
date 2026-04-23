from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import logging
import torch
import math

def segment_object(model: YOLO, rgb_image, **yolo_args):
    results = model(rgb_image, **yolo_args)[0]

    if results.masks is None:
        return None, None

    h, w = rgb_image.shape[:2]
    
    # 1. Get all valid detections
    masks = results.masks.data.cpu().numpy()  # [N, H, W]
    boxes = results.boxes.xyxy.cpu().numpy() # [N, 4]


    # 2. Selection Heuristic: Pick the largest mask (usually the most visible battery)
    # Alternatively, use your argmax(confs) if you prefer confidence
    areas = [np.sum(m) for m in masks]
    best_idx = np.argmax(areas)

    # 3. Processing the best mask
    raw_mask = masks[best_idx]
    binary = (raw_mask > 0.5).astype(np.uint8) * 255
    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. Geometric Refinement (Convex Hull)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        refined_mask = np.zeros_like(binary)
        cv2.drawContours(refined_mask, [hull], -1, 255, -1)
    else:
        refined_mask = binary

    # 5. Boundary Erosion (Safety margin to avoid background pixels)
    # This prevents FoundationPose from seeing the "table"
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.erode(refined_mask, kernel, iterations=1)

    # Return the mask and the bounding box (useful for cropping later)
    return refined_mask, boxes[best_idx]

def segment_all_objects(model: YOLO, rgb_image, conf_threshold=0.5, **yolo_args):
    results = model(rgb_image, conf_threshold=conf_threshold, **yolo_args)[0]
    
    if results.masks is None:
        return []

    h, w = rgb_image.shape[:2]
    detections = []

    # Iterate through each detected instance
    for i in range(len(results.masks)):
        mask_data = results.masks.data[i].cpu().numpy()
        conf = results.boxes.conf[i].cpu().item()
        box = results.boxes.xyxy[i].cpu().numpy() # [x1, y1, x2, y2]

        # Resize and refine
        binary = (mask_data > 0.5).astype(np.uint8) * 255
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Optional: Apply the Convex Hull cleanup here to each mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            refined_mask = np.zeros_like(binary)
            cv2.drawContours(refined_mask, [hull], -1, 255, -1)
            
            detections.append({
                "mask": refined_mask,
                "box": box,
                "confidence": conf,
                "id": i
            })

    return detections