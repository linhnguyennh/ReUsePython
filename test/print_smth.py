import cv2
import numpy as np
from ultralytics import YOLO


def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Don't multiply by 255 here
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB (not RGB to BGR)
    # Letterboxing should be applied to maintain aspect ratio
    orig_shape = image.shape[:2]
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0  # Normalize to 0-1
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    return image, orig_shape

def postprocess_obb(output, orig_shape, conf_thresh=0.25):
    # Transpose from [1, 6, 8400] to [1, 8400, 6]
    predictions = output[0].transpose((0, 2, 1))[0]
    
    # Filter by confidence
    mask = predictions[:, 4] > conf_thresh
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []
    
    # Get boxes [cx, cy, w, h, angle] and scores
    boxes = np.concatenate([predictions[:, :4], predictions[:, 5:6]], axis=1)
    scores = predictions[:, 4]
    class_ids = predictions[:, 5].astype(np.int32)
    
    # Scale boxes to original image size
    scale_x, scale_y = orig_shape[1] / 640, orig_shape[0] / 640
    boxes[:, 0] *= scale_x  # cx
    boxes[:, 1] *= scale_y  # cy
    boxes[:, 2] *= scale_x  # w
    boxes[:, 3] *= scale_y  # h
    
    return boxes, scores, class_ids
# 2) Load exported NCNN model with Ultralytics runtime
def main():

    image_path = r"data\yolo_retrain_251017\morrow00043.png"

    image, orig_shape = preprocess_image(image_path)


    model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203_ncnn_model", task='obb')  # or your custom *_ncnn_model directory

    # 3) Inference
    results = model(image_path, imgsz=640,show=True)
    print(results[0].obb)

if __name__ == "__main__":
    main()