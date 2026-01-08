from ultralytics import YOLO

model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.pt")

# Export the model
model.export(format="onnx", opset = 18, imgsz = 640,simplify=True)