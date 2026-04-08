from ultralytics import YOLO

model = YOLO(r"models\focus1\retrain_obb_BIGMAP_260108\train\weights\morrow_obb_260108.pt")

# Export the model
# model.export(format="onnx", opset = 18, imgsz = 640)
model.export(format="ncnn", imgsz = (480,640))