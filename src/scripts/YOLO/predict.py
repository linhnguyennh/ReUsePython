from ultralytics import YOLO

image_path= r"..\test_imgs\7.jpg"
model = YOLO(r"models\focus1\retrain_obb_251113\train6\weights\morrow_obb_251119.pt")
#print(model.names)

results = model(source=0,conf=0.1, show = True)