from ultralytics import YOLO

model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.pt")

if __name__ == '__main__':
    model.train(data=r"data\annotated\yolo_retrain_obb_260108.yolov11\data.yaml", 
                epochs=500, patience=50, batch=32, imgsz=640, cache = 'disk',
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"models\focus1\retrain_obb_BIGMAP_260108")

