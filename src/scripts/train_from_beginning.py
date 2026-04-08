from ultralytics import YOLO

model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data=r"data\annotated\yolo_retrain_obb_260108.yolov11\data.yaml", 
                epochs=500, patience=50, batch=16, 
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"..\models")