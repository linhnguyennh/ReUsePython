from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

if __name__ == '__main__':
    model.train(data=r"data\morrow_overall_260421.yolov11\data.yaml", 
                epochs=300, patience=30, batch=32, workers=8,
                cache= True,
                device=0, #optimizer='AdamW'
                seed=42, 
                lr0 = 0.002,
                cos_lr=True, imgsz = 640,
                project=r"models\focus1\260421")