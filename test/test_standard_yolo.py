from ultralytics import YOLO
import config.config as config
import json

if __name__ == "__main__":
    model = YOLO(config.MODEL_PATH_NCNN)

    metrics = model.val(data=config.DATA_PATH, imgsz = 640, batch=1, device="cpu")

    # Convert metrics to dict and save
    metrics_dict = metrics.results_dict
    with open(f"results/{config.TEST_SAVE_NAME}", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics saved to {config.TEST_SAVE_NAME}")