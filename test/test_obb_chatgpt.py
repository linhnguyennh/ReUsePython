import ncnn
import cv2
import numpy as np
import math

INPUT_SIZE = 640

class YOLO11_OBB_NCNN:
    def __init__(self, param_path, bin_path, use_vulkan=True):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        mat_in = ncnn.Mat(padded.astype(np.float32))
        mat_in.substract_mean_normalize([0, 0, 0], [1/255., 1/255., 1/255.])
        return mat_in, scale

    def infer(self, img):
        mat_in, scale = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError("NCNN inference failed.")

        # Convert to numpy
        out = np.array(mat_out, copy=True)
        if out.size == 0:
            return []

        detections = []
        for row in out:
            if len(row) < 6:
                continue
            cx, cy, w, h, angle, conf = row[:6]
            cls_scores = row[6:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            final_conf = conf * cls_conf
            if final_conf < 0.25:
                continue
            # Rescale
            cx /= scale
            cy /= scale
            w /= scale
            h /= scale
            detections.append({
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "angle": angle,
                "confidence": final_conf,
                "class_id": cls_id
            })
        return detections

def draw_obb(img, dets, class_names):
    for d in dets:
        cx, cy = d["cx"], d["cy"]
        w, h = d["w"], d["h"]
        angle_deg = math.degrees(d["angle"])
        if w < h:
            angle_deg += 90
        rrect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rrect).astype(np.int32)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        label = f"{class_names[d['class_id']]} {d['confidence']:.2f}"
        tl = box[np.argmin(box.sum(axis=1))]
        cv2.putText(img, label, (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img

if __name__ == "__main__":
    param = r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203_ncnn_model\model.ncnn.param"
    binf = r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203_ncnn_model\model.ncnn.bin"

    model = YOLO11_OBB_NCNN(param, binf)
    class_names = {0: "battery_housing", 1: "terminal"}

    img = cv2.imread(r"data\annotated\Focus1-BIGMAP.v2-morrow_251020.yolov11\train\images\morrow00005_png.rf.f84bc1a4f9ce6791d8d33dc42b243199.jpg")
    dets = model.infer(img)
    print("Detections:", dets)
    img = draw_obb(img, dets, class_names)
    cv2.imshow("OBB Detection", img)
    cv2.waitKey(0)
