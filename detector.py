import os
import sys
import torch
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV9_DIR = os.path.join(PROJECT_DIR, "yolov9")

if YOLOV9_DIR not in sys.path:
    sys.path.insert(0, YOLOV9_DIR)

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


class YOLOv9Detector:
    def __init__(
        self,
        fire_weights="best.pt",
        person_weights="yolov9/yolov9-c.pt",
        fire_conf_thres=0.25,
        person_conf_thres=0.35,
        iou_thres=0.45,
        img_size=640,
    ):
        # use GPU if available, else CPU
        device_str = "0" if torch.cuda.is_available() else "cpu"
        self.device = select_device(device_str)

        self.fire_conf_thres = float(fire_conf_thres)
        self.person_conf_thres = float(person_conf_thres)
        self.iou_thres = float(iou_thres)
        self.img_size = int(img_size)

        self.fire_model = attempt_load(fire_weights, device=self.device)
        self.fire_model.eval()
        self.fire_names = self.fire_model.names if hasattr(self.fire_model, "names") else {}

        self.person_model = attempt_load(person_weights, device=self.device)
        self.person_model.eval()
        self.person_names = self.person_model.names if hasattr(self.person_model, "names") else {}

        print("Using device:", self.device)
        print("Fire model names:", self.fire_names)
        print("Person model names:", self.person_names)

    def set_conf(self, conf):
        # optional runtime control, but keep sane values
        try:
            c = float(conf)
            self.fire_conf_thres = max(0.10, min(c, 0.60))
            self.person_conf_thres = max(0.25, min(c, 0.70))
        except Exception:
            pass

    def _preprocess(self, frame):
        img0 = frame.copy()
        img = letterbox(img0, self.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img0, img

    def _is_valid_fire_box(self, x1, y1, x2, y2, frame_shape):
        """
        Reject tiny noisy fire boxes that often create false positives.
        """
        h, w = frame_shape[:2]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh
        frame_area = w * h

        # reject extremely tiny boxes
        if bw < 12 or bh < 12:
            return False

        # reject extremely tiny area compared to full frame
        if area < max(150, int(frame_area * 0.00015)):
            return False

        return True

    def _run_model(self, model, names, frame, conf_thres, mode):
        img0, img = self._preprocess(frame)

        with torch.no_grad():
            pred = model(img)[0]

        pred = non_max_suppression(pred, conf_thres, self.iou_thres)

        detections = []

        for det in pred:
            if det is None or len(det) == 0:
                continue

            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                cls_id = int(cls.item()) if hasattr(cls, "item") else int(cls)

                if isinstance(names, dict):
                    label = names.get(cls_id, str(cls_id))
                elif isinstance(names, (list, tuple)) and cls_id < len(names):
                    label = names[cls_id]
                else:
                    label = str(cls_id)

                label_lower = str(label).lower().strip()

                if mode == "fire" and "fire" not in label_lower:
                    continue

                if mode == "person" and label_lower != "person":
                    continue

                x1, y1, x2, y2 = [
                    int(v.item()) if hasattr(v, "item") else int(v)
                    for v in xyxy
                ]
                conf_val = float(conf.item()) if hasattr(conf, "item") else float(conf)

                # extra filtering for fire to reduce false positives
                if mode == "fire":
                    if conf_val < self.fire_conf_thres:
                        continue
                    if not self._is_valid_fire_box(x1, y1, x2, y2, img0.shape):
                        continue

                # extra filtering for person
                if mode == "person":
                    if conf_val < self.person_conf_thres:
                        continue

                detections.append(([x1, y1, x2, y2], label_lower, conf_val))

        return detections

    def detect(self, frame):
        fire_dets = self._run_model(
            self.fire_model,
            self.fire_names,
            frame,
            self.fire_conf_thres,
            "fire",
        )

        person_dets = self._run_model(
            self.person_model,
            self.person_names,
            frame,
            self.person_conf_thres,
            "person",
        )

        print("Fire detections:", fire_dets)
        print("Person detections:", person_dets)

        return fire_dets + person_dets 
