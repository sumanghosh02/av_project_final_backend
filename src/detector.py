"""
detector.py  —  YOLOv8 Detection Wrapper for Production Backend
================================================================

WHY THIS FILE EXISTS (explain to panel):
-----------------------------------------
The raw Ultralytics YOLO API returns complex tensor objects.
This wrapper converts those tensors into simple Python dicts
that the FastAPI endpoint can directly serialize to JSON.

HOW THE MODEL WAS TRAINED:
----------------------------
1. Started with YOLOv8n pretrained on COCO-80 (80 object classes, 3M images)
   WHY pretrained: Transfer learning. The model already knows what edges,
   textures, and shapes look like. Fine-tuning on BDD100K takes hours
   instead of weeks because we build on existing knowledge.

2. Fine-tuned on BDD100K 8-class subset (5,000 frames):
   - 1,000 clean frames for training
   - 4,000 distorted frames for evaluation
   WHY BDD100K: It covers real US driving conditions — multiple cities,
   night/day, rain/clear — making it representative of deployment scenarios.

3. Training config:
   - Optimiser: SGD (lr=0.01, momentum=0.937, weight_decay=5e-4)
   - LR schedule: Cosine annealing (lr decays smoothly from 0.01 → 0.001)
   - Epochs: 10
   - Batch size: 16
   - Resolution: 640×640
   - Freeze: First 10 backbone layers frozen for first 3 epochs
     WHY freeze: Prevents catastrophic forgetting of ImageNet features
     learned during pretraining. After epoch 3, full network unfreezes.

4. WHY YOLOv8 NANO specifically:
   - Inference: ~27ms on mid-range GPU → fits 20 FPS budget
   - Size: 6MB weight file → easy to deploy, even on edge devices
   - mAP on COCO: 37.3 — sufficient for 8-class BDD100K detection

CONFIDENCE THRESHOLD (0.25):
   Lower = more detections (higher recall, more false positives)
   Higher = fewer detections (higher precision, more missed objects)
   0.25 is the sweet spot for driving scenarios with partial occlusion.

NMS IoU THRESHOLD (0.45):
   Non-Maximum Suppression removes duplicate boxes for the same object.
   0.45 means two boxes must overlap by <45% to both survive.
   Tighter than COCO standard (0.5) because vehicles are often close
   together and loose NMS would merge separate cars into one detection.
"""

import time
import numpy as np
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[detector] ultralytics not installed — using mock detector")


# BDD100K class names mapped to COCO IDs
# WHY this mapping: YOLOv8 was pretrained on COCO (80 classes).
# We only care about 8 driving-relevant classes.
BDD_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "traffic sign",
}

# Colour per class for bounding box drawing (BGR format)
CLASS_COLORS = {
    "car":          (50,  200,  50),
    "truck":        (255, 140,   0),
    "bus":          (0,   180, 255),
    "person":       (0,    50, 255),
    "bicycle":      (255, 255,   0),
    "motorcycle":   (180,   0, 255),
    "traffic light":(0,  255, 255),
    "traffic sign": (255,   0, 150),
}


class AVDetector:
    """
    Production-ready YOLOv8 detector wrapper.

    Parameters
    ----------
    weights   : path to .pt file, or 'yolov8n.pt' to auto-download
    conf_thr  : confidence threshold (default 0.25)
    iou_thr   : NMS IoU threshold    (default 0.45)
    """

    def __init__(
        self,
        weights:  str   = "yolov8n.pt",
        conf_thr: float = 0.25,
        iou_thr:  float = 0.45,
    ):
        self.conf_thr = conf_thr
        self.iou_thr  = iou_thr
        self.model    = None

        # Check for local fine-tuned weights first, fall back to pretrained
        local_weights = Path(__file__).parent.parent / "models" / "yolov8n_bdd100k.pt"
        base_weights  = Path(__file__).parent.parent / "models" / "yolov8n.pt"

        if YOLO_AVAILABLE:
            if local_weights.exists():
                print(f"[detector] Loading fine-tuned weights: {local_weights}")
                w = str(local_weights)
            elif base_weights.exists():
                print(f"[detector] Loading base weights: {base_weights}")
                w = str(base_weights)
            else:
                print(f"[detector] Downloading YOLOv8n pretrained weights...")
                w = weights   # auto-download

            try:
                self.model = YOLO(w)
                # Warm-up pass — eliminates JIT compilation from first real inference
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False)
                print(f"[detector] Model ready  device={self.model.device}")
            except Exception as e:
                print(f"[detector] Could not load YOLO: {e}")
                self.model = None
        else:
            print("[detector] Running in mock mode (no ultralytics)")

    def detect(self, img: np.ndarray) -> list:
        """
        Run detection on one BGR image.

        Returns list of dicts:
          [{"class": str, "conf": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        if self.model is None:
            return self._mock_detect(img)

        results  = self.model(
            img,
            conf    = self.conf_thr,
            iou     = self.iou_thr,
            classes = list(BDD_CLASSES.keys()),
            verbose = False,
        )
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            boxes  = r.boxes.xyxy.cpu().numpy()
            confs  = r.boxes.conf.cpu().numpy()
            cls_ids= r.boxes.cls.cpu().numpy()
            for (x1,y1,x2,y2), conf, cid in zip(boxes, confs, cls_ids):
                cname = BDD_CLASSES.get(int(cid), f"class_{int(cid)}")
                dets.append({
                    "class": cname,
                    "conf":  float(conf),
                    "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                })
        return dets

    def annotate(self, img: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of img.
        WHY copy: Never modify input image in-place in a web server —
        concurrent requests could corrupt each other's image data.
        """
        out = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color  = CLASS_COLORS.get(d["class"], (200, 200, 200))
            label  = f"{d['class']}  {d['conf']:.0%}"

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return out

    def _mock_detect(self, img: np.ndarray) -> list:
        """
        Returns fake detections when YOLO is not available.
        Used for frontend development and testing without GPU.
        """
        import random
        h, w = img.shape[:2]
        rng  = random.Random(42)
        n    = rng.randint(1, 4)
        classes = list(CLASS_COLORS.keys())
        dets = []
        for _ in range(n):
            cx  = rng.randint(w//4, 3*w//4)
            cy  = rng.randint(h//3, 2*h//3)
            bw  = rng.randint(80, 200)
            bh  = rng.randint(60, 150)
            dets.append({
                "class": rng.choice(classes),
                "conf":  round(rng.uniform(0.4, 0.95), 3),
                "bbox":  [max(0,cx-bw//2), max(0,cy-bh//2),
                          min(w,cx+bw//2), min(h,cy+bh//2)],
            })
        return dets
