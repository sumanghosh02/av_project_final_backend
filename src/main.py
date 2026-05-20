"""
main.py  —  FastAPI Backend for AV Perception System
=====================================================

WHY FastAPI?
------------
FastAPI is chosen over Flask because it:
  1. Provides automatic API documentation at /docs (shows panel exactly what each endpoint does)
  2. Is 3x faster than Flask for image-heavy endpoints
  3. Supports async I/O — multiple users can upload images simultaneously
  4. Has built-in request validation — wrong inputs are caught before reaching the model

HOW THE API WORKS (explain to panel):
  POST /api/detect     → upload image → get detections + annotated image back
  POST /api/video      → upload video → get processed video back
  GET  /api/health     → check if server is running
  GET  /api/stats      → get total detections processed so far
  GET  /api/conditions → list all supported disturbance conditions

DEPLOYMENT:
  Local:  uvicorn main:app --host 0.0.0.0 --port 8000
  Cloud:  Same command, but on a server (Render / Railway / AWS EC2)
"""

import io
import time
import base64
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import our own modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from detector  import AVDetector
from preprocess import AdaptivePreprocessor
from disturbance import apply_fog, apply_rain, apply_lowlight, apply_blur

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AV Perception System API",
    description="""
    Adaptive Preprocessing for Robust Autonomous Vehicle Perception.
    Upload an image → apply optional disturbance → preprocess → detect objects.

    **Developed by:** Suman Ghosh | Roll No. 123241330036 | MCA 2024-2026
    **JIS College of Engineering, Kalyani**
    """,
    version="1.0.0",
)

# CORS — allows the React frontend (running on port 3000) to call this backend (port 8000)
# WHY we need CORS: browsers block requests between different ports by default for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace * with your actual domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global objects (loaded once at startup, reused for every request) ────────
# WHY load once: loading YOLOv8 takes ~2 seconds. Reloading it per request would
# make every detection take 2+ seconds. Loading once at startup means detections
# run at ~30ms after the initial warmup.
detector     = None
preprocessor = None
stats        = {"total_requests": 0, "total_detections": 0, "uptime_start": time.time()}

DISTURBANCE_FNS = {
    "fog":      apply_fog,
    "rain":     apply_rain,
    "lowlight": apply_lowlight,
    "blur":     apply_blur,
}

# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global detector, preprocessor
    print("[startup] Loading YOLOv8 model...")
    detector     = AVDetector()
    preprocessor = AdaptivePreprocessor()
    print("[startup] Ready.")

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Check if the server and model are running. Used by the frontend to show status."""
    return {
        "status":    "ok",
        "model":     "YOLOv8n",
        "uptime_s":  round(time.time() - stats["uptime_start"], 1),
        "total_req": stats["total_requests"],
    }

# ─── Stats ────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    """Returns cumulative detection statistics for the dashboard counter."""
    return {
        **stats,
        "uptime_minutes": round((time.time() - stats["uptime_start"]) / 60, 1),
    }

# ─── Supported conditions ─────────────────────────────────────────────────────
@app.get("/api/conditions")
async def get_conditions():
    """List all supported disturbance types that can be applied to an image."""
    return {
        "conditions": [
            {"id": "none",      "label": "Clean (No Disturbance)",  "icon": "☀️"},
            {"id": "fog",       "label": "Fog",                     "icon": "🌫️"},
            {"id": "rain",      "label": "Rain",                    "icon": "🌧️"},
            {"id": "lowlight",  "label": "Low Light",               "icon": "🌙"},
            {"id": "blur",      "label": "Motion Blur",             "icon": "💨"},
        ]
    }

# ─── Main detection endpoint ──────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(
    file:        UploadFile = File(..., description="Image file (JPG, PNG)"),
    disturbance: str        = Form("none", description="Disturbance type: none/fog/rain/lowlight/blur"),
    preprocess:  bool       = Form(True,   description="Whether to apply preprocessing pipeline"),
):
    """
    Main endpoint — takes an image, optionally degrades it, optionally preprocesses it,
    runs YOLOv8 detection, and returns both the annotated image AND raw detection data.

    HOW IT WORKS (for panel explanation):
      1. Frontend sends image as multipart form data
      2. We decode bytes → OpenCV BGR numpy array
      3. Apply disturbance (if selected) — simulates adverse weather
      4. Run baseline detection (no preprocessing) for comparison
      5. Apply preprocessing (if enabled) — our 5-stage pipeline
      6. Run proposed detection on preprocessed image
      7. Encode annotated images to base64 strings
      8. Return both annotated images + all detection data + timing as JSON
    """
    global stats
    t_total = time.perf_counter()

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, detail=f"Expected JPEG or PNG, got {file.content_type}")

    # Decode image
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="Could not decode image. Please upload a valid JPEG or PNG.")

    # Resize large images to max 1280px to keep latency reasonable
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)))

    result = {}

    # Step 1: Apply disturbance
    t0 = time.perf_counter()
    if disturbance != "none" and disturbance in DISTURBANCE_FNS:
        distorted = DISTURBANCE_FNS[disturbance](img.copy())
    else:
        distorted = img.copy()
    result["disturbance_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 2: Baseline detection (no preprocessing)
    t0 = time.perf_counter()
    baseline_dets    = detector.detect(distorted)
    result["baseline_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    baseline_ann     = detector.annotate(distorted, baseline_dets)

    # Step 3: Preprocessing
    t0 = time.perf_counter()
    if preprocess:
        clean_img = preprocessor.run(distorted.copy())
    else:
        clean_img = distorted.copy()
    result["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 4: Proposed detection (with preprocessing)
    t0 = time.perf_counter()
    proposed_dets    = detector.detect(clean_img)
    result["proposed_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    proposed_ann     = detector.annotate(clean_img, proposed_dets)

    # Step 5: Encode annotated images to base64
    def to_b64(frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode()

    result["original_b64"]  = to_b64(img)
    result["distorted_b64"] = to_b64(distorted)
    result["baseline_b64"]  = to_b64(baseline_ann)
    result["proposed_b64"]  = to_b64(proposed_ann)

    # Step 6: Detection data
    result["baseline_detections"] = [
        {"class": d["class"], "confidence": round(d["conf"], 3),
         "bbox": d["bbox"]}
        for d in baseline_dets
    ]
    result["proposed_detections"] = [
        {"class": d["class"], "confidence": round(d["conf"], 3),
         "bbox": d["bbox"]}
        for d in proposed_dets
    ]

    result["baseline_count"]  = len(baseline_dets)
    result["proposed_count"]  = len(proposed_dets)
    result["detection_gain"]  = len(proposed_dets) - len(baseline_dets)
    result["disturbance"]     = disturbance
    result["preprocessing"]   = preprocess
    result["total_ms"]        = round((time.perf_counter() - t_total) * 1000, 1)
    result["image_size"]      = {"width": img.shape[1], "height": img.shape[0]}

    # Update global stats
    stats["total_requests"]  += 1
    stats["total_detections"] += len(proposed_dets)

    return JSONResponse(content=result)


# ─── Batch endpoint — multiple images at once ─────────────────────────────────
@app.post("/api/batch")
async def batch_detect(
    files:       list[UploadFile] = File(...),
    disturbance: str              = Form("fog"),
):
    """
    Process multiple images at once and return aggregate metrics.
    Used to reproduce the paper's Table 9.1 results live in the demo.
    """
    if len(files) > 10:
        raise HTTPException(400, detail="Maximum 10 images per batch.")

    results = []
    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        dist   = DISTURBANCE_FNS.get(disturbance, lambda x: x)(img.copy())
        base   = detector.detect(dist)

        t0     = time.perf_counter()
        prep   = preprocessor.run(dist.copy())
        pre_ms = (time.perf_counter() - t0) * 1000

        prop   = detector.detect(prep)

        results.append({
            "filename":       f.filename,
            "baseline_count": len(base),
            "proposed_count": len(prop),
            "gain":           len(prop) - len(base),
            "preprocess_ms":  round(pre_ms, 1),
        })

    avg_gain = sum(r["gain"] for r in results) / max(len(results), 1)
    return {
        "files_processed": len(results),
        "disturbance":     disturbance,
        "results":         results,
        "avg_detection_gain": round(avg_gain, 2),
    }


# ─── Entry point (for local development) ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
