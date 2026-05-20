"""
disturbance.py  —  Physics-Based Camera Degradation (Backend)
==============================================================
Four functions, each modelling a real degradation type.
Imported by main.py and exposed via the /api/detect endpoint.
"""

import cv2
import numpy as np


def apply_fog(img: np.ndarray, severity: float = 0.6) -> np.ndarray:
    """Atmospheric scattering: I_fog = I*t + A*(1-t)"""
    img_f = img.astype(np.float32) / 255.0
    h, w  = img_f.shape[:2]
    depth = np.tile(np.linspace(0.2, 1.0, h).reshape(h, 1), (1, w))
    depth = cv2.GaussianBlur(depth.astype(np.float32), (51, 51), 0)
    A = 0.9
    t = np.exp(-severity * depth * 3.0)
    t = np.clip(t, 0.05, 1.0)
    t3 = np.stack([t, t, t], axis=2)
    out = img_f * t3 + A * (1.0 - t3)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def apply_rain(img: np.ndarray, num_drops: int = 800) -> np.ndarray:
    """Directional streak overlay at 70-80 degrees."""
    rain = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape[:2]
    rng  = np.random.default_rng(seed=42)
    xs   = rng.integers(0, w, num_drops)
    ys   = rng.integers(0, h, num_drops)
    angs = rng.uniform(70, 80, num_drops)
    for x, y, ang in zip(xs, ys, angs):
        rad = np.deg2rad(ang)
        dx  = int(20 * np.cos(rad))
        dy  = int(20 * np.sin(rad))
        cv2.line(rain, (int(x), int(y)), (int(x+dx), int(y+dy)),
                 (200, 200, 200), 1, lineType=cv2.LINE_AA)
    rain = cv2.GaussianBlur(rain, (3, 3), 0)
    out  = cv2.addWeighted(img.astype(np.float32), 1.0, rain, 0.5, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_lowlight(img: np.ndarray, scale: float = 0.35, noise_std: float = 15.0) -> np.ndarray:
    """Scale brightness + additive Gaussian noise (EMVA1288 model)."""
    dark  = img.astype(np.float32) * scale
    noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """1-D horizontal motion blur kernel."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, k)
