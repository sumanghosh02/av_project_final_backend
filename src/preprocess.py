"""
preprocess.py  —  5-Stage Adaptive Preprocessing Pipeline (Backend)
====================================================================

Identical logic to the research version but packaged for production use
inside the FastAPI backend. Imported by main.py.

PIPELINE ORDER (memorise this for panel):
  Step 1: CLAHE         → contrast recovery (fog, low light)
  Step 2: Denoise       → noise removal (low light, sensor noise)
  Step 3: Sharpen       → edge recovery (fog, blur)
  Step 4: Gamma         → brightness lift (low light) — ADAPTIVE
  Step 5: Laplacian     → boundary enhancement (all conditions)

WHY this order (key point for panel):
  - CLAHE first: improves signal quality before denoising
  - Denoise before sharpen: sharpening amplifies noise, so clean first
  - Gamma after sharpen: no point brightening before edges are sharp
  - Laplacian last: enhances clean edges, not raw distorted ones
"""

import cv2
import numpy as np


class AdaptivePreprocessor:
    """
    Chains 5 image restoration steps in sequence.
    Brightness correction is conditional — activates only on dark frames.

    Parameters (all match the thesis paper values):
      clahe_clip    : CLAHE clip limit (2.0)
      clahe_tile    : tile grid size for CLAHE (8,8)
      denoise_k     : Gaussian blur kernel size (5)
      denoise_sigma : Gaussian sigma (1.0)
      sharp_alpha   : unsharp mask strength (1.5)
      bright_thresh : mean pixel threshold to trigger gamma (80)
      bright_gamma  : gamma exponent when triggered (0.6)
      edge_weight   : Laplacian blend weight (0.3)
    """

    def __init__(self, cfg: dict = None):
        c = cfg or {}
        self.clahe_clip    = c.get("clahe_clip",    2.0)
        self.clahe_tile    = c.get("clahe_tile",    (8, 8))
        self.denoise_k     = c.get("denoise_k",     5)
        self.denoise_sigma = c.get("denoise_sigma",  1.0)
        self.sharp_alpha   = c.get("sharp_alpha",    1.5)
        self.bright_thresh = c.get("bright_thresh",  80.0)
        self.bright_gamma  = c.get("bright_gamma",   0.6)
        self.edge_weight   = c.get("edge_weight",    0.3)

        # Pre-build CLAHE object (reused across requests for speed)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_tile
        )
        # Pre-build gamma LUT (256-entry lookup table, O(1) per pixel)
        self._lut = np.array([
            min(255, int(255 * (i / 255.0) ** self.bright_gamma))
            for i in range(256)
        ], dtype=np.uint8)

    # ── Step 1: CLAHE ────────────────────────────────────────────────────────
    def _clahe_step(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of LAB colour space.

        WHY LAB not BGR:
          In BGR, contrast enhancement affects all three channels independently,
          causing colour shifts (skin tones turn green, sky turns yellow).
          LAB separates Luminance (L) from colour (A=green-red, B=blue-yellow).
          Enhancing only L recovers contrast without touching colour at all.

        WHY clip limit 2.0:
          Without clipping, CLAHE amplifies noise in flat regions
          (like a plain grey road surface). Clip limit 2.0 limits the
          histogram redistribution, preventing noise amplification while
          still recovering meaningful contrast.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # ── Step 2: Gaussian Denoising ───────────────────────────────────────────
    def _denoise_step(self, img: np.ndarray) -> np.ndarray:
        """
        Apply 5×5 Gaussian blur to suppress high-frequency noise.

        WHY Gaussian not median / bilateral:
          Median: good for salt-and-pepper noise, slow for CMOS read noise.
          Bilateral: best quality but 10-15x slower (breaks real-time budget).
          Gaussian 5×5: separable convolution, ~1ms, handles CMOS noise well.

        WHY sigma 1.0:
          Conservative — suppresses noise but preserves structures ≥ 3px.
          Object boundaries (cars, pedestrians) are typically 5-30px wide,
          so sigma 1.0 smooths noise without blurring what matters.
        """
        return cv2.GaussianBlur(img, (self.denoise_k, self.denoise_k), self.denoise_sigma)

    # ── Step 3: Unsharp Masking ──────────────────────────────────────────────
    def _sharpen_step(self, img: np.ndarray) -> np.ndarray:
        """
        Unsharp mask: Enhanced = Original + α × (Original - Blurred)

        WHY this works:
          (Original - Blurred) isolates high-frequency content (edges, detail).
          Adding it back amplifies those frequencies.
          α=1.5 gives visible recovery without creating halo artefacts.

        WHY α=1.5 not higher:
          Above α=2.5, edges develop bright halos that confuse the detector.
          The paper validated 1.5 gives best F1 gain on the BDD100K val split.
        """
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        img_f   = img.astype(np.float32)
        blur_f  = blurred.astype(np.float32)
        return np.clip(img_f + self.sharp_alpha * (img_f - blur_f), 0, 255).astype(np.uint8)

    # ── Step 4: Adaptive Gamma Correction ────────────────────────────────────
    def _gamma_step(self, img: np.ndarray) -> np.ndarray:
        """
        Conditional gamma correction: activates only when mean < threshold.

        WHY conditional (not always applied):
          A brightness boost on an already well-exposed image:
          - Washes out highlights (sky, headlights)
          - Reduces local contrast that CLAHE just recovered
          - Can push pixel values to 255 (clipping) losing detail
          By gating on mean intensity < 80, we lift only genuinely dark frames.

        WHY gamma 0.6 (not 0.5 or 0.7):
          gamma < 1 lifts shadows, compresses highlights.
          0.6 lifts a pixel value of 50 (dark) to ~102 (visible) while
          lifting 200 (bright) only to ~215 — gentle on highlights.
          Pre-computed as a 256-entry LUT for O(1) speed (microseconds).
        """
        if img.mean() < self.bright_thresh:
            return cv2.LUT(img, self._lut)
        return img   # no change needed

    # ── Step 5: Laplacian Edge Enhancement ───────────────────────────────────
    def _edge_step(self, img: np.ndarray) -> np.ndarray:
        """
        Blend Laplacian edge map back at weight 0.3.

        WHY Laplacian:
          The Laplacian operator computes the second spatial derivative.
          It is large at edges (rapid intensity change) and near-zero in
          flat regions. Adding it back selectively sharpens boundaries
          without brightening uniform areas like road surfaces.

        WHY last step:
          Running edge enhancement on a noisy, blurry frame would amplify
          noise edges. After CLAHE+denoise+sharpen, we enhance clean,
          real object boundaries — not noise artefacts.

        WHY weight 0.3 not higher:
          At 0.5 thin objects (distant pedestrians) develop halos.
          0.3 is the largest weight that avoids artefacts on val split.
        """
        img_f = img.astype(np.float32)
        lap   = cv2.Laplacian(img, cv2.CV_32F)
        return np.clip(img_f + self.edge_weight * lap, 0, 255).astype(np.uint8)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, img: np.ndarray) -> np.ndarray:
        """Execute all 5 steps in order. Input and output: uint8 BGR."""
        img = self._clahe_step(img)
        img = self._denoise_step(img)
        img = self._sharpen_step(img)
        img = self._gamma_step(img)
        img = self._edge_step(img)
        return img

    def run_with_stages(self, img: np.ndarray) -> dict:
        """Return intermediate output after each step (for visualisation)."""
        stages = {"original": img.copy()}
        img = self._clahe_step(img);   stages["clahe"]      = img.copy()
        img = self._denoise_step(img); stages["denoised"]   = img.copy()
        img = self._sharpen_step(img); stages["sharpened"]  = img.copy()
        img = self._gamma_step(img);   stages["brightened"] = img.copy()
        img = self._edge_step(img);    stages["enhanced"]   = img.copy()
        return stages
