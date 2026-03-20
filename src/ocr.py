# src/ocr.py
"""
Step 3 — OCR character extraction using Tesseract.
Preprocesses the aligned plate image and returns the recognised text.

Run standalone:
    python -m src.ocr
Keys:
    q  quit
    s  save OCR snapshot to screenshots/ocr.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np
import pytesseract

from .detect import find_plate_contour
from .align import warp_plate
from .validate import normalise, is_valid_plate

# ---------------------------------------------------------------------------
# Tesseract configuration
# UPDATE THIS PATH IF TESSERACT IS ELSEWHERE ON YOUR SYSTEM
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ABAYO HIRWA JOVIN\Downloads\car-number-plate-extraction-main\car-number-plate-extraction-main\tesseract\tesseract\tesseract.exe"

# psm 8 = single word.  Also try psm 7 (single text line) if results are poor.
_TESS_CONFIG = (
    "--psm 8 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


# ---------------------------------------------------------------------------
# Core OCR helpers (importable)
# ---------------------------------------------------------------------------

def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Convert aligned plate to a clean binary image for OCR.

    Pipeline (order matters):
    1. Upscale FIRST — thresholding on a larger image is more accurate.
    2. CLAHE — normalise contrast across the plate (handles shadows, glare).
    3. Gaussian blur — suppress high-frequency noise before thresholding.
    4. Otsu threshold — automatic global threshold, works well after CLAHE.
    5. Light dilation — thicken thin strokes so Tesseract does not fragment
       characters.  A 2×2 kernel is enough; larger kernels merge adjacent chars.
    6. Invert if needed — Tesseract expects dark characters on a light background.

    The original code applied MORPH_OPEN (erosion → dilation) which was
    destroying thin character strokes (I, 1, thin verticals in R/A/B).
    """
    # --- Step 1: upscale ---
    scale = 3  # 3× gives ~900×240px for a 300×80 input — enough for Tesseract
    h, w = plate_bgr.shape[:2]
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # --- Step 2: CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Step 3: gentle blur ---
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- Step 4: Otsu binarisation ---
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 5: dilate to thicken strokes ---
    # 2×2 kernel adds ~1px to each character stroke without merging chars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # --- Step 6: ensure dark chars on white background ---
    # If the majority of pixels are dark, the plate is inverted — flip it.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def read_plate(plate_bgr: np.ndarray) -> str:
    """Run Tesseract on the aligned plate and return cleaned text."""
    binary = preprocess_plate(plate_bgr)
    raw = pytesseract.image_to_string(binary, config=_TESS_CONFIG)
    return normalise(raw)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _put_text(img: np.ndarray, text: str, xy=(10, 30), scale=0.8, thickness=2, color=(255, 255, 255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(cam_index: int = 0):
    save_dir = Path("screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {cam_index} not available. Try index 0 or 1.")

    blank_plate = np.zeros((80, 300, 3), dtype=np.uint8)
    last_plate_img = blank_plate.copy()
    last_text = ""
    last_valid = False

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    ocr_every = 8
    frame_idx = 0

    print("[ocr] OCR demo running. Press 'q' to quit, 's' to save.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contour = find_plate_contour(frame)
        vis = frame.copy()

        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
            plate, _ = warp_plate(frame, contour)
            if plate is not None and plate.size:
                last_plate_img = plate.copy()
                if frame_idx % ocr_every == 0:
                    last_text = read_plate(plate)
                    last_valid = is_valid_plate(last_text)

        frame_idx += 1

        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

        color = (0, 255, 0) if last_valid else (0, 0, 255)
        _put_text(vis, f"Plate: {last_text}", (10, 35), 0.85, 2, color)
        status = "VALID" if last_valid else "INVALID / NOT FOUND"
        _put_text(vis, status, (10, 68), 0.7, 2, color)
        _put_text(vis, f"FPS: {fps:.1f}", (10, 100), 0.65, 2, (255, 255, 0))

        cv2.imshow("ocr - Step 3 (camera)", vis)
        cv2.imshow("ocr - Step 3 (plate)", last_plate_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out_path = save_dir / "ocr.png"
            cv2.imwrite(str(out_path), vis)
            # Also save the preprocessed binary for debugging
            binary = preprocess_plate(last_plate_img)
            cv2.imwrite(str(save_dir / "ocr_binary.png"), binary)
            print(f"[ocr] saved: {out_path}  text={last_text!r}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()