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
from typing import Optional, Tuple
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
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"

# Tesseract page-segmentation mode 8 = single word, good for plates
_TESS_CONFIG = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "


# ---------------------------------------------------------------------------
# Core OCR helpers (importable)
# ---------------------------------------------------------------------------

def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Convert aligned plate to a clean binary image for OCR.
    Enhanced to remove graph paper grid lines using morphological operations.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    
    # Upscale
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    
    # Adaptive thresholding to handle uneven lighting on paper
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Morphological Opening to remove thin lines (the graph grid)
    # A 3x3 kernel will remove lines thinner than characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Invert so characters are black on white (if needed by Tesseract, though --psm handles it)
    # But usually Tesseract prefers black on white. 
    # Adaptive Threshold above gives white on black if not careful.
    # Let's ensure black on white.
    if np.mean(opening) < 127:
        opening = cv2.bitwise_not(opening)
        
    return opening


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
    # Throttle OCR to once every N frames to keep UI responsive
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

        # FPS
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
            print(f"[ocr] saved: {out_path}  text={last_text!r}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
