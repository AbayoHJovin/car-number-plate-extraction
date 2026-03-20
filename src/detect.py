# src/detect.py
"""
Step 1 — Plate Detection.
Uses edge detection and contour analysis to locate license plates in a frame.
Modelled after the Face_recognition_with_Arcface detect.py style.

Run standalone:
    python -m src.detect
Keys:
    q  quit
    s  save current frame with detection to screenshots/detection.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core detection logic (importable)
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale, blur, and enhance edges for plate localisation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray


def find_plate_contour(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the best 4-point contour that likely contains a number plate,
    or None if nothing suitable is found.
    """
    gray = preprocess(frame)
    # Adaptive thresholding often works better than Canny for various lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edged = cv2.Canny(gray, 30, 200)
    
    # Combine Canny and Adaptive results
    edged = cv2.bitwise_or(edged, cv2.bitwise_not(binary))

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        # Try a few approximation strengths
        for eps in [0.01, 0.02, 0.03]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            
            # Usually 4, but hand-held paper sometimes has 5-6 due to fingers/folds
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / float(h)
                # Relaxed ratio: 1.2 to 9.0
                if 1.2 <= aspect <= 9.0 and w > 60:
                    # If we have 5-6 pts, just take the bounding box as the 4 corners
                    if len(approx) > 4:
                        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                    return approx
                
    return None


def draw_detection(frame: np.ndarray, contour: Optional[np.ndarray]) -> np.ndarray:
    """Draw the detected plate contour on a copy of the frame."""
    vis = frame.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            vis,
            "Plate Detected",
            (x, max(0, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            "Searching...",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main(cam_index: int = 0):
    save_dir = Path("screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {cam_index} not available. Try index 0 or 1.")

    print("[detect] Plate detection running. Press 'q' to quit, 's' to save.")

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contour = find_plate_contour(frame)
        vis = draw_detection(frame, contour)

        # FPS counter
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("detect - Step 1", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out_path = save_dir / "detection.png"
            cv2.imwrite(str(out_path), vis)
            print(f"[detect] saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
