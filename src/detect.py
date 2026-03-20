# src/detect.py
"""
Step 1 — Plate Detection.
Uses edge detection and contour analysis to locate license plates in a frame.

Run standalone:
    python -m src.detect
Keys:
    q  quit
    s  save current frame with detection to screenshots/detection.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core detection logic (importable)
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply CLAHE for contrast normalisation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE handles uneven lighting far better than a flat bilateral filter
    # for plate detection — it boosts local contrast where chars meet background.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Mild blur to suppress noise without killing edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def find_plate_contour(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the best 4-point contour that likely contains a number plate,
    or None if nothing suitable is found.

    Key changes vs original:
    - Removed the bitwise_or with inverted adaptive threshold — that was
      flooding the edge map and returning spurious large contours.
    - Only accept exactly 4-point approximations; 5-6 point contours are
      collapsed via convex hull (not bounding box) so tilt is preserved.
    - Added minimum contour area filter to reject small noise regions.
    - Tightened aspect ratio to 2.0–7.0 (Rwandan plates are ~4.5:1).
    """
    gray = preprocess(frame)
    edged = cv2.Canny(gray, 30, 200)

    # Morphological closing to bridge small gaps in plate edges only
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area descending; top 30 is enough and avoids noise contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    frame_area = frame.shape[0] * frame.shape[1]

    for c in cnts:
        area = cv2.contourArea(c)
        # Reject contours that are too small (noise) or suspiciously large
        # (frame border, hand, background objects)
        if area < 1500 or area > frame_area * 0.25:
            continue

        peri = cv2.arcLength(c, True)

        # Try a conservative epsilon first, then relax slightly
        for eps in [0.015, 0.02, 0.025, 0.03]:
            approx = cv2.approxPolyDP(c, eps * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / float(h)
                # Rwandan plates: ~520mm × 110mm ≈ 4.7:1
                # Allow 2.0–7.0 to handle mild perspective distortion
                if 2.0 <= aspect <= 7.0 and w > 80:
                    return approx

            elif 5 <= len(approx) <= 8:
                # Use convex hull to reduce to a clean quadrilateral —
                # this preserves the actual corner positions unlike bbox.
                hull = cv2.convexHull(approx)
                hull_peri = cv2.arcLength(hull, True)
                quad = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
                if len(quad) == 4:
                    x, y, w, h = cv2.boundingRect(quad)
                    aspect = w / float(h)
                    if 2.0 <= aspect <= 7.0 and w > 80:
                        return quad

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