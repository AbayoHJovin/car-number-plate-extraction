# src/align.py
"""
Step 2 — Plate Alignment (Perspective Rectification).
Warps the detected quadrilateral plate region into a flat, standard-size image
ready for OCR.  Mirrors the structure of align.py from Face_recognition_with_Arcface.

Run standalone:
    python -m src.align
Keys:
    q  quit
    s  save current aligned plate to screenshots/alignment.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import time

import cv2
import numpy as np

from .detect import find_plate_contour


# ---------------------------------------------------------------------------
# Core alignment helpers (importable)
# ---------------------------------------------------------------------------

_PLATE_W = 300
_PLATE_H = 80


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Return corners in order: top-left, top-right, bottom-right, bottom-left.
    Works for any convex quadrilateral.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest x-y
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest x-y

    return rect


def warp_plate(
    frame: np.ndarray,
    contour: np.ndarray,
    out_size: Tuple[int, int] = (_PLATE_W, _PLATE_H),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply a four-point perspective transform to extract and rectify the plate.

    Returns:
        (warped_plate, transform_matrix)  or  (None, None) if contour invalid.
    """
    if contour is None:
        return None, None

    rect = _order_corners(contour)
    dst = np.array(
        [[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1], [0, out_size[1] - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, out_size)
    return warped, M


def _put_text(img: np.ndarray, text: str, xy=(10, 30), scale=0.8, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _safe_imshow(win: str, img: Optional[np.ndarray]):
    if img is not None and img.size > 0:
        cv2.imshow(win, img)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main(cam_index: int = 0):
    save_dir = Path("screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {cam_index} not available. Try index 0 or 1.")

    blank = np.zeros((_PLATE_H, _PLATE_W, 3), dtype=np.uint8)
    last_aligned: np.ndarray = blank.copy()

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print("[align] Plate alignment running. Press 'q' to quit, 's' to save.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contour = find_plate_contour(frame)
        vis = frame.copy()
        aligned = None

        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
            aligned, _ = warp_plate(frame, contour)
            if aligned is not None and aligned.size:
                last_aligned = aligned
            _put_text(vis, "Plate found — aligning", (10, 30), 0.75, 2)
        else:
            _put_text(vis, "Searching...", (10, 30), 0.75, 2)

        # FPS
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

        _put_text(vis, f"FPS: {fps:.1f}", (10, 58), 0.7, 2)
        _put_text(vis, f"warp -> {_PLATE_W}x{_PLATE_H}", (10, 86), 0.7, 2)

        _safe_imshow("align - Step 2 (camera)", vis)
        _safe_imshow("align - Step 2 (aligned)", last_aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out_path = save_dir / "alignment.png"
            cv2.imwrite(str(out_path), last_aligned)
            print(f"[align] saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
