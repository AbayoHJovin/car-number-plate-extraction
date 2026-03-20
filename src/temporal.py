# src/main.py
"""
Full Live ANPR Pipeline — Detection → Alignment → OCR → Validation → CSV Logging.
Mirrors the structure of recognize.py from Face_recognition_with_Arcface.

Run:
    python -m src.main
    OR
    python src/main.py

Keys:
    q  quit
    s  save current frame snapshot to screenshots/
    r  reset confirmation buffer
    d  toggle debug overlay
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detect import find_plate_contour, draw_detection
from .align import warp_plate
from .ocr import read_plate
from .validate import is_valid_plate, normalise


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIRM_THRESHOLD = 5   # same plate text must appear N times before logging
CSV_PATH = Path("data/plates.csv")
SCREENSHOTS_DIR = Path("screenshots")
CAM_INDEX = 0
OCR_EVERY = 8           # run OCR once every N frames (keeps CPU manageable)


# ---------------------------------------------------------------------------
# Temporal confirmation buffer
# (same concept as FaceDBMatcher — track observations over time)
# ---------------------------------------------------------------------------
@dataclass
class ConfirmationBuffer:
    threshold: int = CONFIRM_THRESHOLD
    _counts: Counter = field(default_factory=Counter)
    _confirmed: set = field(default_factory=set)

    def observe(self, text: str) -> bool:
        """Record an observation. Returns True if this observation triggers confirmation."""
        if not text or not is_valid_plate(text):
            return False
        if text in self._confirmed:
            return False          # already logged, skip
        self._counts[text] += 1
        if self._counts[text] >= self.threshold:
            self._confirmed.add(text)
            return True
        return False

    def reset(self):
        self._counts.clear()
        # NOTE: _confirmed is kept so we never re-log a plate in one session


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
def _init_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "plate", "valid"])
        print(f"[main] created CSV: {path}")


def log_plate(path: Path, plate: str, valid: bool):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, plate, valid])
    print(f"[main] LOGGED -> {plate}  valid={valid}  at {timestamp}")


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------
def _put(img, text, xy=(10, 30), scale=0.75, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(cam_index: int = CAM_INDEX):
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    _init_csv(CSV_PATH)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {cam_index} not available. Try index 0 or 1.")

    buf = ConfirmationBuffer(threshold=CONFIRM_THRESHOLD)

    last_plate_img = np.zeros((80, 300, 3), dtype=np.uint8)
    last_text = ""
    last_valid = False
    last_confirmed = ""

    fps_t0 = time.time()
    fps_n = 0
    fps: Optional[float] = None
    frame_idx = 0
    show_debug = False

    print(
        "[main] Full ANPR pipeline running.\n"
        "  q=quit  s=screenshot  r=reset buffer  d=debug overlay"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        contour = find_plate_contour(frame)

        # ---- Detection ----
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)

            # ---- Alignment ----
            plate_img, _ = warp_plate(frame, contour)

            if plate_img is not None and plate_img.size:
                last_plate_img = plate_img.copy()

                # ---- OCR (throttled) ----
                if frame_idx % OCR_EVERY == 0:
                    text = read_plate(plate_img)
                    if text:
                        last_text = text
                        last_valid = is_valid_plate(text)

                        # ---- Temporal Confirmation ----
                        if buf.observe(text):
                            last_confirmed = text
                            log_plate(CSV_PATH, text, last_valid)
                            # Save screenshot on first confirmation
                            ts = int(time.time())
                            cv2.imwrite(str(SCREENSHOTS_DIR / f"confirmed_{ts}.png"), vis)

        # ---- FPS ----
        fps_n += 1
        frame_idx += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

        # ---- HUD overlays ----
        plate_color = (0, 255, 0) if last_valid else (0, 0, 255)
        _put(vis, f"Plate: {last_text or '---'}", (10, 35), 0.85, plate_color, 2)
        _put(vis, "VALID" if last_valid else "INVALID", (10, 68), 0.7, plate_color, 2)

        if last_confirmed:
            _put(vis, f"CONFIRMED: {last_confirmed}", (10, 100), 0.8, (0, 255, 255), 2)

        fps_str = f"FPS: {fps:.1f}" if fps else "FPS: --"
        _put(vis, fps_str, (10, 130), 0.65, (255, 255, 0), 2)
        _put(vis, f"Buffer: {dict(buf._counts)}", (10, 160), 0.5, (200, 200, 200), 1)

        if show_debug and contour is not None:
            for pt in contour.reshape(-1, 2):
                cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)

        header = f"ANPR | thr={CONFIRM_THRESHOLD} | OCR@1/{OCR_EVERY}frames"
        _put(vis, header, (10, vis.shape[0] - 12), 0.55, (180, 180, 180), 1)

        cv2.imshow("ANPR - Full Pipeline", vis)
        cv2.imshow("ANPR - Aligned Plate", last_plate_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            p = SCREENSHOTS_DIR / f"snapshot_{ts}.png"
            cv2.imwrite(str(p), vis)
            print(f"[main] saved: {p}")
        elif key == ord("r"):
            buf.reset()
            print("[main] buffer reset")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[main] debug overlay: {'ON' if show_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[main] Done. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
