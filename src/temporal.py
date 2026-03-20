# src/main.py
"""
Full Live ANPR Pipeline — Detection → Alignment → OCR → Validation → CSV Logging.

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
from .validate import is_valid_plate, normalise, fuzzy_match, clean_text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIRM_THRESHOLD = 5   # same plate text must appear N times before logging
CSV_PATH = Path("data/plates.csv")
SCREENSHOTS_DIR = Path("screenshots")
CAM_INDEX = 0
OCR_EVERY = 6           # slightly more frequent OCR for better accumulation


# ---------------------------------------------------------------------------
# Confirmation buffer with fuzzy matching
# ---------------------------------------------------------------------------
@dataclass
class ConfirmationBuffer:
    """
    Track observations over time, grouping near-identical readings together.

    The key change from the original: instead of requiring exact string
    equality, we use fuzzy_match() to accumulate frames where Tesseract
    reads 'RAB123C' and 'RAB123C' but also 'RAB1Z3C' (a single-char flip)
    toward the same plate.  The canonical text stored is the most common
    reading seen so far for that cluster.
    """
    threshold: int = CONFIRM_THRESHOLD
    # Maps canonical plate text → count of observations in its cluster
    _counts: Counter = field(default_factory=Counter)
    # Confirmed plates (never re-logged in the same session)
    _confirmed: set = field(default_factory=set)

    def _find_cluster(self, text: str) -> Optional[str]:
        """Return the canonical key for an existing cluster that matches text."""
        for key in self._counts:
            if fuzzy_match(text, key, max_distance=1):
                return key
        return None

    def observe(self, text: str) -> bool:
        """
        Record an observation. Returns True if this triggers first confirmation.
        """
        if not text or not is_valid_plate(text):
            return False

        canonical = clean_text(normalise(text))

        # Check if this reading is already confirmed
        if canonical in self._confirmed:
            return False

        # Merge into an existing near-match cluster, or start a new one
        cluster_key = self._find_cluster(canonical)
        if cluster_key is None:
            cluster_key = canonical

        self._counts[cluster_key] += 1

        if self._counts[cluster_key] >= self.threshold:
            self._confirmed.add(cluster_key)
            return True

        return False

    def reset(self):
        self._counts.clear()
        # _confirmed intentionally kept to avoid double-logging


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

        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)

            plate_img, _ = warp_plate(frame, contour)

            if plate_img is not None and plate_img.size:
                last_plate_img = plate_img.copy()

                if frame_idx % OCR_EVERY == 0:
                    text = read_plate(plate_img)
                    if text:
                        last_text = text
                        last_valid = is_valid_plate(text)

                        if buf.observe(text):
                            last_confirmed = clean_text(normalise(text))
                            log_plate(CSV_PATH, last_confirmed, last_valid)
                            ts = int(time.time())
                            cv2.imwrite(str(SCREENSHOTS_DIR / f"confirmed_{ts}.png"), vis)

        fps_n += 1
        frame_idx += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

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