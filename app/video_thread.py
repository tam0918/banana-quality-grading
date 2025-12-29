from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import os

import cv2
import numpy as np

from .grader import BananaGrader, GradeResult


@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    grade: GradeResult
    fps: float


class VideoThread:
    """Background capture + grading loop to keep UI responsive."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        grader: Optional[BananaGrader] = None,
        on_frame: Optional[Callable[[FramePacket], None]] = None,
    ):
        self._source = source
        data_yaml_path = "datasets/data.yaml" if os.path.exists("datasets/data.yaml") else None
        self._grader = grader or BananaGrader(model_path="weights/best.pt", data_yaml_path=data_yaml_path)
        self._on_frame = on_frame

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def set_source(self, source: Union[int, str]) -> None:
        self._source = source

    def _run(self) -> None:
        self._cap = cv2.VideoCapture(self._source)
        # Try a decent default size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        last_t = time.time()
        fps = 0.0

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            # FPS estimate
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            grade = self._grader.grade(frame)

            if self._on_frame is not None:
                self._on_frame(FramePacket(frame_bgr=frame, grade=grade, fps=fps))

        # Cleanup
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None
