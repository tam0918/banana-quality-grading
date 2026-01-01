from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Optional, Union

import os

import cv2
import numpy as np

from .grader import BananaGrader, FrameGrade, GradeResult


@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    grade: FrameGrade
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
        self._capture_thread: Optional[threading.Thread] = None
        self._grade_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Keep grading work off the capture loop.
        self._grade_queue: "Queue[np.ndarray]" = Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._latest_grade: GradeResult = GradeResult(
            category_key="none",
            label_vi="—",
            label_en="—",
            status_vi="—",
            confidence=0.0,
            bbox_xyxy=None,
            debug={},
        )
        self._latest_frame_grade: FrameGrade = FrameGrade(overall=self._latest_grade, items=[])
        self._latest_fps: float = 0.0

        # Optional cap to keep CPU usage under control.
        # When set (e.g., 10-15), grading runs at most that rate while UI can still update at 30fps.
        try:
            self._max_infer_fps = float(os.environ.get("BANANA_MAX_INFER_FPS", "0") or 0)
        except Exception:
            self._max_infer_fps = 0.0

    def start(self) -> None:
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._run_capture, daemon=True)
        self._grade_thread = threading.Thread(target=self._run_grade, daemon=True)
        self._capture_thread.start()
        self._grade_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=1.5)
        if self._grade_thread:
            self._grade_thread.join(timeout=1.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def is_running(self) -> bool:
        return bool(self._capture_thread and self._capture_thread.is_alive() and not self._stop.is_set())

    def set_source(self, source: Union[int, str]) -> None:
        self._source = source

    def _open_capture(self) -> cv2.VideoCapture:
        # On Windows, DirectShow often gives lower latency for webcams.
        try:
            if isinstance(self._source, int) and os.name == "nt":
                return cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        except Exception:
            pass
        return cv2.VideoCapture(self._source)

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        # Prefer lower latency over max resolution.
        # (You can tune these later if you want higher quality.)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Try a moderate resolution to keep inference fast.
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        except Exception:
            pass

        # Ask for smoother capture; may be ignored by backend.
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        # MJPG can reduce CPU & latency on some webcams.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    def _run_capture(self) -> None:
        self._cap = self._open_capture()
        self._configure_capture(self._cap)

        last_t = time.time()
        fps = 0.0
        last_ui_emit = 0.0
        min_ui_interval_s = 1.0 / 30.0  # cap UI updates to ~30fps

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            with self._state_lock:
                self._latest_fps = fps
                frame_grade = self._latest_frame_grade

            # Feed grading queue with newest frame (drop old).
            try:
                self._grade_queue.put_nowait(frame)
            except Exception:
                try:
                    _ = self._grade_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._grade_queue.put_nowait(frame)
                except Exception:
                    pass

            # Emit to UI at a bounded rate to reduce CPU spikes from image conversions.
            if self._on_frame is not None and (now - last_ui_emit) >= min_ui_interval_s:
                last_ui_emit = now
                self._on_frame(FramePacket(frame_bgr=frame, grade=frame_grade, fps=fps))

        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _run_grade(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._grade_queue.get(timeout=0.1)
            except Empty:
                continue
            except Exception:
                continue

            try:
                # Grade on a copy so UI overlays can't affect the analysis.
                t0 = time.time()
                frame_grade = self._grader.grade_frame(frame.copy())
            except Exception:
                # Keep the last grade if inference fails.
                continue

            with self._state_lock:
                self._latest_frame_grade = frame_grade

            # Throttle if requested.
            if self._max_infer_fps and self._max_infer_fps > 0:
                dt = time.time() - t0
                target = 1.0 / max(1.0, float(self._max_infer_fps))
                if dt < target:
                    time.sleep(max(0.0, target - dt))
