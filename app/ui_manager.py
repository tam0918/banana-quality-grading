from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
import os

from .grader import BananaGrader, GradeResult
from .text_overlay import FontConfig, UnicodeTextRenderer
from .video_thread import FramePacket, VideoThread


@dataclass(frozen=True)
class UIConfig:
    font_path: str
    model_path: str
    detector_model_path: str = "yolov8n.pt"
    camera_index: int = 0


class UI_Manager:
    """Modern dark-mode dashboard UI (Vietnamese) built with CustomTkinter."""

    def __init__(self, config: UIConfig):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._config = config
        self._root = ctk.CTk()
        self._root.title("Hệ Thống Kiểm Tra Chất Lượng Chuối")
        self._root.geometry("1200x720")
        self._root.minsize(1100, 650)

        data_yaml_path = "datasets/data.yaml" if os.path.exists("datasets/data.yaml") else None
        self._grader = BananaGrader(
            model_path=config.model_path,
            detector_model_path=config.detector_model_path,
            data_yaml_path=data_yaml_path,
        )
        self._text = UnicodeTextRenderer(FontConfig(font_path=config.font_path, font_size=22))

        self._packet_lock = threading.Lock()
        self._latest_packet: Optional[FramePacket] = None

        self._video_thread = VideoThread(
            source=config.camera_index,
            grader=self._grader,
            on_frame=self._on_frame_from_thread,
        )

        self._ctk_image = None

        self._build_layout()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # UI refresh loop
        self._root.after(20, self._ui_tick)

    # ---------- Public ----------

    def run(self) -> None:
        self._root.mainloop()

    # ---------- Layout ----------

    def _build_layout(self) -> None:
        self._root.grid_columnconfigure(0, weight=4)
        self._root.grid_columnconfigure(1, weight=2)
        self._root.grid_rowconfigure(0, weight=1)

        self._left = ctk.CTkFrame(self._root, corner_radius=16)
        self._left.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self._left.grid_rowconfigure(0, weight=1)
        self._left.grid_columnconfigure(0, weight=1)

        self._right = ctk.CTkFrame(self._root, corner_radius=16)
        self._right.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        self._right.grid_rowconfigure(7, weight=1)
        self._right.grid_columnconfigure(0, weight=1)

        # Video canvas
        self._video_label = ctk.CTkLabel(self._left, text="", corner_radius=16)
        self._video_label.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        # Right panel widgets
        title = ctk.CTkLabel(
            self._right,
            text="Bảng Điều Khiển",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self._status_label = ctk.CTkLabel(
            self._right,
            text="Trạng thái: —",
            font=ctk.CTkFont(size=16, weight="normal"),
            anchor="w",
        )
        self._status_label.grid(row=1, column=0, sticky="ew", padx=16, pady=6)

        self._grade_label = ctk.CTkLabel(
            self._right,
            text="Phân loại: —",
            font=ctk.CTkFont(size=16, weight="normal"),
            anchor="w",
        )
        self._grade_label.grid(row=2, column=0, sticky="ew", padx=16, pady=6)

        self._confidence_label = ctk.CTkLabel(
            self._right,
            text="Độ tin cậy: —",
            font=ctk.CTkFont(size=16, weight="normal"),
            anchor="w",
        )
        self._confidence_label.grid(row=3, column=0, sticky="ew", padx=16, pady=6)

        self._fps_label = ctk.CTkLabel(
            self._right,
            text="FPS: —",
            font=ctk.CTkFont(size=16, weight="normal"),
            anchor="w",
        )
        self._fps_label.grid(row=4, column=0, sticky="ew", padx=16, pady=6)

        # Enhanced quality info section
        quality_frame = ctk.CTkFrame(self._right, corner_radius=12, fg_color="#1a1a2e")
        quality_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=6)
        quality_frame.grid_columnconfigure(0, weight=1)
        
        quality_title = ctk.CTkLabel(
            quality_frame,
            text="📊 Phân Tích Chất Lượng",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        quality_title.grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))
        
        self._quality_score_label = ctk.CTkLabel(
            quality_frame,
            text="Điểm chất lượng: —",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self._quality_score_label.grid(row=1, column=0, sticky="ew", padx=12, pady=2)
        
        self._color_analysis_label = ctk.CTkLabel(
            quality_frame,
            text="Màu sắc: —",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self._color_analysis_label.grid(row=2, column=0, sticky="ew", padx=12, pady=2)
        
        self._spot_label = ctk.CTkLabel(
            quality_frame,
            text="Đốm/Khuyết điểm: —",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self._spot_label.grid(row=3, column=0, sticky="ew", padx=12, pady=(2, 8))

        # Controls
        controls = ctk.CTkFrame(self._right, corner_radius=14)
        controls.grid(row=6, column=0, sticky="ew", padx=16, pady=(14, 16))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        self._btn_start = ctk.CTkButton(
            controls,
            text="Bắt đầu Camera",
            corner_radius=12,
            command=self._start,
        )
        self._btn_start.grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=12)

        self._btn_stop = ctk.CTkButton(
            controls,
            text="Dừng",
            corner_radius=12,
            fg_color="#444444",
            hover_color="#3a3a3a",
            command=self._stop,
        )
        self._btn_stop.grid(row=0, column=1, sticky="ew", padx=(8, 12), pady=12)

        hint = ctk.CTkLabel(
            self._right,
            text=(
                "Gợi ý: Đặt 1 quả chuối trước nền đơn giản\n"
                "và đủ ánh sáng để phân loại ổn định."
            ),
            font=ctk.CTkFont(size=13),
            justify="left",
            anchor="w",
        )
        hint.grid(row=7, column=0, sticky="sw", padx=16, pady=(0, 16))

    # ---------- Thread handoff ----------

    def _on_frame_from_thread(self, packet: FramePacket) -> None:
        # Never touch tkinter from the capture thread.
        # Copy frame to avoid sharing mutable memory between threads.
        safe_packet = FramePacket(frame_bgr=packet.frame_bgr.copy(), grade=packet.grade, fps=packet.fps)
        with self._packet_lock:
            self._latest_packet = safe_packet

    # ---------- UI loop ----------

    def _ui_tick(self) -> None:
        packet = None
        with self._packet_lock:
            packet = self._latest_packet
            self._latest_packet = None

        if packet is not None:
            annotated = self._annotate_frame(packet.frame_bgr, packet.grade)
            self._render_frame(annotated)
            self._update_right_panel(packet.grade, packet.fps)

        self._root.after(20, self._ui_tick)

    def _update_right_panel(self, grade: GradeResult, fps: float) -> None:
        if grade.category_key == "none":
            self._status_label.configure(text="Trạng thái: —")
            self._grade_label.configure(text=f"Phân loại: {grade.label_vi}")
            self._confidence_label.configure(text="Độ tin cậy: —")
            self._quality_score_label.configure(text="Điểm chất lượng: —")
            self._color_analysis_label.configure(text="Màu sắc: —")
            self._spot_label.configure(text="Đốm/Khuyết điểm: —")
        else:
            self._status_label.configure(text=f"Trạng thái: {grade.status_vi}")
            self._grade_label.configure(text=f"Phân loại: {grade.label_vi}")
            
            # Show enhanced confidence with refinement indicator
            conf_text = f"Độ tin cậy: {grade.confidence * 100:.0f}%"
            if grade.refined:
                conf_text += " ✓ (refined)"
            self._confidence_label.configure(text=conf_text)
            
            # Show quality score with color indicator
            if grade.quality_score > 0:
                quality_text = f"Điểm chất lượng: {grade.quality_score:.0%}"
                if grade.quality_score >= 0.7:
                    quality_text += " 🟢 Tốt"
                elif grade.quality_score >= 0.4:
                    quality_text += " 🟡 Trung bình"
                else:
                    quality_text += " 🔴 Kém"
                self._quality_score_label.configure(text=quality_text)
            else:
                self._quality_score_label.configure(text="Điểm chất lượng: —")
            
            # Show color analysis
            if grade.color_features:
                yellow = grade.color_features.get("yellow_ratio", 0) * 100
                green = grade.color_features.get("green_ratio", 0) * 100
                brown = grade.color_features.get("brown_ratio", 0) * 100
                color_text = f"Màu sắc: 🟡{yellow:.0f}% 🟢{green:.0f}% 🟤{brown:.0f}%"
                self._color_analysis_label.configure(text=color_text)
            else:
                self._color_analysis_label.configure(text="Màu sắc: —")
            
            # Show spot count
            if grade.spot_count > 0:
                spot_text = f"Đốm/Khuyết điểm: {grade.spot_count} đốm"
                if grade.spot_count > 10:
                    spot_text += " ⚠️"
                self._spot_label.configure(text=spot_text)
            else:
                self._spot_label.configure(text="Đốm/Khuyết điểm: Không có")

        self._fps_label.configure(text=f"FPS: {fps:.1f}")

    # ---------- Rendering ----------

    @staticmethod
    def _grade_color_bgr(key: str) -> Tuple[int, int, int]:
        # Required mapping:
        # Green (Unripe), Blue (Export), Orange (Overripe), Red (Defective).
        if key == "unripe":
            return (0, 255, 0)
        if key == "export":
            return (255, 0, 0)
        if key == "overripe":
            return (0, 165, 255)
        if key == "defective":
            return (0, 0, 255)
        return (180, 180, 180)

    def _annotate_frame(self, frame_bgr: np.ndarray, grade: GradeResult) -> np.ndarray:
        out = frame_bgr
        if grade.bbox_xyxy is None:
            return out

        x1, y1, x2, y2 = grade.bbox_xyxy
        color = self._grade_color_bgr(grade.category_key)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label = f"{grade.label_vi}  •  {grade.confidence * 100:.0f}%"
        label_en = f"{grade.label_en}  •  {grade.confidence * 100:.0f}%"
        text_y = max(6, y1 - 36)
        self._text.draw_label(
            out,
            text_vi=label,
            text_en=label_en,
            xy=(x1, text_y),
            color_bgr=color,
            bg_bgr=(0, 0, 0),
        )
        
        # Draw enhanced quality overlay (below bbox)
        if grade.quality_score > 0 and grade.color_features:
            quality_y = y2 + 10
            h, w = out.shape[:2]
            if quality_y + 60 < h:  # Make sure there's space
                self._text.draw_quality_info(
                    out,
                    quality_score=grade.quality_score,
                    spot_count=grade.spot_count,
                    color_features=grade.color_features,
                    xy=(x1, quality_y),
                )
        
        return out

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        # Fit into label size while keeping aspect ratio
        target_w = max(640, self._left.winfo_width() - 60)
        target_h = max(360, self._left.winfo_height() - 60)

        h, w = frame_bgr.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # CustomTkinter image wrapper (keep reference on self)
        self._ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=(new_w, new_h))
        self._video_label.configure(image=self._ctk_image, text="")

    # ---------- Controls ----------

    def _start(self) -> None:
        if self._video_thread.is_running():
            return
        self._video_thread.start()

    def _stop(self) -> None:
        if not self._video_thread.is_running():
            return
        self._video_thread.stop()

    def _on_close(self) -> None:
        try:
            self._video_thread.stop()
        finally:
            self._root.destroy()
