from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FontConfig:
    font_path: str
    font_size: int = 22


class UnicodeTextRenderer:
    """Draws Vietnamese/Unicode text on an OpenCV BGR frame using Pillow.

    Why: cv2.putText() does not support Vietnamese Unicode well on many systems.

    Behavior:
    - If the TTF font cannot be loaded, falls back to English text via cv2.putText
      and prints a warning once.
    """

    def __init__(self, font_config: FontConfig):
        self._font_config = font_config
        self._font: Optional[ImageFont.FreeTypeFont] = None
        self._warned = False

    def _load_font(self) -> Optional[ImageFont.FreeTypeFont]:
        if self._font is not None:
            return self._font
        try:
            self._font = ImageFont.truetype(self._font_config.font_path, self._font_config.font_size)
            return self._font
        except Exception:
            if not self._warned:
                self._warned = True
                print(
                    f"[WARN] Không tải được font tiếng Việt: {self._font_config.font_path}. "
                    "Fallback sang English (cv2.putText).\n"
                    "       Hãy đặt font .ttf hợp lệ (vd: C:/Windows/Fonts/arial.ttf) "
                    "hoặc copy font vào assets/fonts/."
                )
            return None

    @staticmethod
    def _clamp_xy(frame: np.ndarray, x: int, y: int) -> Tuple[int, int]:
        h, w = frame.shape[:2]
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def draw_label(
        self,
        frame_bgr: np.ndarray,
        text_vi: str,
        text_en: str,
        xy: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        bg_bgr: Tuple[int, int, int] = (0, 0, 0),
        padding: int = 6,
    ) -> np.ndarray:
        """Draw text at xy (top-left). Returns the same frame for convenience."""

        x, y = self._clamp_xy(frame_bgr, int(xy[0]), int(xy[1]))
        font = self._load_font()

        if font is None:
            # English fallback with OpenCV
            cv2.putText(
                frame_bgr,
                text_en,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
            return frame_bgr

        # PIL rendering (supports Unicode)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Measure text box
        text = text_vi
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        box_w = (right - left) + padding * 2
        box_h = (bottom - top) + padding * 2

        # Background rectangle
        draw.rounded_rectangle(
            (x, y, x + box_w, y + box_h),
            radius=10,
            fill=(bg_bgr[2], bg_bgr[1], bg_bgr[0]),
            outline=None,
        )

        # Text
        draw.text(
            (x + padding, y + padding),
            text,
            font=font,
            fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
        )

        out_rgb = np.array(pil_img)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr[:, :, :] = out_bgr
        return frame_bgr
