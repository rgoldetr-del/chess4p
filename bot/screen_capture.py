"""screen_capture.py — захват экрана через mss.

Улучшения:
- переиспользуемый mss-сессия;
- поддержка MONITOR_INDEX из config;
- регион для захвата доски/таймеров.
"""

import numpy as np
import cv2
import mss

from config import MONITOR_INDEX


class ScreenGrabber:
    """Обёртка над mss, чтобы не создавать сессию на каждый кадр."""

    def __init__(self):
        self.sct = mss.mss()

    def grab(self, region=None):
        """
        Захват экрана.

        region:
            None -> весь выбранный монитор.
            (left, top, width, height) -> область.

        Возвращает BGR numpy array.
        """
        if region is None:
            monitors = self.sct.monitors

            # monitors[0] обычно виртуальный полный рабочий стол.
            # monitors[1] обычно основной монитор.
            idx = MONITOR_INDEX
            if idx <= 0 or idx >= len(monitors):
                idx = 1 if len(monitors) > 1 else 0

            monitor = monitors[idx]
        else:
            left, top, width, height = region
            monitor = {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height),
            }

        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass


_GRABBER = None


def get_grabber() -> ScreenGrabber:
    global _GRABBER

    if _GRABBER is None:
        _GRABBER = ScreenGrabber()

    return _GRABBER


def capture_screen(region=None):
    """
    Совместимая функция захвата экрана.

    region:
        None -> весь монитор.
        (left, top, width, height) -> область.
    """
    grabber = get_grabber()
    return grabber.grab(region)