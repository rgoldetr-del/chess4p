"""human_input_watcher.py — слежение за ходом только по таймерам.

Улучшения:
- путь к калибровке берётся из config.TIMER_CALIB_FILE;
- поддержка старого формата timer_calibration.py;
- score использует brightness / bright_ratio / saturation;
- score нормализуется и клампится в [0, 1].
"""

import json
import os
import time

import cv2
import numpy as np

from config import (
    TIMER_CALIB_FILE,
    TIMER_CFG,
    COLOR_TO_IDX,
)

from screen_capture import capture_screen


COLOR_NAMES = ["RED", "BLUE", "YELLOW", "GREEN"]

# Размер индикатора для ручной калибровки.
INDICATOR_SIZE = 12


def _normalize_color_key(k):
    """
    Поддерживает:
        "0", 0, "red", "RED"
    """
    if str(k).isdigit():
        return int(k)

    return COLOR_TO_IDX.get(str(k).lower(), None)


def _normalize_features(f):
    if not isinstance(f, dict):
        return {
            "brightness": 0.0,
            "saturation": 0.0,
            "bright_ratio": 0.0,
        }

    return {
        "brightness": float(f.get("brightness", 0.0)),
        "saturation": float(f.get("saturation", 0.0)),
        "bright_ratio": float(f.get("bright_ratio", 0.0)),
    }


class TimerDetector:
    """У каждого цвета свой регион и свои активный/неактивный образцы."""

    def __init__(self, path=None):
        self.path = path or TIMER_CALIB_FILE

        self.regions = {}
        self.active = {}
        self.inactive = {}

        self.load()

    # ============================================================
    # Load / save
    # ============================================================

    def load(self):
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠ TimerDetector load failed: {e}")
            return

        # Regions: int keys or color names.
        regions_raw = data.get("regions", {})

        for k, v in regions_raw.items():
            ci = _normalize_color_key(k)

            if ci is None:
                continue

            try:
                self.regions[ci] = tuple(int(x) for x in v)
            except Exception:
                continue

        # Active/inactive samples.
        active_raw = data.get("active", data.get("active_samples", {}))
        inactive_raw = data.get("inactive", data.get("inactive_samples", {}))

        for k, v in active_raw.items():
            ci = _normalize_color_key(k)

            if ci is not None:
                self.active[ci] = _normalize_features(v)

        for k, v in inactive_raw.items():
            ci = _normalize_color_key(k)

            if ci is not None:
                self.inactive[ci] = _normalize_features(v)

    def save(self):
        data = {
            "regions": {
                str(k): list(v)
                for k, v in self.regions.items()
            },
            "active": {
                str(k): v
                for k, v in self.active.items()
            },
            "inactive": {
                str(k): v
                for k, v in self.inactive.items()
            },
        }

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def is_ready(self):
        return (
            len(self.regions) == 4
            and len(self.active) == 4
            and len(self.inactive) == 4
        )

    # ============================================================
    # Features / score
    # ============================================================

    @staticmethod
    def _features(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return {
            "brightness": float(img.mean()),
            "saturation": float(hsv[:, :, 1].mean()),
            "bright_ratio": float(np.mean(img > 150)),
        }

    def _score(self, color):
        """
        0 = неактивен.
        1 = активен.
        """
        if color not in self.regions:
            return 0.0

        if color not in self.active or color not in self.inactive:
            return 0.0

        img = capture_screen(self.regions[color])

        if img is None or img.size == 0:
            return 0.0

        f = self._features(img)
        a = self.active[color]
        i = self.inactive[color]

        w_br = float(TIMER_CFG.get("brightness_weight", 0.45))
        w_rt = float(TIMER_CFG.get("bright_ratio_weight", 0.30))
        w_sat = float(TIMER_CFG.get("saturation_weight", 0.25))

        w_sum = w_br + w_rt + w_sat

        if w_sum <= 0.0:
            return 0.0

        def norm(cur, act, inact, eps):
            rng = act - inact

            if abs(rng) <= eps:
                return 0.0

            return (cur - inact) / rng

        norm_br = norm(
            f["brightness"],
            a["brightness"],
            i["brightness"],
            1.0,
        )

        norm_rt = norm(
            f["bright_ratio"],
            a["bright_ratio"],
            i["bright_ratio"],
            0.01,
        )

        norm_sat = norm(
            f["saturation"],
            a["saturation"],
            i["saturation"],
            1.0,
        )

        score = (
            w_br * norm_br
            + w_rt * norm_rt
            + w_sat * norm_sat
        ) / w_sum

        return float(np.clip(score, 0.0, 1.0))

    # ============================================================
    # Detection
    # ============================================================

    def detect_active(self, min_score=None):
        """color_idx активного таймера или None."""
        if min_score is None:
            min_score = float(TIMER_CFG.get("min_score", 0.35))

        if not self.is_ready():
            return None

        best = None
        best_sc = -1.0

        for c in self.regions:
            sc = self._score(c)

            if sc > best_sc:
                best_sc = sc
                best = c

        if best is None or best_sc < min_score:
            return None

        return best

    def read_stable(self, polls=None, interval=None):
        """Стабильное чтение: одно и то же значение несколько раз подряд."""
        if polls is None:
            polls = int(TIMER_CFG.get("stable_polls", 3))

        if interval is None:
            interval = float(TIMER_CFG.get("stable_interval", 0.12))

        polls = max(2, polls)

        last = None
        ok = 0

        for _ in range(polls):
            cur = self.detect_active()

            if cur == last and cur is not None:
                ok += 1

                if ok >= polls - 1:
                    return cur
            else:
                ok = 0
                last = cur

            time.sleep(interval)

        return last if last is not None else None

    def wait_for_turn_change(
        self,
        current,
        timeout=300.0,
        poll=None,
    ):
        """
        Ждём, пока активный таймер станет НЕ current.

        Возвращает:
            (new_active, elapsed_sec)
            или (None, elapsed_sec) при timeout.
        """
        if poll is None:
            poll = float(TIMER_CFG.get("wait_poll", 0.15))

        t0 = time.time()

        while time.time() - t0 < timeout:
            act = self.read_stable(polls=2, interval=0.10)

            if act is not None and act != current:
                return act, time.time() - t0

            time.sleep(poll)

        return None, time.time() - t0


# ============================================================
# Calibration
# ============================================================

def calibrate_timers():
    import pyautogui

    print("")
    print(
        "⏱️  КАЛИБРОВКА ТАЙМЕРОВ: "
        "4 региона + 4 активных + 4 неактивных"
    )
    print("=" * 70)

    regions = {}
    active = {}
    inactive = {}

    print("")
    print(f"📍 Шаг 1/3: Регионы таймеров ({INDICATOR_SIZE}×{INDICATOR_SIZE} px)")

    for c in range(4):
        input(f"   🎯 Наведите на таймер {COLOR_NAMES[c]} → Enter...")

        pos = pyautogui.position()

        regions[c] = (
            int(pos.x) - INDICATOR_SIZE // 2,
            int(pos.y) - INDICATOR_SIZE // 2,
            INDICATOR_SIZE,
            INDICATOR_SIZE,
        )

        print(f"      ✓ {regions[c]}")

    print("")
    print("📍 Шаг 2/3: АКТИВНЫЕ образцы (таймер подсвечен)")

    for c in range(4):
        input(f"   🟢 {COLOR_NAMES[c]} активен → Enter...")

        active[c] = TimerDetector._features(
            capture_screen(regions[c])
        )

        print(
            f"      ✓ ярк={active[c]['brightness']:.1f} "
            f"насыщ={active[c]['saturation']:.1f} "
            f"ярких={active[c]['bright_ratio']:.2f}"
        )

    print("")
    print("📍 Шаг 3/3: НЕАКТИВНЫЕ образцы (таймер тусклый)")

    for c in range(4):
        input(f"   🔴 {COLOR_NAMES[c]} неактивен → Enter...")

        inactive[c] = TimerDetector._features(
            capture_screen(regions[c])
        )

        print(
            f"      ✓ ярк={inactive[c]['brightness']:.1f} "
            f"насыщ={inactive[c]['saturation']:.1f} "
            f"ярких={inactive[c]['bright_ratio']:.2f}"
        )

    det = TimerDetector()
    det.regions = regions
    det.active = active
    det.inactive = inactive
    det.save()

    print("")
    print(f"✅ Сохранено: {TIMER_CALIB_FILE}")

    print("")
    print("📊 Различимость:")

    for c in range(4):
        d_br = abs(
            active[c]["brightness"]
            - inactive[c]["brightness"]
        )

        d_rt = abs(
            active[c]["bright_ratio"]
            - inactive[c]["bright_ratio"]
        )

        mark = "✓" if d_br > 10 or d_rt > 0.10 else "⚠ МАЛО"

        print(
            f"   {COLOR_NAMES[c]:6s}: "
            f"Δярк={d_br:.1f} "
            f"Δярких={d_rt:.2f} "
            f"{mark}"
        )


if __name__ == "__main__":
    calibrate_timers()