import time
import numpy as np
import cv2


class CenterLineDetector:
    """
    Detector de la línea negra central.

    Usa el detector original (Otsu + contornos) que funciona bien en curvas,
    con dos mejoras:
      1. ROI recortada lateralmente (40% central) para ignorar bordes y cuadraditos.
      2. Devuelve el error normalizado [-1, 1] directamente, además de (x, y).
         error > 0 → línea a la derecha del centro → girar derecha (Wc < 0)
         error < 0 → línea a la izquierda           → girar izquierda (Wc > 0)
    """

    def __init__(self):
        self.last_x   = None
        self.prev_x   = None
        self.alpha    = 0.3       # suavizado EMA de la posición

        self._last_detect_time = None
        self.lost_timeout      = 2.0   # s → devuelve None

    def reset_memory(self):
        self.last_x   = None
        self.prev_x   = None
        self._last_detect_time = None

    # ──────────────────────────────────────────────────────────────────────
    def detect(self, image):
        """
        Retorna (error_norm, cx, cy) o None si línea perdida.
        error_norm ∈ [-1, 1]: negativo=izquierda, positivo=derecha.
        """
        h, w, _ = image.shape

        # ── ROI: cuarto inferior, franja central 40% ──────────────────────
        roi_full = image[int(3 * h / 4):h, :]
        rh, rw   = roi_full.shape[:2]
        x_margin = int(rw * 0.30)          # 30% cada lado → 40% central
        roi      = roi_full[:, x_margin: rw - x_margin]
        x_offset = x_margin

        # ── Threshold Otsu inverso ────────────────────────────────────────
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        k = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        center_x = w // 2
        if self.last_x is None:
            reference_x = center_x
        elif self.prev_x is not None:
            delta = np.clip(self.last_x - self.prev_x, -20, 20)
            reference_x = int(self.last_x + delta)
        else:
            reference_x = self.last_x

        best_score = float('inf')
        best_cx    = None
        best_cy    = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"]) + x_offset   # frame completo
            cy = int(M["m01"] / M["m00"])

            bottom_pt = tuple(cnt[cnt[:, :, 1].argmax()][0])
            cx_bot    = bottom_pt[0] + x_offset

            bx, by, bw, bh = cv2.boundingRect(cnt)

            dist_center   = abs(cx - center_x)
            height_bonus  = -bh
            area_bonus    = -area / 100.0
            score         = dist_center + height_bonus + area_bonus

            if self.last_x is not None:
                motion      = abs(self.last_x - self.prev_x) if self.prev_x is not None else 0
                jump_w      = 0.8 if motion > 25 else 2.0
                score      += abs(cx - reference_x) * jump_w

            score += abs(cx_bot - center_x) * 2.0

            if self.last_x is None and abs(cx - center_x) > 50:
                continue
            if bw > 50:
                score += 200

            if score < best_score:
                best_score = score
                best_cx    = cx
                best_cy    = cy + int(3 * h / 4)

        # ── Sin candidato ─────────────────────────────────────────────────
        if best_cx is None:
            now = time.time()
            if (self._last_detect_time is None or
                    now - self._last_detect_time > self.lost_timeout):
                return None
            # Dentro del timeout: mantener último x conocido
            x = self.last_x if self.last_x is not None else center_x
            err = (x - center_x) / float(center_x)
            return (float(np.clip(err, -1, 1)), x, int(0.9 * h))

        # ── Candidato ─────────────────────────────────────────────────────
        self._last_detect_time = time.time()
        self.prev_x = self.last_x

        # Suavizado EMA
        if self.last_x is None:
            self.last_x = float(best_cx)
        else:
            self.last_x = self.alpha * best_cx + (1 - self.alpha) * self.last_x

        smooth_cx = int(self.last_x)
        err_norm  = (smooth_cx - center_x) / float(center_x)
        return (float(np.clip(err_norm, -1, 1)), smooth_cx, best_cy)

    # ── Alias para compatibilidad con draw_debug del controller ───────────
    def detect_center_line(self, image):
        result = self.detect(image)
        if result is None:
            return None
        _, cx, cy = result
        return (cx, cy)

    def draw_debug(self, image, result):
        h, w = image.shape[:2]
        roi_y    = int(3 * h / 4)
        x_margin = int(w * 0.30)

        cv2.rectangle(image,
                      (x_margin, roi_y), (w - x_margin, h),
                      (0, 0, 255), 2)
        cv2.line(image, (w // 2, roi_y), (w // 2, h), (255, 255, 0), 1)

        if result is not None:
            cx, cy = result
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(image, (w // 2, cy), (cx, cy), (0, 255, 0), 2)
        return image