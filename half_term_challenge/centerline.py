import time
from typing import List, Optional
import numpy as np
import cv2


class CenterLineDetector:
    """
    Detector de línea central mejorado.

    Mejoras respecto a la versión original:
    - ROI adaptativo: empieza en 60% del frame para ver la curva antes
    - Error normalizado suavizado (EMA doble) listo para consumirse en PID
    - Historial de últimas N posiciones para detección de curva
    - Penalización por anchura de contorno refinada (evita bordes del carril)
    - Timeout configurable por parámetro
    """

    def __init__(self,
                 alpha: float = 0.35,
                 lost_timeout: float = 2.0,
                 roi_top_frac: float = 0.60,
                 roi_left_frac: float = 0.0,
                 roi_right_frac: float = 1.0,
                 lookahead_weight: float = 0.0):
        """
        alpha           : factor EMA para suavizado de posición  (0 < α < 1)
        lost_timeout    : segundos sin detección antes de reportar None
        roi_top_frac    : fracción vertical donde empieza el ROI
        roi_left_frac   : fracción horizontal del borde izquierdo del ROI
        roi_right_frac  : fracción horizontal del borde derecho del ROI
        lookahead_weight: ignorado (compatibilidad con versiones anteriores)
        """
        self.alpha          = alpha
        self.lost_timeout   = lost_timeout
        self.roi_top_frac   = roi_top_frac
        self.roi_left_frac  = roi_left_frac
        self.roi_right_frac = roi_right_frac

        # Historial de posiciones (últimas 5)
        self._history: List[int] = []
        self._history_size = 5

        # Estado para cálculo de error derivativo externo
        self.smooth_x          : Optional[float] = None
        self._last_detect_time : Optional[float] = None

        # Error normalizado suavizado (para el PID)
        self._smooth_error     : float = 0.0
        self._error_alpha      : float = 0.25  # más suavizado (era 0.4)

    # ─────────────────────────────────────────────────────────────────────
    def reset_memory(self):
        """Llamar tras un giro largo para no arrastrar posición antigua."""
        self._history.clear()
        self.smooth_x          = None
        self._last_detect_time = None
        self._smooth_error     = 0.0

    # ─────────────────────────────────────────────────────────────────────
    @property
    def smooth_error(self) -> float:
        """Error normalizado suavizado (última lectura). Úsalo en el PID."""
        return self._smooth_error

    # ─────────────────────────────────────────────────────────────────────
    def detect_center_line(self, image: np.ndarray):
        """
        Devuelve (cx, cy) en coordenadas del frame completo, o None si la
        línea se perdió por más de `lost_timeout` segundos.

        También actualiza self.smooth_error para el controlador PID.
        """
        h, w = image.shape[:2]
        center_x = w // 2

        # ── ROI ─────────────────────────────────────────────────────────
        roi_y  = int(self.roi_top_frac   * h)
        roi_x1 = int(self.roi_left_frac  * w)
        roi_x2 = int(self.roi_right_frac * w)
        roi    = image[roi_y:h, roi_x1:roi_x2]

        # ── Preprocesado ─────────────────────────────────────────────────
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # ── Contornos ─────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Referencia de predicción lineal sobre el historial
        if len(self._history) >= 2:
            delta       = self._history[-1] - self._history[-2]
            delta       = max(min(delta, 25), -25)
            reference_x = int(self._history[-1] + delta)
        elif self._history:
            reference_x = self._history[-1]
        else:
            reference_x = center_x

        # ── Scoring de candidatos ─────────────────────────────────────
        best_candidate = None
        best_score     = float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)

            # Filtro de inicialización: primera vez, solo acepta contornos
            # cercanos al centro (evita arrancar en un borde incorrecto)
            if not self._history and abs(cx - center_x) > 60:
                continue

            # Penalización por anchura excesiva (bordes del carril)
            width_penalty = 150 if w_b > 60 else 0

            # Distancia al centro del frame
            dist_center = abs(cx - center_x) * 0.5

            # Penalización por salto respecto a la predicción
            in_motion   = (len(self._history) >= 2 and
                           abs(self._history[-1] - self._history[-2]) > 20)
            jump_weight = 0.7 if in_motion else 2.0
            jump        = abs(cx - reference_x) * jump_weight

            # Premio a contornos grandes y altos (más visibles)
            size_bonus  = -(h_b * 0.5 + area / 120.0)

            score = dist_center + jump + size_bonus + width_penalty

            if score < best_score:
                best_score     = score
                best_candidate = (cx + roi_x1, cy + roi_y)

        # ── Sin candidato válido ──────────────────────────────────────
        if best_candidate is None:
            now = time.time()
            if (self._last_detect_time is None or
                    now - self._last_detect_time > self.lost_timeout):
                self._smooth_error = 0.0
                return None

            # Dentro del timeout: mantener última posición conocida
            last_x = self._history[-1] if self._history else center_x
            raw_error = (last_x - center_x) / center_x
            self._smooth_error = (self._error_alpha * raw_error +
                                  (1 - self._error_alpha) * self._smooth_error)
            return (last_x, int(0.9 * h))

        # ── Candidato encontrado ──────────────────────────────────────
        self._last_detect_time = time.time()

        cx, cy = best_candidate

        # EMA sobre posición
        if self.smooth_x is None:
            self.smooth_x = float(cx)
        else:
            self.smooth_x = self.alpha * cx + (1 - self.alpha) * self.smooth_x

        # Historial de posiciones (para predicción)
        self._history.append(cx)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # Error normalizado suavizado (directo para el PID)
        raw_error = (self.smooth_x - center_x) / center_x
        self._smooth_error = (self._error_alpha * raw_error +
                              (1 - self._error_alpha) * self._smooth_error)

        return (int(self.smooth_x), cy)

    # ─────────────────────────────────────────────────────────────────────
    def draw_debug(self, image: np.ndarray, result) -> np.ndarray:
        h, w = image.shape[:2]
        roi_y = int(self.roi_top_frac * h)

        # ROI box (with horizontal crop)
        roi_x1 = int(getattr(self, 'roi_left_frac',  0.0) * w)
        roi_x2 = int(getattr(self, 'roi_right_frac', 1.0) * w)
        cv2.rectangle(image, (roi_x1, roi_y), (roi_x2, h), (100, 100, 255), 1)
        # Línea central
        cv2.line(image, (w // 2, roi_y), (w // 2, h), (255, 255, 0), 1)

        if result is not None:
            cx, cy = result
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(image, (w // 2, cy), (cx, cy), (0, 255, 0), 2)

            # Error suavizado
            cv2.putText(image,
                        f"err={self._smooth_error:+.3f}",
                        (10, roi_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return image