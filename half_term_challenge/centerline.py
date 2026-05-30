import time
from typing import List, Optional, Tuple
import numpy as np
import cv2


class CenterLineDetector:
    """
    Detector de línea central con zona dual (cercana + lejana).

    Cambios respecto a la versión anterior:
    - roi_top_frac por defecto bajado a 0.45  → ve ~15% más lejos
    - Detección en DOS bandas dentro del ROI:
        · Zona lejana  (roi_top  … roi_mid): anticipa curvas
        · Zona cercana (roi_mid  … bottom):  estabiliza la posición actual
    - smooth_x mezcla ambas zonas con lookahead_weight (0-1)
        0.0 = solo zona cercana  |  1.0 = solo zona lejana
        Recomendado: 0.35 – 0.45 para pistas de laboratorio
    - La zona lejana usa su propio EMA más rápido (alpha_far)
    - API pública idéntica a la versión anterior (drop-in replacement)
    """

    def __init__(self,
                 alpha: float            = 0.35,
                 lost_timeout: float     = 2.0,
                 roi_top_frac: float     = 0.45,   # ← subido desde 0.60
                 roi_left_frac: float    = 0.15,   # fracción horizontal izquierda a ignorar
                 roi_right_frac: float   = 0.85,   # fracción horizontal derecha a ignorar
                 lookahead_weight: float = 0.40,   # fracción de influencia de la zona lejana
                 alpha_far: float        = 0.50):  # EMA más rápido para la zona lejana
        """
        alpha           : EMA de posición para la zona cercana  (0 < α < 1)
        lost_timeout    : segundos sin detección antes de reportar None
        roi_top_frac    : fracción vertical donde empieza el ROI  (0 = top)
        roi_left_frac   : fracción horizontal del borde izquierdo del ROI (0–1)
        roi_right_frac  : fracción horizontal del borde derecho del ROI  (0–1)
        lookahead_weight: peso de la zona lejana en la posición mezclada (0–1)
        alpha_far       : EMA de posición para la zona lejana
        """
        self.alpha            = alpha
        self.lost_timeout     = lost_timeout
        self.roi_top_frac     = roi_top_frac
        self.roi_left_frac    = roi_left_frac
        self.roi_right_frac   = roi_right_frac
        self.lookahead_weight = lookahead_weight
        self.alpha_far        = alpha_far

        # ── Historial de posiciones (últimas N de la zona cercana) ────────
        self._history: List[int] = []
        self._history_size = 5

        # ── Estado suavizado ──────────────────────────────────────────────
        self.smooth_x      : Optional[float] = None   # zona cercana
        self._smooth_x_far : Optional[float] = None   # zona lejana

        self._last_detect_time: Optional[float] = None

        # Error normalizado suavizado (para el PID)
        self._smooth_error : float = 0.0
        self._error_alpha  : float = 0.3

    # ─────────────────────────────────────────────────────────────────────
    def reset_memory(self):
        """Llamar tras un giro largo para no arrastrar posición antigua."""
        self._history.clear()
        self.smooth_x          = None
        self._smooth_x_far     = None
        self._last_detect_time = None
        self._smooth_error     = 0.0

    # ─────────────────────────────────────────────────────────────────────
    @property
    def smooth_error(self) -> float:
        """Error normalizado suavizado (última lectura). Úsalo en el PID."""
        return self._smooth_error

    # ─────────────────────────────────────────────────────────────────────
    def _best_candidate(self,
                        band: np.ndarray,
                        band_y_offset: int,
                        center_x: int,
                        reference_x: int,
                        require_center_init: bool) -> Optional[Tuple[int, int]]:
        """
        Extrae el mejor candidato de contorno dentro de una banda de imagen.
        Devuelve (cx, cy) en coordenadas del frame completo, o None.
        """
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_candidate = None
        best_score     = float('inf')

        in_motion = (len(self._history) >= 2 and
                     abs(self._history[-1] - self._history[-2]) > 20)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            _, _, w_b, h_b = cv2.boundingRect(cnt)

            if require_center_init and not self._history:
                if abs(cx - center_x) > 60:
                    continue

            width_penalty = 150 if w_b > 60 else 0
            dist_center   = abs(cx - center_x) * 0.5
            jump_weight   = 0.7 if in_motion else 2.0
            jump          = abs(cx - reference_x) * jump_weight
            size_bonus    = -(h_b * 0.5 + area / 120.0)

            score = dist_center + jump + size_bonus + width_penalty

            if score < best_score:
                best_score     = score
                best_candidate = (cx, cy + band_y_offset)

        return best_candidate

    # ─────────────────────────────────────────────────────────────────────
    def detect_center_line(self, image: np.ndarray):
        """
        Devuelve (cx, cy) en coordenadas del frame completo, o None si la
        línea se perdió por más de `lost_timeout` segundos.

        También actualiza self.smooth_error para el controlador PID.

        La posición cx mezcla zona cercana y lejana:
            cx = (1 - lookahead_weight) * near_x + lookahead_weight * far_x
        """
        h, w = image.shape[:2]
        center_x = w // 2

        # ── Límites del ROI y de las dos bandas ───────────────────────────
        roi_y   = int(self.roi_top_frac   * h)   # top del ROI completo
        roi_mid = roi_y + (h - roi_y) // 2       # mitad del ROI (frontera far/near)
        roi_x1  = int(self.roi_left_frac  * w)   # borde izquierdo (ignora bordes)
        roi_x2  = int(self.roi_right_frac * w)   # borde derecho

        # Bandas de imagen (recortadas horizontal y verticalmente)
        band_far  = image[roi_y  : roi_mid, roi_x1 : roi_x2]
        band_near = image[roi_mid : h,       roi_x1 : roi_x2]

        # ── Predicción lineal ─────────────────────────────────────────────
        if len(self._history) >= 2:
            delta       = self._history[-1] - self._history[-2]
            delta       = max(min(delta, 25), -25)
            reference_x = int(self._history[-1] + delta)
        elif self._history:
            reference_x = self._history[-1]
        else:
            reference_x = center_x

        # ── Detección en cada banda ───────────────────────────────────────
        # center_x y reference_x se ajustan al espacio recortado horizontalmente
        cropped_center_x    = center_x    - roi_x1
        cropped_reference_x = reference_x - roi_x1

        candidate_near = self._best_candidate(
            band_near, roi_mid, cropped_center_x, cropped_reference_x,
            require_center_init=True
        )
        candidate_far  = self._best_candidate(
            band_far, roi_y, cropped_center_x, cropped_reference_x,
            require_center_init=False   # zona lejana: sin restricción de init
        )

        # Convertir cx de vuelta a coordenadas del frame completo
        if candidate_near is not None:
            candidate_near = (candidate_near[0] + roi_x1, candidate_near[1])
        if candidate_far is not None:
            candidate_far  = (candidate_far[0]  + roi_x1, candidate_far[1])

        # ── Sin candidato en zona cercana (principal) ─────────────────────
        if candidate_near is None:
            now = time.time()
            if (self._last_detect_time is None or
                    now - self._last_detect_time > self.lost_timeout):
                self._smooth_error = 0.0
                return None

            last_x = self._history[-1] if self._history else center_x
            raw_error = (last_x - center_x) / center_x
            self._smooth_error = (self._error_alpha * raw_error +
                                  (1 - self._error_alpha) * self._smooth_error)
            return (last_x, int(0.9 * h))

        # ── Candidato cercano encontrado ──────────────────────────────────
        self._last_detect_time = time.time()

        near_cx, near_cy = candidate_near

        # EMA zona cercana
        if self.smooth_x is None:
            self.smooth_x = float(near_cx)
        else:
            self.smooth_x = self.alpha * near_cx + (1 - self.alpha) * self.smooth_x

        # Historial (solo zona cercana, más estable)
        self._history.append(near_cx)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # ── Zona lejana (opcional, más rápida) ────────────────────────────
        if candidate_far is not None:
            far_cx, _ = candidate_far
            if self._smooth_x_far is None:
                self._smooth_x_far = float(far_cx)
            else:
                self._smooth_x_far = (self.alpha_far * far_cx +
                                      (1 - self.alpha_far) * self._smooth_x_far)
        else:
            # Sin detección lejana: usar posición cercana como fallback
            if self._smooth_x_far is None:
                self._smooth_x_far = self.smooth_x
            # (mantener último valor conocido)

        # ── Mezcla near + far ─────────────────────────────────────────────
        blended_x = ((1 - self.lookahead_weight) * self.smooth_x +
                      self.lookahead_weight       * self._smooth_x_far)

        # ── Error normalizado suavizado (para el PID) ─────────────────────
        raw_error = (blended_x - center_x) / center_x
        self._smooth_error = (self._error_alpha * raw_error +
                              (1 - self._error_alpha) * self._smooth_error)

        return (int(blended_x), near_cy)

    # ─────────────────────────────────────────────────────────────────────
    def draw_debug(self, image: np.ndarray, result) -> np.ndarray:
        h, w    = image.shape[:2]
        roi_y   = int(self.roi_top_frac   * h)
        roi_mid = roi_y + (h - roi_y) // 2
        roi_x1  = int(self.roi_left_frac  * w)
        roi_x2  = int(self.roi_right_frac * w)

        # ROI completo con recorte horizontal
        cv2.rectangle(image, (roi_x1, roi_y), (roi_x2, h), (100, 100, 255), 1)
        # Frontera entre bandas far/near
        cv2.line(image, (roi_x1, roi_mid), (roi_x2, roi_mid), (200, 100, 255), 1)
        # Etiquetas de banda
        cv2.putText(image, "FAR",  (roi_x1 + 4, roi_y  + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 100, 255), 1)
        cv2.putText(image, "NEAR", (roi_x1 + 4, roi_mid + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 255), 1)
        # Línea central del frame
        cv2.line(image, (w // 2, roi_y), (w // 2, h), (255, 255, 0), 1)

        # Posición de la zona lejana (si existe)
        if self._smooth_x_far is not None:
            far_x = int(self._smooth_x_far)
            cv2.circle(image, (far_x, roi_y + (roi_mid - roi_y) // 2), 5, (255, 100, 200), -1)

        if result is not None:
            cx, cy = result
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(image, (w // 2, cy), (cx, cy), (0, 255, 0), 2)

            cv2.putText(image,
                        f"err={self._smooth_error:+.3f}  lk={self.lookahead_weight:.2f}",
                        (10, roi_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return image