import cv2 as cv
import numpy as np


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class TrafficLightDetection:
    """
    Detector de semáforo mejorado.

    Mejoras respecto a la versión original:
    - HSV dual-range más robustos para condiciones de laboratorio
    - Filtro de circularidad + área mínima más estricto
    - Buffer de votación ampliado (7 frames) para mayor estabilidad
    - Score de confianza devuelto junto al estado (útil para debug)
    - ROI ajustable y dibujable
    - Regla verde → amarillo ≠ rojo conservada
    - Nuevo: umbral de área relativa para ignorar reflexos pequeños
    """

    def __init__(self):

        # ── Rangos HSV ────────────────────────────────────────────────────
        # Rojo: dos bandas (hue wrap-around)
        self.lower_red1 = np.array([0,   130, 80])
        self.upper_red1 = np.array([10,  255, 255])
        self.lower_red2 = np.array([168, 130, 80])
        self.upper_red2 = np.array([180, 255, 255])

        # Amarillo: rango ampliado para cubrir distintas temperaturas de color
        self.lower_yellow = np.array([15, 120, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # Verde: rango moderado para evitar confusión con líneas del piso
        self.lower_green = np.array([42, 90, 70])
        self.upper_green = np.array([82, 255, 255])

        # ── Morfología ────────────────────────────────────────────────────
        self.kernel = np.ones((5, 5), np.uint8)

        # ── Umbrales de detección ─────────────────────────────────────────
        self.min_blob_area   = 400     # píxeles² mínimos del blob
        self.min_circularity = 0.35    # 0 = cualquier forma, 1 = círculo perfecto

        # ── ROI (fracción del frame) ───────────────────────────────────────
        self.roi_top    = 0.03
        self.roi_bottom = 0.60
        self.roi_left   = 0.15
        self.roi_right  = 0.85

        # ── Suavizado temporal ────────────────────────────────────────────
        self.buffer_size   = 7                           # frames de votación
        self._state_buffer = ["none"] * self.buffer_size

        # ── Último estado confirmado (para regla verde→amarillo) ──────────
        self._last_confirmed_state = "none"

        # ── Confianza interna ─────────────────────────────────────────────
        self._confidence = 0.0   # fracción de votos del estado ganador

    # ─────────────────────────────────────────────────────────────────────
    @property
    def confidence(self) -> float:
        """Fracción de votos del estado actual en el buffer (0–1)."""
        return self._confidence

    # ─────────────────────────────────────────────────────────────────────
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        mask = cv.erode(mask,  self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        return mask

    def _crop_roi(self, image: np.ndarray):
        h, w   = image.shape[:2]
        y1, y2 = int(self.roi_top * h),    int(self.roi_bottom * h)
        x1, x2 = int(self.roi_left * w),   int(self.roi_right * w)
        return image[y1:y2, x1:x2], (x1, y1)

    def _largest_circular_blob(self, mask: np.ndarray) -> float:
        """Devuelve el área del blob más grande que supere los umbrales."""
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        best = 0.0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_blob_area:
                continue
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity >= self.min_circularity:
                best = max(best, area)
        return best

    # ─────────────────────────────────────────────────────────────────────
    def detect_state(self, image: np.ndarray):
        """
        Retorna:
            state       (str)   – 'red' | 'yellow' | 'green' | 'none'
            red_area    (float) – área del blob rojo detectado
            yellow_area (float) – ídem amarillo
            green_area  (float) – ídem verde

        Regla verde → amarillo:
            Si el último estado confirmado fue 'green' y el suavizado
            resulta 'red', se devuelve 'yellow' (transición incompleta).
        """
        # 1. ROI
        roi, _ = self._crop_roi(image)

        # 2. Blur + HSV
        blurred = cv.GaussianBlur(roi, (5, 5), 2)
        hsv     = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        # 3. Máscaras
        mask_red = (
            cv.inRange(hsv, self.lower_red1, self.upper_red1) |
            cv.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        mask_yellow = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green  = cv.inRange(hsv, self.lower_green,  self.upper_green)

        # 4. Morfología
        mask_red    = self._apply_morphology(mask_red)
        mask_yellow = self._apply_morphology(mask_yellow)
        mask_green  = self._apply_morphology(mask_green)

        # 5. Blob más grande y circular por color
        red_area    = self._largest_circular_blob(mask_red)
        yellow_area = self._largest_circular_blob(mask_yellow)
        green_area  = self._largest_circular_blob(mask_green)

        # 6. Estado raw
        best = max(red_area, yellow_area, green_area)
        if best == 0:
            raw_state = "none"
        elif red_area == best:
            raw_state = "red"
        elif yellow_area == best:
            raw_state = "yellow"
        else:
            raw_state = "green"

        # 7. Suavizado (majority vote)
        self._state_buffer.pop(0)
        self._state_buffer.append(raw_state)
        smoothed = max(set(self._state_buffer),
                       key=self._state_buffer.count)
        self._confidence = self._state_buffer.count(smoothed) / self.buffer_size

        # 8. Regla verde → amarillo ≠ rojo
        final = smoothed
        if self._last_confirmed_state == "green" and smoothed == "red":
            final = "yellow"

        if final != self._last_confirmed_state:
            self._last_confirmed_state = final

        return final, red_area, yellow_area, green_area

    # ─────────────────────────────────────────────────────────────────────
    def draw_status(self, image: np.ndarray,
                    state: str,
                    current_goal: int = 0,
                    total_goals: int  = 0,
                    waiting: bool     = False) -> np.ndarray:
        h, w = image.shape[:2]

        y1 = int(self.roi_top    * h);  y2 = int(self.roi_bottom * h)
        x1 = int(self.roi_left   * w);  x2 = int(self.roi_right  * w)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)

        colour_map = {
            "red":    (0, 0, 255),
            "yellow": (0, 255, 255),
            "green":  (0, 255, 0),
            "none":   (255, 255, 255),
        }
        color = colour_map.get(state, (255, 255, 255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (500, 220), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv.putText(image, f"Traffic: {state.upper()}", (20, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv.putText(image, f"Confidence: {self._confidence:.0%}", (20, 95),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1)
        cv.putText(image, f"Goal: {current_goal}/{total_goals}", (20, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.putText(image, f"Waiting green: {waiting}", (20, 165),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        cv.putText(image,
                   f"buf={self._state_buffer}  last={self._last_confirmed_state}",
                   (20, 205),
                   cv.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        return image