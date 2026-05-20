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
        % (sensor_id, capture_width, capture_height, framerate,
           flip_method, display_width, display_height)
    )


class TrafficLightDetection:
    """
    Detector de semáforo para LEDs físicos en protoboard.

    Semáforo real observado:
        - Rojo:    LED rojo estándar     → hue 0-10 / 170-180
        - Amarillo: LED naranja-ámbar    → hue 10-25  (NO amarillo puro)
        - Verde:   LED verde estándar    → hue 40-80

    Lógica de transición (verde → rojo NO existe):
        Si last_confirmed = "green" y se detecta "red" → tratar como "yellow"
        Solo se puede llegar a "red" desde "yellow" o "none"

    ROI: solo la mitad superior-izquierda del frame donde está el semáforo.
    Detección por blob circular más grande (no por conteo de píxeles).
    """

    def __init__(self):
        # ── HSV: Rojo ─────────────────────────────────────────────────────
        self.lower_red1 = np.array([0,   120, 60])
        self.upper_red1 = np.array([10,  255, 255])
        self.lower_red2 = np.array([170, 120, 60])
        self.upper_red2 = np.array([180, 255, 255])

        # ── HSV: Amarillo/Naranja-ámbar ───────────────────────────────────
        # LED naranja-ámbar real: hue 10-25, saturación alta
        self.lower_yellow = np.array([10,  120, 60])
        self.upper_yellow = np.array([28,  255, 255])

        # ── HSV: Verde ────────────────────────────────────────────────────
        self.lower_green = np.array([40, 80, 50])
        self.upper_green = np.array([80, 255, 255])

        # ── ROI: semáforo siempre en la mitad izquierda, zona superior ────
        # En imagen se ve el LED rojo en ~20-40% del alto, lado izquierdo
        self.roi_top    = 0.0    # desde arriba
        self.roi_bottom = 0.80   # hasta 80% del alto (más cobertura)
        self.roi_left   = 0.0    # desde la izquierda
        self.roi_right  = 0.50   # hasta 50% del ancho

        # ── Morfología ────────────────────────────────────────────────────
        self.kernel = np.ones((5, 5), np.uint8)

        # ── Umbrales ──────────────────────────────────────────────────────
        self.min_blob_area   = 80     # área mínima del blob LED
        self.min_circularity = 0.3    # LEDs son circulares

        # ── Suavizado temporal ────────────────────────────────────────────
        self.buffer_size   = 5
        self._state_buffer = ["none"] * self.buffer_size

        # ── Regla verde→amarillo≠rojo ─────────────────────────────────────
        self._last_confirmed = "none"

    # ──────────────────────────────────────────────────────────────────────
    def _morphology(self, mask):
        mask = cv.erode(mask,  self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        return mask

    def _crop_roi(self, image):
        h, w = image.shape[:2]
        y1 = int(self.roi_top    * h)
        y2 = int(self.roi_bottom * h)
        x1 = int(self.roi_left   * w)
        x2 = int(self.roi_right  * w)
        return image[y1:y2, x1:x2], (x1, y1)

    def _best_blob(self, mask):
        """Retorna el área del blob circular más grande, o 0."""
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        best = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_blob_area:
                continue
            perim = cv.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ >= self.min_circularity:
                best = max(best, area)
        return best

    # ──────────────────────────────────────────────────────────────────────
    def detect_state(self, image):
        """
        Retorna (state, red_area, yellow_area, green_area).
        state: 'red', 'yellow', 'green', 'none'

        Regla verde→rojo:
            Si last_confirmed='green' y smoothed='red' → devuelve 'yellow'
        """
        roi, _ = self._crop_roi(image)
        blurred = cv.GaussianBlur(roi, (5, 5), 2)
        hsv     = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask_red = (cv.inRange(hsv, self.lower_red1, self.upper_red1) |
                    cv.inRange(hsv, self.lower_red2, self.upper_red2))
        mask_yel = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_grn = cv.inRange(hsv, self.lower_green,  self.upper_green)

        mask_red = self._morphology(mask_red)
        mask_yel = self._morphology(mask_yel)
        mask_grn = self._morphology(mask_grn)

        red_area = self._best_blob(mask_red)
        yel_area = self._best_blob(mask_yel)
        grn_area = self._best_blob(mask_grn)

        # Estado raw
        raw = "none"
        best = max(red_area, yel_area, grn_area)
        if best > 0:
            if red_area == best:
                raw = "red"
            elif yel_area == best:
                raw = "yellow"
            else:
                raw = "green"

        # Suavizado temporal
        self._state_buffer.pop(0)
        self._state_buffer.append(raw)
        smoothed = max(set(self._state_buffer), key=self._state_buffer.count)

        # ── Regla: verde → rojo NO existe → tratar como yellow ───────────
        final = smoothed
        if self._last_confirmed == "green" and smoothed == "red":
            final = "yellow"

        if final != self._last_confirmed:
            self._last_confirmed = final

        return final, red_area, yel_area, grn_area

    # ──────────────────────────────────────────────────────────────────────
    def draw_status(self, image, state, waiting):
        h, w = image.shape[:2]

        # Dibujar ROI del semáforo
        y1 = int(self.roi_top    * h)
        y2 = int(self.roi_bottom * h)
        x1 = int(self.roi_left   * w)
        x2 = int(self.roi_right  * w)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 1)

        color = {"red": (0,0,255), "yellow": (0,165,255),
                 "green": (0,255,0), "none": (255,255,255)}.get(state, (255,255,255))

        cv.putText(image, f"TL: {state.upper()}", (x1+5, y1+30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv.putText(image, f"last={self._last_confirmed}",
                   (x1+5, y1+55), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                   (200,200,200), 1)
        cv.putText(image, f"buf={self._state_buffer}",
                   (x1+5, y1+72), cv.FONT_HERSHEY_SIMPLEX, 0.35,
                   (180,180,180), 1)
        if waiting:
            cv.putText(image, "WAIT GREEN", (w//2-80, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return image