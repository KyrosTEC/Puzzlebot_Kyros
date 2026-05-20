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
    Detector de semáforo con regla de transición verde→amarillo.

    Regla clave:
        Si el semáforo estaba en VERDE y detecta AMARILLO, reporta AMARILLO
        (velocidad lenta), NO rojo. La transición verde→rojo directa no existe:
        si pasó por amarillo, ya fue tratado como amarillo.

        Esto se controla con `_last_confirmed_state`:
        - Si last=green y raw=red  → se fuerza a yellow (aún ajustando máscara)
        - Si last=green y raw=yellow → yellow normal
        - Si last=yellow y raw=red → red normal (ya pasó por amarillo)
    """

    def __init__(self):

        # ── HSV ranges ────────────────────────────────────────────────────
        self.lower_red1 = np.array([0,   120, 70])
        self.upper_red1 = np.array([10,  255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_yellow = np.array([18, 120, 100])
        self.upper_yellow = np.array([32, 255, 255])

        self.lower_green = np.array([40, 80, 60])
        self.upper_green = np.array([80, 255, 255])

        # ── Morphology kernel ─────────────────────────────────────────────
        self.kernel = np.ones((5, 5), np.uint8)

        # ── Detection thresholds ──────────────────────────────────────────
        self.min_blob_area    = 300
        self.min_circularity  = 0.4

        # ── ROI ───────────────────────────────────────────────────────────
        self.roi_top    = 0.05
        self.roi_bottom = 0.65
        self.roi_left   = 0.20
        self.roi_right  = 0.80

        # ── Temporal smoothing ────────────────────────────────────────────
        self.buffer_size   = 5
        self._state_buffer = ["none"] * self.buffer_size

        # ── Estado confirmado previo (para la regla verde→amarillo) ───────
        # Solo se actualiza cuando el estado cambia realmente (post-smoothing).
        self._last_confirmed_state = "none"

    # ──────────────────────────────────────────────────────────────────────
    def _apply_morphology(self, mask):
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

    def _largest_circular_blob(self, mask):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        best_area = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_blob_area:
                continue
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity >= self.min_circularity:
                best_area = max(best_area, area)
        return best_area

    # ──────────────────────────────────────────────────────────────────────
    def detect_state(self, image):
        """
        Returns:
            state  (str)  – 'red', 'yellow', 'green', o 'none'
            red_area, yellow_area, green_area (int) – áreas para logging

        Regla verde→amarillo≠rojo:
            Si el último estado confirmado fue 'green' y el estado suavizado
            resulta 'red', se devuelve 'yellow' en su lugar.
            Esto cubre el caso en que la máscara de amarillo aún no está
            bien calibrada y el robot salta de verde a rojo sin pasar
            visualmente por amarillo.
        """
        # 1. Crop ROI
        roi, _ = self._crop_roi(image)

        # 2. Blur
        blurred = cv.GaussianBlur(roi, (5, 5), 2)

        # 3. HSV
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        # 4. Máscaras de color
        mask_red = (
            cv.inRange(hsv, self.lower_red1, self.upper_red1) |
            cv.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        mask_yellow = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green  = cv.inRange(hsv, self.lower_green,  self.upper_green)

        # 5. Morfología
        mask_red    = self._apply_morphology(mask_red)
        mask_yellow = self._apply_morphology(mask_yellow)
        mask_green  = self._apply_morphology(mask_green)

        # 6. Área del blob más grande y circular
        red_area    = self._largest_circular_blob(mask_red)
        yellow_area = self._largest_circular_blob(mask_yellow)
        green_area  = self._largest_circular_blob(mask_green)

        # 7. Estado raw
        raw_state = "none"
        best = max(red_area, yellow_area, green_area)
        if best > 0:
            if red_area == best:
                raw_state = "red"
            elif yellow_area == best:
                raw_state = "yellow"
            else:
                raw_state = "green"

        # 8. Suavizado temporal (majority vote)
        self._state_buffer.pop(0)
        self._state_buffer.append(raw_state)
        smoothed_state = max(set(self._state_buffer),
                             key=self._state_buffer.count)

        # 9. ── REGLA VERDE → AMARILLO ≠ ROJO ──────────────────────────────
        #    Si el último estado confirmado fue verde y ahora detectamos rojo,
        #    lo tratamos como amarillo (transición incompleta / máscara sucia).
        #    Solo se permite ir a rojo desde amarillo o none/rojo.
        final_state = smoothed_state
        if self._last_confirmed_state == "green" and smoothed_state == "red":
            final_state = "yellow"

        # Actualizar último estado confirmado solo si cambia
        if final_state != self._last_confirmed_state:
            self._last_confirmed_state = final_state

        return final_state, red_area, yellow_area, green_area

    # ──────────────────────────────────────────────────────────────────────
    def draw_status(self, image, state, current_goal, total_goals, waiting):
        h, w = image.shape[:2]

        y1 = int(self.roi_top    * h)
        y2 = int(self.roi_bottom * h)
        x1 = int(self.roi_left   * w)
        x2 = int(self.roi_right  * w)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)

        colour_map = {
            "red":    (0, 0, 255),
            "yellow": (0, 255, 255),
            "green":  (0, 255, 0),
            "none":   (255, 255, 255),
        }
        color = colour_map.get(state, (255, 255, 255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (480, 210), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv.putText(image, f"Traffic: {state.upper()}", (20, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv.putText(image, f"Goal: {current_goal}/{total_goals}", (20, 105),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.putText(image, f"Waiting green: {waiting}", (20, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(image,
                   f"Buffer: {self._state_buffer}  last={self._last_confirmed_state}",
                   (20, 190),
                   cv.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        return image