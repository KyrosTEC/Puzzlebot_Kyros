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

    def __init__(self):

        # ── Rangos HSV ────────────────────────────────────────────────────
        # Rojo: hue 0-10 y 168-180 (sin solaparse con amarillo)
        self.lower_red1 = np.array([0,   140, 80])
        self.upper_red1 = np.array([10,  255, 255])
        self.lower_red2 = np.array([168, 140, 80])
        self.upper_red2 = np.array([180, 255, 255])

        # Amarillo + Ámbar + Naranja: hue 11-40
        # Empieza en 11 para NO solaparse con rojo (0-10)
        self.lower_yellow = np.array([11, 80, 60])
        self.upper_yellow = np.array([40, 255, 255])

        # Verde
        self.lower_green = np.array([42, 90, 70])
        self.upper_green = np.array([82, 255, 255])

        # ── Morfología ────────────────────────────────────────────────────
        self.kernel = np.ones((5, 5), np.uint8)

        # ── Umbrales de detección ─────────────────────────────────────────
        self.min_blob_area   = 400
        self.min_circularity = 0.35

        # ── ROI ───────────────────────────────────────────────────────────
        self.roi_top    = 0.03
        self.roi_bottom = 0.60
        self.roi_left   = 0.15
        self.roi_right  = 0.85

        # ── Suavizado temporal ────────────────────────────────────────────
        self.buffer_size   = 7
        self._state_buffer = ["none"] * self.buffer_size

        # ── Último estado confirmado (regla verde→amarillo) ───────────────
        self._last_confirmed_state = "none"
        self._confidence = 0.0

    # ─────────────────────────────────────────────────────────────────────
    @property
    def confidence(self) -> float:
        return self._confidence

    def _apply_morphology(self, mask):
        mask = cv.erode(mask,  self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        return mask

    def _crop_roi(self, image):
        h, w   = image.shape[:2]
        y1, y2 = int(self.roi_top * h),  int(self.roi_bottom * h)
        x1, x2 = int(self.roi_left * w), int(self.roi_right * w)
        return image[y1:y2, x1:x2], (x1, y1)

    def _largest_circular_blob(self, mask):
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
    def detect_state(self, image):
        """
        Retorna (state, red_area, yellow_area, green_area).

        Regla verde → amarillo ≠ rojo:
            Si last_confirmed='green' y smoothed='red' → devuelve 'yellow'.
            Esto cubre el ámbar del LED que a veces se lee como rojo.
        """
        roi, _ = self._crop_roi(image)
        blurred = cv.GaussianBlur(roi, (5, 5), 2)
        hsv     = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask_red = (
            cv.inRange(hsv, self.lower_red1, self.upper_red1) |
            cv.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        mask_yellow = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green  = cv.inRange(hsv, self.lower_green,  self.upper_green)

        mask_red    = self._apply_morphology(mask_red)
        mask_yellow = self._apply_morphology(mask_yellow)
        mask_green  = self._apply_morphology(mask_green)

        red_area    = self._largest_circular_blob(mask_red)
        yellow_area = self._largest_circular_blob(mask_yellow)
        green_area  = self._largest_circular_blob(mask_green)

        best = max(red_area, yellow_area, green_area)
        if best == 0:
            raw_state = "none"
        elif red_area == best:
            raw_state = "red"
        elif yellow_area == best:
            raw_state = "yellow"
        else:
            raw_state = "green"

        self._state_buffer.pop(0)
        self._state_buffer.append(raw_state)
        smoothed = max(set(self._state_buffer),
                       key=self._state_buffer.count)
        self._confidence = self._state_buffer.count(smoothed) / self.buffer_size

        # Regla verde → rojo NO existe → tratar como yellow
        final = smoothed
        if self._last_confirmed_state == "green" and smoothed == "red":
            final = "yellow"

        if final != self._last_confirmed_state:
            self._last_confirmed_state = final

        return final, red_area, yellow_area, green_area

    # ─────────────────────────────────────────────────────────────────────
    def draw_status(self, image, state,
                    current_goal=0, total_goals=0, waiting=False):
        h, w = image.shape[:2]

        y1 = int(self.roi_top    * h);  y2 = int(self.roi_bottom * h)
        x1 = int(self.roi_left   * w);  x2 = int(self.roi_right  * w)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)

        colour_map = {
            "red":    (0, 0, 255),
            "yellow": (0, 200, 255),   # naranja en pantalla para distinguir del amarillo puro
            "green":  (0, 255, 0),
            "none":   (255, 255, 255),
        }
        color = colour_map.get(state, (255, 255, 255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (500, 220), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv.putText(image, f"Traffic: {state.upper()}", (20, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv.putText(image, f"Confidence: {self._confidence:.0%}", (20, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv.putText(image, f"Waiting green: {waiting}", (20, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv.putText(image,
                   f"buf={self._state_buffer}",
                   (20, 148), cv.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        cv.putText(image,
                   f"last={self._last_confirmed_state}",
                   (20, 165), cv.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        return image