import os
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


class TrafficSignDetection:
    """
    Detector de señales basado en el pipeline de actividad_06:

    1. Segmentar por color (rojo / azul) en HSV
    2. Filtrar contornos por área, aspect ratio y FORMA:
       - Rojo octagonal/cuadrado  → Stop
       - Rojo triangular          → Workers
       - Azul circular            → flecha (izq/der/recto)
    3. Clasificar flecha azul por distribución de píxeles blancos
    4. Validar con Template Matching + SSIM

    Ventajas sobre versión anterior:
    - No confunde la cinta azul del piso (demasiado larga, aspect ratio falla)
    - No confunde señales rojas entre sí (forma octágono vs triángulo)
    - Clasifica flechas sin necesidad de template perfecto
    - Solo busca señales en la mitad derecha + centro del frame
    """

    def __init__(self, templates_path=""):
        self.templates_path = templates_path
        self.templates = self._load_templates()

        # ── HSV ranges ────────────────────────────────────────────────────
        self.lower_red1  = np.array([0,   80, 60])
        self.upper_red1  = np.array([12,  255, 255])
        self.lower_red2  = np.array([165, 80, 60])
        self.upper_red2  = np.array([180, 255, 255])
        self.lower_blue  = np.array([85,  60, 40])
        self.upper_blue  = np.array([135, 255, 255])

        # ── Umbrales de score ─────────────────────────────────────────────
        self.min_score_blue = 0.10   # TM mínimo para azul
        self.min_score_red  = 0.00   # rojo pasa por forma, no necesita TM alto

        # ── Tamaño de template para matching ─────────────────────────────
        self.template_size = (120, 120)

    # ──────────────────────────────────────────────────────────────────────
    def _load_templates(self):
        templates = {}
        file_map = {
            "alto":        "stop",
            "trabajadores":"worker",
            "derecha":     "right",
            "izquierda":   "left",
            "derecho":     "straight",
        }
        for label, basename in file_map.items():
            img = None
            for ext in (".jpeg", ".jpg", ".png"):
                path = os.path.join(self.templates_path, basename + ext)
                img = cv2.imread(path)
                if img is not None:
                    break
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (120, 120))
                templates[label] = gray
            else:
                print(f"[TrafficSign] No se pudo cargar: {basename}")
        return templates

    # ──────────────────────────────────────────────────────────────────────
    def _masks(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        k3  = np.ones((3, 3), np.uint8)
        k5  = np.ones((5, 5), np.uint8)

        mask_red = (cv2.inRange(hsv, self.lower_red1, self.upper_red1) |
                    cv2.inRange(hsv, self.lower_red2, self.upper_red2))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k5)
        mask_red = cv2.dilate(mask_red, k3, iterations=1)

        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, k5)
        mask_blue = cv2.dilate(mask_blue, k3, iterations=1)

        return mask_red, mask_blue

    # ──────────────────────────────────────────────────────────────────────
    def _expand_box(self, x, y, w, h, frame_shape, scale=0.45):
        H, W = frame_shape[:2]
        px = int(w * scale)
        py = int(h * scale)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(W, x + w + px)
        y2 = min(H, y + h + py)
        return x1, y1, x2 - x1, y2 - y1

    # ──────────────────────────────────────────────────────────────────────
    def _get_candidates(self, mask_red, mask_blue, frame_shape, frame_w):
        """
        Retorna lista de (x, y, w, h, color_label).
        Solo busca en la mitad derecha del frame (señales siempre ahí).
        """
        candidates = []
        H, W = frame_shape[:2]

        # ── ROJO ──────────────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue

            # Solo mitad derecha (señales a la derecha, semáforo a la izq)
            cx = x + w // 2
            if cx < frame_w * 0.35:   # ignorar extremo izquierdo
                continue

            aspect = w / float(h)
            if aspect < 0.45 or aspect > 2.2:
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            sides = len(approx)

            x, y, w, h = self._expand_box(x, y, w, h, frame_shape)

            # Stop: octógono/círculo → 6-12 lados, casi cuadrado
            if 6 <= sides <= 12 and 0.75 <= aspect <= 1.35:
                candidates.append((x, y, w, h, "red_stop"))
            # Workers: triángulo → 3-5 lados
            elif 3 <= sides <= 5 and 0.7 <= aspect <= 1.6:
                candidates.append((x, y, w, h, "red_triangle"))
            else:
                candidates.append((x, y, w, h, "red"))

        # ── AZUL ──────────────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 180:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 or h < 15:
                continue

            cx = x + w // 2
            if cx < frame_w * 0.35:
                continue

            aspect = w / float(h)

            # La cinta azul del piso: muy ancha y baja (aspect > 3)
            if aspect > 2.5:
                continue
            if aspect < 0.4 or aspect > 2.2:
                continue

            area_box = w * h
            extent = area / float(area_box)
            if extent < 0.25:
                continue

            x, y, w, h = self._expand_box(x, y, w, h, frame_shape)
            candidates.append((x, y, w, h, "blue"))

        return candidates

    # ──────────────────────────────────────────────────────────────────────
    def _classify_arrow(self, roi):
        """Clasifica flecha azul por distribución de píxeles blancos."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([0, 0, 160]),
                           np.array([180, 80, 255]))
        h, w = mask.shape
        left   = cv2.countNonZero(mask[:, :w // 3])
        center = cv2.countNonZero(mask[:, w // 3: 2 * w // 3])
        right  = cv2.countNonZero(mask[:, 2 * w // 3:])
        top    = cv2.countNonZero(mask[:h // 3, :])
        middle = cv2.countNonZero(mask[h // 3: 2 * h // 3, :])

        # Recto: concentrado arriba y en el centro
        if top > middle * 0.8 and center > left * 0.8 and center > right * 0.8:
            return "derecho"
        if left > right * 1.15:
            return "izquierda"
        if right > left * 1.15:
            return "derecha"
        return None

    # ──────────────────────────────────────────────────────────────────────
    def _classify_roi(self, roi, allowed_labels=None):
        """Template matching + SSIM sobre la ROI."""
        if roi is None or roi.size == 0:
            return None, 0.0, 0.0

        size = self.template_size
        roi_gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), size)

        best_label = None
        best_score = -1.0
        best_tm    = 0.0
        best_ssim  = 0.0

        for label, tmpl in self.templates.items():
            if allowed_labels and label not in allowed_labels:
                continue

            tmpl_r = cv2.resize(tmpl, size)
            tm_res = cv2.matchTemplate(roi_gray, tmpl_r, cv2.TM_CCOEFF_NORMED)
            tm_score = float(tm_res[0][0])

            if SSIM_AVAILABLE:
                ssim_score = float(ssim(roi_gray, tmpl_r, data_range=255))
            else:
                ssim_score = 0.0

            final = tm_score * 0.6 + ssim_score * 0.4
            if final > best_score:
                best_score = final
                best_label = label
                best_tm    = tm_score
                best_ssim  = ssim_score

        return best_label, best_tm, best_ssim

    # ──────────────────────────────────────────────────────────────────────
    def detect_sign(self, frame):
        """
        Retorna (label, score, box) o ("none", 0, None).

        Labels: "Stop", "Workers", "Turn Left", "Turn Right", "Go Straight"
        """
        if frame is None:
            return "none", 0, None

        H, W = frame.shape[:2]
        mask_red, mask_blue = self._masks(frame)
        candidates = self._get_candidates(mask_red, mask_blue, frame.shape, W)

        best_label = "none"
        best_score = -1.0
        best_box   = None

        for (x, y, w, h, color_label) in candidates:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            label    = None
            tm_score = 0.0

            if color_label == "red_stop":
                label    = "alto"
                tm_score = 0.5   # forma confirma, no necesita TM alto

            elif color_label == "red_triangle":
                label    = "trabajadores"
                tm_score = 0.5

            elif color_label == "blue":
                # Primero intentar por distribución de píxeles blancos
                arrow = self._classify_arrow(roi)
                if arrow:
                    label    = arrow
                    tm_score = 0.4
                else:
                    # Fallback: template matching
                    label, tm_score, _ = self._classify_roi(
                        roi, ["derecha", "izquierda", "derecho"]
                    )
                    if tm_score < self.min_score_blue:
                        continue

            else:  # red genérico
                label, tm_score, _ = self._classify_roi(
                    roi, ["alto", "trabajadores"]
                )
                if tm_score < self.min_score_red:
                    continue

            if label is None:
                continue

            if tm_score > best_score:
                best_score = tm_score
                best_label = label
                best_box   = ((x, y), (x + w, y + h))

        # Mapear labels internos → labels del controller
        label_map = {
            "alto":        "Stop",
            "trabajadores":"Workers",
            "derecha":     "Turn Right",
            "izquierda":   "Turn Left",
            "derecho":     "Go Straight",
            "none":        "none",
        }
        return label_map.get(best_label, "none"), best_score, best_box