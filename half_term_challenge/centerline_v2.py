"""
centerline_v2.py
================
Detección del centro del carril — v2.

Fusiona:
  - Pipeline de preprocesamiento de pista_detection v4
    (Invert → Levels → CLAHE → Threshold → Morfología → ROI)
  - Lógica de scoring / tracking temporal de centerline.py
    (memoria, suavizado, timeout, predicción de movimiento)

Interfaz idéntica a CenterLineDetector original:
  detector = CenterLineDetectorV2()
  result   = detector.detect_center_line(image)   # (cx, cy) | None
  image    = detector.draw_debug(image, result)
  detector.reset_memory()

Para usar en controller.py, solo cambia el import:
  from half_term_challenge.centerline_v2 import CenterLineDetectorV2 as CenterLineDetector
"""

import time
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG EMBEBIDA  (defaults de pista_detection v4)
#  Si tienes un config_tuned.json, puedes pasarlo al constructor.
# ─────────────────────────────────────────────────────────────────────────────

class PipelineConfig:
    """Parámetros del pipeline de visión — valores por defecto razonables."""

    def __init__(self):
        # Frame
        self.frame_width  = 640
        self.frame_height = 360

        # Invert
        self.use_invert = True

        # Levels (estilo Photoshop)
        self.use_levels   = True
        self.levels_black = 50
        self.levels_white = 200
        self.levels_gamma = 1.0

        # CLAHE
        self.use_clahe  = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8

        # Blur
        self.blur_kernel = 5          # debe ser impar

        # Threshold
        self.threshold_mode       = 'fixed'   # 'fixed' | 'adaptive'
        self.fixed_threshold      = 127
        self.adaptive_block_size  = 11
        self.adaptive_C           = 2

        # Morfología
        self.morph_kernel     = 5     # debe ser impar
        self.morph_open_iter  = 1
        self.morph_close_iter = 1

        # Área mínima de contorno
        self.min_contour_area = 300

        # ROI  (polígono normalizado 0-1)
        self.roi_tl_x = 0.1     # top-left  x
        self.roi_tl_y = 0.5     # top-left  y  (mitad superior recortada)
        self.roi_tr_x = 0.9     # top-right x
        self.roi_tr_y = 0.5     # top-right y
        self.roi_bl_x = 0.0     # bottom-left  x
        self.roi_bl_y = 1.0     # bottom-left  y
        self.roi_br_x = 1.0     # bottom-right x
        self.roi_br_y = 1.0     # bottom-right y

        # Canny refinamiento (opcional)
        self.use_canny_refine = False
        self.canny_low        = 50
        self.canny_high       = 150
        self.edge_thickness   = 1

    @staticmethod
    def from_json(path: str) -> "PipelineConfig":
        """Carga parámetros desde un JSON (compatible con config_tuned.json)."""
        import json
        cfg = PipelineConfig()
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES DEL PIPELINE  (de pista_detection v4)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_levels(img, black, white, gamma):
    """Ajuste de Levels tipo Photoshop: mapea [black,white]→[0,255] con gamma."""
    if black >= white:
        white = black + 1
    img_f = img.astype(np.float32)
    img_f = (img_f - black) / (white - black)
    np.clip(img_f, 0, 1, out=img_f)
    if gamma != 1.0 and gamma > 0:
        np.power(img_f, gamma, out=img_f)
    return (img_f * 255).astype(np.uint8)


def _preprocess(frame, cfg):
    """Gray → Invert → Levels → CLAHE → Blur  (devuelve solo el blur final)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invertir: líneas oscuras → blancas
    work = cv2.bitwise_not(gray) if cfg.use_invert else gray

    # Levels
    if cfg.use_levels:
        work = _apply_levels(work, cfg.levels_black,
                             cfg.levels_white, cfg.levels_gamma)

    # CLAHE
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )
        work = clahe.apply(work)

    # Blur
    k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
    blur = cv2.GaussianBlur(work, (k, k), 0)
    return blur


def _segment(blur, cfg):
    """Threshold binario (fixed o adaptive)."""
    if cfg.threshold_mode == 'adaptive':
        block = cfg.adaptive_block_size
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, -cfg.adaptive_C
        )
    else:
        _, binary = cv2.threshold(
            blur, cfg.fixed_threshold, 255, cv2.THRESH_BINARY
        )
        return binary


def _morphological_cleanup(binary, cfg):
    k = cfg.morph_kernel if cfg.morph_kernel % 2 == 1 else cfg.morph_kernel + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if cfg.morph_open_iter > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                  iterations=cfg.morph_open_iter)
    if cfg.morph_close_iter > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=cfg.morph_close_iter)
    return binary


def _apply_roi(binary, cfg):
    h, w = binary.shape[:2]
    mask = np.zeros_like(binary)
    pts = np.array([[
        (int(w * cfg.roi_bl_x), int(h * cfg.roi_bl_y)),
        (int(w * cfg.roi_tl_x), int(h * cfg.roi_tl_y)),
        (int(w * cfg.roi_tr_x), int(h * cfg.roi_tr_y)),
        (int(w * cfg.roi_br_x), int(h * cfg.roi_br_y)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(binary, mask), pts


def _filter_small_contours(binary, min_area):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary)
    kept = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)
            kept.append(cnt)
    return filtered, kept


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTOR  v2
# ─────────────────────────────────────────────────────────────────────────────

class CenterLineDetectorV2:
    """
    Drop-in replacement de CenterLineDetector con pipeline mejorado.

    Parámetros
    ----------
    config : PipelineConfig | str | None
        - PipelineConfig → lo usa directo.
        - str            → ruta a JSON, lo carga con PipelineConfig.from_json().
        - None           → usa PipelineConfig() con defaults.
    """

    def __init__(self, config=None):
        # ── Config del pipeline ───────────────────────────────────────────
        if config is None:
            self.cfg = PipelineConfig()
        elif isinstance(config, str):
            self.cfg = PipelineConfig.from_json(config)
        else:
            self.cfg = config

        # ── Estado de tracking (de centerline.py original) ────────────────
        self.last_x   = None
        self.prev_x   = None
        self.smooth_x = None
        self.alpha    = 0.3

        self._last_detect_time = None
        self.lost_timeout      = 2.0

        # ── Debug info (para draw_debug) ──────────────────────────────────
        self._roi_pts      = None
        self._binary_debug = None

    # ──────────────────────────────────────────────────────────────────────
    def reset_memory(self):
        """Llamar después de un giro para no seguir el borde equivocado."""
        self.last_x   = None
        self.prev_x   = None
        self.smooth_x = None
        self._last_detect_time = None

    # ──────────────────────────────────────────────────────────────────────
    def detect_center_line(self, image):
        """
        Detecta el centro de la línea del carril.

        Returns
        -------
        (cx, cy) : tuple[int,int]  — coordenadas en la imagen original
        None                       — línea perdida (timeout)
        """
        h, w = image.shape[:2]
        cfg   = self.cfg

        # ═══ PIPELINE de pista_detection v4 ═══════════════════════════════
        blur    = _preprocess(image, cfg)
        binary  = _segment(blur, cfg)
        clean   = _morphological_cleanup(binary, cfg)
        masked, roi_pts = _apply_roi(clean, cfg)
        filtered, contours = _filter_small_contours(masked, cfg.min_contour_area)

        # Guardar para debug
        self._roi_pts      = roi_pts
        self._binary_debug = filtered

        # ═══ SCORING de centerline.py original ════════════════════════════
        center_x   = w // 2
        best_score = float('inf')
        best_candidate = (0, 0)

        # Referencia para continuidad
        if self.last_x is None:
            reference_x = center_x
        else:
            if self.prev_x is not None:
                delta = self.last_x - self.prev_x
                delta = max(min(delta, 20), -20)
                reference_x = int(self.last_x + delta)
            else:
                reference_x = self.last_x

        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Punto más bajo del contorno
            bottom_point = tuple(cnt[cnt[:, :, 1].argmax()][0])
            cx_bottom    = bottom_point[0]

            # ── Score multi-factor ────────────────────────────────────────
            dist_center   = abs(cx - center_x)
            height_weight = -cv2.boundingRect(cnt)[3]
            area_weight   = -area / 100.0
            score         = dist_center + height_weight + area_weight

            # Penalizar saltos respecto a la referencia
            if self.last_x is not None:
                motion      = abs(self.last_x - self.prev_x) if self.prev_x is not None else 0
                jump_weight = 0.8 if motion > 25 else 2.0
                jump        = abs(cx - reference_x)
                score      += jump * jump_weight

            # Sesgo de curvatura
            curve_bias = abs(cx_bottom - center_x)
            score     += curve_bias * 2.0

            # Filtro de primera detección: no aceptar cosas muy lejanas
            if self.last_x is None and abs(cx - center_x) > 80:
                continue

            # Penalizar contornos demasiado anchos (probablemente no es línea)
            bbox_w = cv2.boundingRect(cnt)[2]
            if bbox_w > 60:
                score += 200

            if score < best_score:
                best_score     = score
                best_candidate = (cx, cy)

        # ═══ RESULTADO ════════════════════════════════════════════════════
        if best_candidate == (0, 0):
            now = time.time()
            if (self._last_detect_time is None or
                    now - self._last_detect_time > self.lost_timeout):
                return None
            # Dentro del timeout: mantener última posición
            final_x = self.last_x if self.last_x is not None else center_x
            return (final_x, int(0.9 * h))

        # Candidato encontrado — actualizar tracking
        self._last_detect_time = time.time()
        self.prev_x = self.last_x
        self.last_x = best_candidate[0]
        return best_candidate

    # ──────────────────────────────────────────────────────────────────────
    def draw_debug(self, image, result):
        """
        Dibuja sobre `image`:
          - ROI del pipeline (amarillo)
          - Máscara binaria semitransparente (verde)
          - Punto detectado y error (verde/rojo)
          - Label "V2" para distinguir del detector original
        """
        h, w = image.shape[:2]

        # ROI polygon
        if self._roi_pts is not None:
            cv2.polylines(image, [self._roi_pts], True, (0, 255, 255), 2)

        # Overlay de la máscara binaria
        if self._binary_debug is not None:
            lane_overlay = np.zeros_like(image)
            lane_overlay[self._binary_debug > 0] = (0, 200, 0)
            cv2.addWeighted(image, 1.0, lane_overlay, 0.25, 0, image)

        # Línea central de referencia
        cv2.line(image, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Punto detectado
        if result is not None:
            cx, cy = result
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(image, (w // 2, cy), (cx, cy), (0, 255, 0), 2)

        # Label
        cv2.putText(image, "CenterLine V2", (w - 160, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        return image
