import os
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import cv2 as cv

from half_term_challenge.traffic_light_detection import TrafficLightDetection, gstreamer_pipeline
from half_term_challenge.centerline import CenterLineDetector


class PIDController:
    """
    Controlador PID con anti-windup y derivada filtrada (sobre el error,
    no sobre la salida, para evitar derivative kick).

    u(t) = Kp·e + Ki·∫e dt + Kd·(de/dt)

    Anti-windup: el integrador se congela cuando la salida está saturada
    y el error sigue sumando en la misma dirección.

    Derivada: promediada con EMA (alpha_d) para reducir ruido de alta
    frecuencia causado por la visión.
    """

    def __init__(self,
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 output_min: float,
                 output_max: float,
                 alpha_d: float = 0.25):
        self.Kp  = Kp
        self.Ki  = Ki
        self.Kd  = Kd
        self.out_min = output_min
        self.out_max = output_max
        self.alpha_d = alpha_d

        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self._prev_time  = None

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self._prev_time  = None

    def compute(self, error: float, dead_zone: float = 0.0) -> float:
        """
        dead_zone: si |error| < dead_zone se trata como cero.
                   Elimina correcciones innecesarias en rectas.
        """
        now = time.time()

        if self._prev_time is None:
            dt = 0.02
        else:
            dt = now - self._prev_time
            dt = max(dt, 1e-4)
        self._prev_time = now

        # ── Zona muerta ────────────────────────────────────────────────
        if abs(error) < dead_zone:
            error = 0.0

        # ── Término proporcional ───────────────────────────────────────
        P = self.Kp * error

        # ── Término derivativo (EMA sobre de/dt) ──────────────────────
        # alpha_d bajo = más suavizado → menos sensible al ruido de visión
        raw_d = (error - self._prev_error) / dt
        self._d_filtered = (self.alpha_d * raw_d +
                            (1 - self.alpha_d) * self._d_filtered)
        D = self.Kd * self._d_filtered

        # ── Anti-windup ────────────────────────────────────────────────
        output_pd = P + D
        output_pd_clamped = float(np.clip(output_pd, self.out_min, self.out_max))
        saturated = (output_pd != output_pd_clamped)
        if not saturated or (error * self._integral < 0):
            self._integral += error * dt

        max_integral = 0.5 / max(self.Ki, 1e-6)
        self._integral = float(np.clip(self._integral, -max_integral, max_integral))
        I = self.Ki * self._integral

        output = float(np.clip(P + I + D, self.out_min, self.out_max))
        self._prev_error = error
        return output


# ══════════════════════════════════════════════════════════════════════════
class LineFollowerController(Node):
    """
    PuzzleBot — seguidor de línea negra + semáforo con controlador PID.

    Controlador angular PID:
        error_norm = (cx - w/2) / (w/2)    ∈ [-1, 1]
        Wc = PID_angular.compute(error_norm)

    Velocidad lineal adaptativa:
        Vc = normal_speed × (1 - |error_norm| × speed_reduction)
        → El robot frena en curvas pronunciadas.

    Semáforo:
        green / none → velocidad normal
        yellow       → slow_speed (sin freno adaptativo adicional)
        red          → parar, esperar verde
    """

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── PID angular ──────────────────────────────────────────────────
        # Kp: ganancia proporcional (principal respuesta al error lateral)
        # Ki: elimina error estático en rectas con sesgo (cinta pegada al piso)
        # Kd: amortigua oscilaciones en curvas
        # Ajusta estos valores en el laboratorio observando la oscilación.
        # Método empírico sugerido:
        #   1. Ki=0, Kd=0 → sube Kp hasta que oscile → Kp_crítico
        #   2. Kp = 0.6 * Kp_crítico
        #   3. Kd ≈ Kp * 0.05  (muy pequeño al inicio)
        #   4. Ki ≈ Kp * 0.01  (si persiste offset)
        # ── Guía de ajuste (tuning) ───────────────────────────────────────
        # Si OSCILA (sways): baja Kp  o  sube Kd.
        # Si TARDA en corregir: sube Kp ligeramente.
        # Si hay OFFSET permanente en recta: sube Ki apenas (0.01 pasos).
        # Si el DERIVATIVO hace ruido: baja alpha_d (más suavizado).
        # Secuencia recomendada:
        #   1. Ki=0, Kd=0 → ajusta Kp hasta seguir sin oscilar
        #   2. Sube Kd hasta que las oscilaciones desaparezcan
        #   3. Solo si persiste drift lateral añade Ki pequeño
        self.pid_angular = PIDController(
            Kp=1.1,   # ↓ era 1.4 — menos agresivo en la corrección inicial
            Ki=0.025,   # ↓ era 0.05 — apagado hasta eliminar la oscilación
            Kd=0.25,   # ↑ era 0.12 — más amortiguación del sway
            output_min=-1.8,
            output_max= 1.8,
            alpha_d=0.2,  # ↓ era 0.25 — derivada más suavizada (menos ruido)
        )

        # Zona muerta angular: errores < este umbral se tratan como cero.
        # Evita micro-correcciones constantes en rectas largas.
        # Sube si el robot tiembla en recta; baja si pierde curvas suaves.
        self.pid_dead_zone = 0.03

        # ── Velocidades ───────────────────────────────────────────────────
        self.normal_speed    = 0.08   # m/s en recta libre
        self.slow_speed      = 0.05   # m/s con semáforo amarillo
        self.speed_reduction = 0.55   # fracción de frenado en curva
        #  Vc_real = normal_speed * (1 - |error| * speed_reduction)
        #  Con error=1.0 → Vc = normal_speed * (1 - 0.55) = 45% de la vel máx

        # ── Estado de control ─────────────────────────────────────────────
        self._cmd_linear   = 0.0
        self._cmd_angular  = 0.0
        self._error_norm   = 0.0
        self._line_lost    = False

        # ── Semáforo ──────────────────────────────────────────────────────
        self.traffic_state     = "none"
        self.waiting_for_green = False

        self.lock = threading.Lock()

        # ── Detectores ────────────────────────────────────────────────────
        self.traffic_detector = TrafficLightDetection()
        self.line_detector    = CenterLineDetector(
            alpha=0.35,
            lost_timeout=2.0,
            roi_top_frac=0.25,     # ve ~55% superior ignorada → anticipa curvas
            roi_left_frac=0.15,    # ignora 15% del borde izquierdo
            roi_right_frac=0.85,   # ignora 15% del borde derecho
            lookahead_weight=0.30,
        )

        # ── Cámara ────────────────────────────────────────────────────────
        self.cap = cv.VideoCapture(
            gstreamer_pipeline(
                flip_method=0,
                capture_width=1280,
                capture_height=720,
                display_width=640,
                display_height=360,
                framerate=15,
            ),
            cv.CAP_GSTREAMER
        )
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera!")
        else:
            self.get_logger().info("Camera opened successfully!")

        # ── Grabación de vídeo ────────────────────────────────────────────
        ts         = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.expanduser(f"~/puzzlebot_run_{ts}.mp4")
        fourcc     = cv.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv.VideoWriter(video_path, fourcc, 10, (640, 360))
        self._video_path   = video_path
        self.get_logger().info(f"Recording to: {video_path}")

        # ── Log throttle ──────────────────────────────────────────────────
        self._log_count = 0

        # ── Publicadores ──────────────────────────────────────────────────
        self.pub_vel   = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/traffic_light_state', 10)

        # ── Timers ────────────────────────────────────────────────────────
        # cam_timer  : procesamiento de imagen  (100 ms → 10 Hz)
        # ctrl_timer : publicar cmd_vel          (20 ms  → 50 Hz)
        self.cam_timer  = self.create_timer(0.10, self.camera_callback)
        self.ctrl_timer = self.create_timer(0.02, self.control_callback)

        self.get_logger().info(
            "LineFollowerController iniciado (PID angular + semáforo)"
        )

    # ══════════════════════════════════════════════════════════════════════
    def camera_callback(self):
        if not self.cap.isOpened():
            return
        ret, image = self.cap.read()
        if not ret:
            return

        h, w = image.shape[:2]

        # ── Semáforo ──────────────────────────────────────────────────────
        tl_state, r_area, y_area, g_area = \
            self.traffic_detector.detect_state(image)

        # ── Detección de línea ────────────────────────────────────────────
        result    = self.line_detector.detect_center_line(image)
        line_lost = (result is None)

        if line_lost:
            # Línea perdida → mantener giro anterior, reducir velocidad
            with self.lock:
                prev_wc = self._cmd_angular
            Vc = self.slow_speed * 0.4
            Wc = prev_wc * 0.5
            error_norm = self._error_norm
        else:
            # ── Error para el PID (ya suavizado por el detector) ──────────
            error_norm = self.line_detector.smooth_error

            # ── PID angular ───────────────────────────────────────────────
            # Negamos porque error>0 (línea a la derecha) → girar derecha (Wc<0)
            Wc = self.pid_angular.compute(-error_norm,
                                          dead_zone=self.pid_dead_zone)

            # ── Velocidad lineal adaptativa ───────────────────────────────
            if tl_state == "yellow":
                Vc = self.slow_speed
            elif tl_state in ("none", "green"):
                # Frenado suave en curva: mayor error → menor Vc
                speed_factor = 1.0 - abs(error_norm) * self.speed_reduction
                speed_factor = max(speed_factor, 0.30)   # mínimo 30% de Vc
                Vc = self.normal_speed * speed_factor
            else:   # red
                Vc = 0.0
                Wc = 0.0
                self.pid_angular.reset()

        with self.lock:
            self.traffic_state = tl_state
            self._cmd_linear   = Vc
            self._cmd_angular  = Wc
            self._error_norm   = error_norm
            self._line_lost    = line_lost

        # ── Publicar estado semáforo ──────────────────────────────────────
        msg      = String()
        msg.data = tl_state
        self.pub_state.publish(msg)

        # ── Debug visual ──────────────────────────────────────────────────
        debug = image.copy()
        debug = self._draw(debug, result, error_norm, tl_state, line_lost, Vc, Wc)
        self._video_writer.write(debug)
        cv.imshow('PuzzleBot PID', debug)
        cv.waitKey(1)

        # ── Log ~1 Hz ─────────────────────────────────────────────────────
        self._log_count += 1
        if self._log_count >= 10:
            self._log_count = 0
            self.get_logger().info(
                f"TL={tl_state}({self.traffic_detector.confidence:.0%}) "
                f"| lost={line_lost} "
                f"| err={error_norm:+.3f} "
                f"| Vc={Vc:.3f} Wc={Wc:+.3f}"
            )

    # ══════════════════════════════════════════════════════════════════════
    def control_callback(self):
        with self.lock:
            tl_state  = self.traffic_state
            Vc        = self._cmd_linear
            Wc        = self._cmd_angular

        # ── Lógica semáforo ───────────────────────────────────────────────
        if tl_state == "red":
            if not self.waiting_for_green:
                self.get_logger().info("RED — detenido, esperando VERDE...")
                self.pid_angular.reset()
            self.waiting_for_green = True

        if self.waiting_for_green and tl_state == "green":
            self.waiting_for_green = False
            self.get_logger().info("GREEN — reanudando recorrido!")

        if self.waiting_for_green:
            self._publish(0.0, 0.0)
            return

        self._publish(Vc, Wc)

    # ══════════════════════════════════════════════════════════════════════
    def _publish(self, linear: float, angular: float):
        cmd             = Twist()
        cmd.linear.x    = float(linear)
        cmd.angular.z   = float(angular)
        self.pub_vel.publish(cmd)

    # ══════════════════════════════════════════════════════════════════════
    def _draw(self, image, result, error_norm, tl_state,
              line_lost, Vc, Wc) -> np.ndarray:
        h, w = image.shape[:2]

        # Línea central de referencia
        cv.line(image, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Overlay del line detector (ROI + punto)
        self.line_detector.draw_debug(image, result)

        # Colores por estado
        tl_color = {
            "red": (0, 0, 255), "yellow": (0, 255, 255),
            "green": (0, 255, 0), "none": (255, 255, 255)
        }.get(tl_state, (255, 255, 255))

        # HUD
        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (420, 155), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        cv.putText(image,
                   f"Traffic: {tl_state.upper()} "
                   f"({self.traffic_detector.confidence:.0%})",
                   (20, 45), cv.FONT_HERSHEY_SIMPLEX, 0.85, tl_color, 2)
        cv.putText(image,
                   f"Err: {error_norm:+.3f}  Wait: {self.waiting_for_green}",
                   (20, 82), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv.putText(image,
                   f"Vc={Vc:.3f} m/s   Wc={Wc:+.3f} rad/s",
                   (20, 115), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1)
        cv.putText(image,
                   f"PID Kp={self.pid_angular.Kp} "
                   f"Ki={self.pid_angular.Ki} "
                   f"Kd={self.pid_angular.Kd}",
                   (20, 148), cv.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        if line_lost:
            cv.putText(image, "LINE LOST", (w // 2 - 100, h // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return image

    # ══════════════════════════════════════════════════════════════════════
    def destroy_node(self):
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        self.cap.release()
        self._video_writer.release()
        cv.destroyAllWindows()
        self.get_logger().info(f"Video guardado: {self._video_path}")
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Detenido por el usuario.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()