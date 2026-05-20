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


class LineFollowerController(Node):
    """
    PuzzleBot — seguidor de línea negra + semáforo.
    Señales de tráfico desactivadas.

    Control:
        error_norm = (cx - w/2) / (w/2)
        Wc = -Kw * error_norm
        Vc = constante según semáforo

    Semáforo:
        green/none → normal_speed
        yellow     → slow_speed
        red        → parar, esperar verde
    """

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── Ganancias ─────────────────────────────────────────────────────
        self.Kw     = 1.2   # antes 1.5 — más suave en curvas
        self.Wc_max = 1.8   # antes 2.0

        # ── Velocidades ───────────────────────────────────────────────────
        self.normal_speed = 0.08   # antes 0.15
        self.slow_speed   = 0.05   # antes 0.08

        # ── Comando actual (calculado en cam_timer, publicado en ctrl_timer)
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0
        self._error_norm  = 0.0
        self._line_lost   = False

        # ── Semáforo ──────────────────────────────────────────────────────
        self.traffic_state     = "none"
        self.waiting_for_green = False

        self.lock = threading.Lock()

        # ── Detectores ───────────────────────────────────────────────────
        self.traffic_detector = TrafficLightDetection()
        self.line_detector    = CenterLineDetector()

        # ── Cámara ───────────────────────────────────────────────────────
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

        # ── Grabación ─────────────────────────────────────────────────────
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.expanduser(f"~/puzzlebot_run_{ts}.mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv.VideoWriter(video_path, fourcc, 10, (640, 360))
        self._video_path   = video_path
        self.get_logger().info(f"Recording to: {video_path}")

        # ── Log throttle ──────────────────────────────────────────────────
        self._log_count = 0

        # ── Publicadores ──────────────────────────────────────────────────
        self.pub_vel   = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/traffic_light_state', 10)

        # ── Timers ───────────────────────────────────────────────────────
        self.cam_timer  = self.create_timer(0.1,  self.camera_callback)
        self.ctrl_timer = self.create_timer(0.02, self.control_callback)

        self.get_logger().info("Line Follower Controller Started (semaforo + linea)")

    # ══════════════════════════════════════════════════════════════════════
    def camera_callback(self):
        if not self.cap.isOpened():
            return
        ret, image = self.cap.read()
        if not ret:
            return

        h, w = image.shape[:2]

        # ── Semáforo ──────────────────────────────────────────────────────
        tl_state, r_area, y_area, g_area = self.traffic_detector.detect_state(image)

        # ── Línea ─────────────────────────────────────────────────────────
        result = self.line_detector.detect_center_line(image)
        line_lost = (result is None)

        if line_lost:
            # Sin línea: mantener Wc anterior, reducir Vc
            with self.lock:
                prev_wc = self._cmd_angular
            Vc = self.slow_speed * 0.4
            Wc = prev_wc * 0.5   # reducir giro gradualmente
            error_norm = self._error_norm
        else:
            cx, cy = result
            error_norm = (cx - w / 2.0) / (w / 2.0)
            Wc = float(np.clip(-self.Kw * error_norm, -self.Wc_max, self.Wc_max))
            if tl_state == "yellow":
                Vc = self.slow_speed
            elif tl_state in ("none", "green"):
                Vc = self.normal_speed
            else:
                Vc = 0.0
                Wc = 0.0

        with self.lock:
            self.traffic_state = tl_state
            self._cmd_linear   = Vc
            self._cmd_angular  = Wc
            self._error_norm   = error_norm
            self._line_lost    = line_lost

        # ── Publicar estado semáforo ──────────────────────────────────────
        msg = String()
        msg.data = tl_state
        self.pub_state.publish(msg)

        # ── Debug visual ──────────────────────────────────────────────────
        debug = image.copy()
        debug = self._draw(debug, result, error_norm, tl_state, line_lost)
        self._video_writer.write(debug)
        cv.imshow('PuzzleBot', debug)
        cv.waitKey(1)

        # ── Log ~1 Hz ─────────────────────────────────────────────────────
        self._log_count += 1
        if self._log_count >= 10:
            self._log_count = 0
            self.get_logger().info(
                f"TL={tl_state} | lost={line_lost} err={error_norm:+.2f} "
                f"Vc={Vc:.2f} Wc={Wc:.2f}"
            )

    # ══════════════════════════════════════════════════════════════════════
    def control_callback(self):
        with self.lock:
            tl_state  = self.traffic_state
            Vc        = self._cmd_linear
            Wc        = self._cmd_angular
            line_lost = self._line_lost

        # ── Lógica semáforo ───────────────────────────────────────────────
        if tl_state == "red":
            if not self.waiting_for_green:
                self.get_logger().info("RED — esperando VERDE...")
            self.waiting_for_green = True

        if self.waiting_for_green and tl_state == "green":
            self.waiting_for_green = False
            self.get_logger().info("GREEN — reanudando!")

        if self.waiting_for_green:
            self._publish(0.0, 0.0)
            return

        self._publish(Vc, Wc)

    # ══════════════════════════════════════════════════════════════════════
    def _publish(self, linear, angular):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.pub_vel.publish(cmd)

    def _draw(self, image, result, error_norm, tl_state, line_lost):
        h, w = image.shape[:2]

        # Línea central de referencia
        cv.line(image, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Punto y ROI del line detector
        self.line_detector.draw_debug(image, result)

        # Punto detectado
        if result is not None:
            cx, cy = result
            color = (0, 255, 0) if not line_lost else (0, 0, 255)
            cv.circle(image, (cx, cy), 8, color, -1)
            cv.line(image, (w // 2, cy), (cx, cy), color, 2)

        # HUD
        tl_color = {
            "red": (0,0,255), "yellow": (0,255,255),
            "green": (0,255,0), "none": (255,255,255)
        }.get(tl_state, (255,255,255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (400, 120), (0,0,0), -1)
        cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv.putText(image, f"Traffic: {tl_state.upper()}", (20, 48),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, tl_color, 2)
        cv.putText(image, f"Err: {error_norm:+.2f}  Wait: {self.waiting_for_green}",
                   (20, 85), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
        if line_lost:
            cv.putText(image, "LINE LOST", (w//2 - 90, h//2),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

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


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Detenido por el usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()