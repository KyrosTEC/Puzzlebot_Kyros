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
    def __init__(self,
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 output_min: float,
                 output_max: float,
                 alpha_d: float = 0.25):
        self.Kp      = Kp
        self.Ki      = Ki
        self.Kd      = Kd
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
        now = time.time()
        if self._prev_time is None:
            dt = 0.02
        else:
            dt = max(now - self._prev_time, 1e-4)
        self._prev_time = now

        if abs(error) < dead_zone:
            error = 0.0

        P     = self.Kp * error
        raw_d = (error - self._prev_error) / dt
        self._d_filtered = (self.alpha_d * raw_d +
                            (1 - self.alpha_d) * self._d_filtered)
        D = self.Kd * self._d_filtered

        output_pd         = P + D
        output_pd_clamped = float(np.clip(output_pd, self.out_min, self.out_max))
        saturated         = (output_pd != output_pd_clamped)
        if not saturated or (error * self._integral < 0):
            self._integral += error * dt

        max_integral   = 0.5 / max(self.Ki, 1e-6)
        self._integral = float(np.clip(self._integral, -max_integral, max_integral))
        I = self.Ki * self._integral

        output           = float(np.clip(P + I + D, self.out_min, self.out_max))
        self._prev_error = error
        return output


# ══════════════════════════════════════════════════════════════════════════
class LineFollowerController(Node):

    # ── Camera mode ───────────────────────────────────────────────────────
    # "csi"  → original CSI camera via Argus/GStreamer
    # "v4l2" → wide-angle IMX219 via V4L2 (/dev/video0)
    CAMERA_MODE = "v4l2"   # ← change to "v4l2" for wide-angle

    # ── Calibration file ──────────────────────────────────────────────────
    # Path to the .npz file produced by calibrate.py.
    # Set to None to disable undistortion (e.g. when using the CSI lens).
    # When using the wide-angle lens, set to the path of your calibration file:
    if CAMERA_MODE == "v4l2":
        CALIB_FILE = "./calibracion_jetson.npz"
    else:
        CALIB_FILE = None

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── Load calibration (if available) ───────────────────────────────
        self._camera_matrix     = None
        self._dist_coeffs       = None
        self._new_camera_matrix = None

        if self.CALIB_FILE is not None:
            if os.path.isfile(self.CALIB_FILE):
                data = np.load(self.CALIB_FILE)
                self._camera_matrix     = data['camera_matrix']
                self._dist_coeffs       = data['dist_coeffs']
                self._new_camera_matrix = data['new_camera_matrix']
                self.get_logger().info(
                    f"Calibration loaded from {self.CALIB_FILE} — "
                    "undistortion ENABLED"
                )
            else:
                self.get_logger().warn(
                    f"Calibration file not found: {self.CALIB_FILE} — "
                    "undistortion DISABLED"
                )

        # ── PID angular ───────────────────────────────────────────────────
        # Tuning guide:
        #   OSCILLATES          → lower Kp or raise Kd
        #   SLOW to correct     → raise Kp slightly
        #   STEADY DRIFT        → raise Ki in 0.01 steps (keep it small)
        #   NOISY derivative    → lower alpha_d
        self.pid_angular = PIDController(
            Kp=1.0,
            Ki=0.025,    # keep OFF until Kp/Kd are stable
            Kd=0.25,
            output_min=-1.8,
            output_max= 1.8,
            alpha_d=0.15,
        )
        self.pid_dead_zone = 0.04

        # ── Velocidades ───────────────────────────────────────────────────
        self.normal_speed    = 0.08
        self.slow_speed      = 0.05
        self.speed_reduction = 0.55

        # ── Estado de control ─────────────────────────────────────────────
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0
        self._error_norm  = 0.0
        self._line_lost   = False

        # ── Semáforo ──────────────────────────────────────────────────────
        self.traffic_state     = "none"
        self.waiting_for_green = False
        self.lock              = threading.Lock()

        # ── Detectores ────────────────────────────────────────────────────
        self.traffic_detector = TrafficLightDetection()
        self.line_detector    = CenterLineDetector(
            alpha=0.35,
            lost_timeout=2.0,
            roi_top_frac=0.40,
        )

        # ── Cámara ────────────────────────────────────────────────────────
        if self.CAMERA_MODE == "v4l2":
            self.get_logger().info("Opening camera via V4L2 (/dev/video0)...")
            self.cap = cv.VideoCapture(0, cv.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
                self.cap.set(cv.CAP_PROP_FPS,           15)
                self.cap.set(cv.CAP_PROP_BUFFERSIZE,     1)
        else:
            self.get_logger().info("Opening camera via GStreamer/Argus (CSI)...")
            self.cap = cv.VideoCapture(
                gstreamer_pipeline(
                    sensor_id=0,
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
            self.get_logger().error(
                f"Cannot open camera! (mode={self.CAMERA_MODE})"
            )
        else:
            self.get_logger().info(
                f"Camera opened! (mode={self.CAMERA_MODE})"
            )

        # ── Grabación de vídeo ────────────────────────────────────────────
        ts         = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.expanduser(f"~/puzzlebot_run_{ts}.mp4")
        fourcc     = cv.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv.VideoWriter(video_path, fourcc, 10, (640, 360))
        self._video_path   = video_path
        self.get_logger().info(f"Recording to: {video_path}")

        self._log_count = 0

        # ── Publicadores ──────────────────────────────────────────────────
        self.pub_vel   = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/traffic_light_state', 10)

        # ── Timers ────────────────────────────────────────────────────────
        self.cam_timer  = self.create_timer(0.10, self.camera_callback)
        self.ctrl_timer = self.create_timer(0.02, self.control_callback)

        self.get_logger().info("LineFollowerController iniciado.")

    # ══════════════════════════════════════════════════════════════════════
    def _undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lens undistortion if calibration data is loaded.
        Uses the optimal new camera matrix so there are no black borders.
        Returns the original image unchanged if no calibration is loaded.
        """
        if self._camera_matrix is None:
            return image
        return cv.undistort(image,
                            self._camera_matrix,
                            self._dist_coeffs,
                            None,
                            self._new_camera_matrix)

    # ══════════════════════════════════════════════════════════════════════
    def camera_callback(self):
        if not self.cap.isOpened():
            return
        ret, image = self.cap.read()
        if not ret:
            return

        # ── Undistort first (no-op if CALIB_FILE is None) ─────────────────
        image = self._undistort(image)

        # ── Semáforo ──────────────────────────────────────────────────────
        tl_state, r_area, y_area, g_area = \
            self.traffic_detector.detect_state(image)

        # ── Detección de línea ────────────────────────────────────────────
        result    = self.line_detector.detect_center_line(image)
        line_lost = (result is None)

        if line_lost:
            with self.lock:
                prev_wc = self._cmd_angular
            Vc         = self.slow_speed * 0.4
            Wc         = prev_wc * 0.5
            error_norm = self._error_norm
        else:
            error_norm = self.line_detector.smooth_error
            Wc         = self.pid_angular.compute(-error_norm,
                                                  dead_zone=self.pid_dead_zone)
            if tl_state == "yellow":
                Vc = self.slow_speed
            elif tl_state in ("none", "green"):
                speed_factor = max(1.0 - abs(error_norm) * self.speed_reduction,
                                   0.30)
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

        msg      = String()
        msg.data = tl_state
        self.pub_state.publish(msg)

        debug = image.copy()
        debug = self._draw(debug, result, error_norm, tl_state, line_lost, Vc, Wc)
        self._video_writer.write(debug)
        cv.imshow('PuzzleBot PID', debug)
        cv.waitKey(1)

        self._log_count += 1
        if self._log_count >= 10:
            self._log_count = 0
            calib_str = "calib=ON" if self._camera_matrix is not None else "calib=OFF"
            self.get_logger().info(
                f"TL={tl_state}({self.traffic_detector.confidence:.0%}) "
                f"| lost={line_lost} "
                f"| err={error_norm:+.3f} "
                f"| Vc={Vc:.3f} Wc={Wc:+.3f} "
                f"| {calib_str}"
            )

    # ══════════════════════════════════════════════════════════════════════
    def control_callback(self):
        with self.lock:
            tl_state = self.traffic_state
            Vc       = self._cmd_linear
            Wc       = self._cmd_angular

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
        cmd           = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.pub_vel.publish(cmd)

    # ══════════════════════════════════════════════════════════════════════
    def _draw(self, image, result, error_norm, tl_state,
              line_lost, Vc, Wc) -> np.ndarray:
        h, w = image.shape[:2]

        cv.line(image, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        self.line_detector.draw_debug(image, result)

        tl_color = {
            "red":    (0, 0, 255),
            "yellow": (0, 255, 255),
            "green":  (0, 255, 0),
            "none":   (255, 255, 255),
        }.get(tl_state, (255, 255, 255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (430, 175), (0, 0, 0), -1)
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

        calib_label = "undistort=ON" if self._camera_matrix is not None \
                      else "undistort=OFF"
        cv.putText(image, calib_label,
                   (20, 168), cv.FONT_HERSHEY_SIMPLEX, 0.38,
                   (0, 255, 100) if self._camera_matrix is not None
                   else (100, 100, 100), 1)

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