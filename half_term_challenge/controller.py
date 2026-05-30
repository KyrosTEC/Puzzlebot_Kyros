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

# Directory where this file is installed — used to find calibracion_jetson.npz
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


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
        self._integral = float(np.clip(self._integral,
                                       -max_integral, max_integral))
        I = self.Ki * self._integral

        output           = float(np.clip(P + I + D, self.out_min, self.out_max))
        self._prev_error = error
        return output


# ══════════════════════════════════════════════════════════════════════════
class LineFollowerController(Node):
    """
    PuzzleBot — line follower + traffic light with PID controller.

    Camera: wide-angle IMX219 via V4L2 (/dev/video0)
    Calibration: calibracion_jetson.npz (same file used in dataset capture)
      - Keys: K (camera matrix), dist (distortion coefficients)
      - Remap maps precomputed once at startup for fast per-frame undistortion
    """

    # ── Camera resolution ─────────────────────────────────────────────────
    # Must match the resolution used when calibracion_jetson.npz was created.
    CAM_W = 640
    CAM_H = 480

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── Calibration ───────────────────────────────────────────────────
        # Load calibracion_jetson.npz from the same folder as this file.
        # Uses initUndistortRectifyMap + remap — same method as the dataset
        # capture tool (maps computed once, fast lookup every frame).
        self._map1 = None
        self._map2 = None

        calib_path = os.path.join(_PKG_DIR, "calibracion_jetson.npz")
        if os.path.isfile(calib_path):
            data    = np.load(calib_path)
            K       = data['K']       # 3x3 camera matrix
            dist    = data['dist']    # distortion coefficients

            new_K, _ = cv.getOptimalNewCameraMatrix(
                K, dist, (self.CAM_W, self.CAM_H), 0,
                (self.CAM_W, self.CAM_H)
            )
            self._map1, self._map2 = cv.initUndistortRectifyMap(
                K, dist, None, new_K,
                (self.CAM_W, self.CAM_H), cv.CV_16SC2
            )
            self.get_logger().info(
                f"Calibration loaded: {calib_path} — undistortion ENABLED"
            )
        else:
            self.get_logger().warn(
                f"Calibration file not found: {calib_path} — "
                "undistortion DISABLED. "
                "Add calibracion_jetson.npz to the package and rebuild."
            )

        # ── PID angular ───────────────────────────────────────────────────
        # Tuning guide:
        #   OSCILLATES      → lower Kp or raise Kd
        #   SLOW to correct → raise Kp slightly
        #   STEADY DRIFT    → raise Ki in 0.01 steps
        #   NOISY deriv     → lower alpha_d
        self.pid_angular = PIDController(
            Kp=1.1,
            Ki=0.025,
            Kd=0.25,
            output_min=-1.8,
            output_max= 1.8,
            alpha_d=0.2,
        )
        self.pid_dead_zone = 0.03

        # ── Velocidades ───────────────────────────────────────────────────
        self.normal_speed    = 0.08   # m/s on straight
        self.slow_speed      = 0.05   # m/s on yellow light
        self.speed_reduction = 0.55   # adaptive braking on curves

        # ── Control state ─────────────────────────────────────────────────
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0
        self._error_norm  = 0.0
        self._line_lost   = False

        # ── Traffic light state ───────────────────────────────────────────
        self.traffic_state     = "none"
        self.waiting_for_green = False
        self.lock              = threading.Lock()

        # ── Detectors ─────────────────────────────────────────────────────
        self.traffic_detector = TrafficLightDetection()
        self.line_detector    = CenterLineDetector(
            alpha=0.35,
            lost_timeout=2.0,
            roi_top_frac=0.25,
            roi_left_frac=0.15,
            roi_right_frac=0.85,
            lookahead_weight=0.30,
        )

        # ── Camera — same pipeline as dataset capture tool ────────────────
        # Uses nvarguscamerasrc (Argus) at 640x480, exactly like the dataset
        # capture script that successfully opened the wide-angle camera.
        argus_pipeline = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), "
            f"width={self.CAM_W}, height={self.CAM_H}, "
            f"format=NV12, framerate=15/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.CAM_W}, height={self.CAM_H}, "
            f"format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )
        self.get_logger().info(
            f"Opening camera via Argus ({self.CAM_W}x{self.CAM_H})..."
        )
        self.cap = cv.VideoCapture(argus_pipeline, cv.CAP_GSTREAMER)
        if self.cap.isOpened():
            self.get_logger().info("Camera opened successfully!")
        else:
            self.get_logger().error(
                "Cannot open camera! Try: sudo systemctl restart nvargus-daemon"
            )

        # ── Video recording ───────────────────────────────────────────────
        ts         = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.expanduser(f"~/puzzlebot_run_{ts}.mp4")
        fourcc     = cv.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv.VideoWriter(
            video_path, fourcc, 10, (self.CAM_W, self.CAM_H)
        )
        self._video_path = video_path
        self.get_logger().info(f"Recording to: {video_path}")

        self._log_count = 0

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_vel   = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/traffic_light_state', 10)

        # ── Timers ────────────────────────────────────────────────────────
        self.cam_timer  = self.create_timer(0.10, self.camera_callback)
        self.ctrl_timer = self.create_timer(0.02, self.control_callback)

        self.get_logger().info("LineFollowerController started.")

    # ══════════════════════════════════════════════════════════════════════
    def _undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort frame using precomputed remap maps (fast, ~1ms per frame).
        Identical approach to the dataset capture tool (cv2.remap).
        Returns image unchanged if calibration was not loaded.
        """
        if self._map1 is None:
            return image
        return cv.remap(image, self._map1, self._map2, cv.INTER_LINEAR)

    # ══════════════════════════════════════════════════════════════════════
    def camera_callback(self):
        if not self.cap.isOpened():
            return
        ret, image = self.cap.read()
        if not ret:
            return

        # ── Undistort (no-op if calibration not loaded) ───────────────────
        image = self._undistort(image)

        # ── Traffic light detection ───────────────────────────────────────
        tl_state, r_area, y_area, g_area = \
            self.traffic_detector.detect_state(image)

        # ── Line detection ────────────────────────────────────────────────
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
                speed_factor = max(
                    1.0 - abs(error_norm) * self.speed_reduction, 0.30
                )
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
        debug = self._draw(debug, result, error_norm, tl_state, line_lost,
                           Vc, Wc)
        self._video_writer.write(debug)
        cv.imshow('PuzzleBot PID', debug)
        cv.waitKey(1)

        self._log_count += 1
        if self._log_count >= 10:
            self._log_count = 0
            calib = "undistort=ON" if self._map1 is not None else "undistort=OFF"
            self.get_logger().info(
                f"TL={tl_state}({self.traffic_detector.confidence:.0%}) "
                f"| lost={line_lost} "
                f"| err={error_norm:+.3f} "
                f"| Vc={Vc:.3f} Wc={Wc:+.3f} "
                f"| {calib}"
            )

    # ══════════════════════════════════════════════════════════════════════
    def control_callback(self):
        with self.lock:
            tl_state = self.traffic_state
            Vc       = self._cmd_linear
            Wc       = self._cmd_angular

        if tl_state == "red":
            if not self.waiting_for_green:
                self.get_logger().info("RED — stopped, waiting for GREEN...")
                self.pid_angular.reset()
            self.waiting_for_green = True

        if self.waiting_for_green and tl_state == "green":
            self.waiting_for_green = False
            self.get_logger().info("GREEN — resuming!")

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
        cv.rectangle(overlay, (10, 10), (430, 180), (0, 0, 0), -1)
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

        calib_label = "undistort=ON" if self._map1 is not None \
                      else "undistort=OFF"
        cv.putText(image, calib_label,
                   (20, 170), cv.FONT_HERSHEY_SIMPLEX, 0.38,
                   (0, 255, 100) if self._map1 is not None
                   else (80, 80, 80), 1)

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
        self.get_logger().info(f"Video saved: {self._video_path}")
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()