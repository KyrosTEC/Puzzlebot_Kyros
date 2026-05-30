import os
import time
import threading
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
import numpy as np
import cv2 as cv

from half_term_challenge.traffic_light_detection import TrafficLightDetection, gstreamer_pipeline
from half_term_challenge.centerline import CenterLineDetector

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# ── State machine states ──────────────────────────────────────────────────
ST_FOLLOWING   = "FOLLOWING"
ST_SLOW_DOWN   = "SLOW_DOWN"
ST_STOP_SIGN   = "STOP_SIGN"
ST_WAIT_RED    = "WAIT_RED"
ST_TURN_LEFT   = "TURN_LEFT"
ST_TURN_RIGHT  = "TURN_RIGHT"
ST_GO_STRAIGHT = "GO_STRAIGHT"
ST_CROSSWALK   = "CROSSWALK"

# ── Sign detection thresholds ─────────────────────────────────────────────
BBOX_AREA_MIN   = 20000   # px² — ignore tiny detections
SIGNAL_COOLDOWN = 8.0     # seconds before same sign triggers again

# ── State durations ───────────────────────────────────────────────────────
T_STOP_SIGN = 5.0
T_TURN      = 1.5
T_STRAIGHT  = 1.5
T_CROSSWALK = 3.0


# ══════════════════════════════════════════════════════════════════════════
class PIDController:
    def __init__(self, Kp, Ki, Kd, output_min, output_max, alpha_d=0.25):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.out_min = output_min; self.out_max = output_max
        self.alpha_d = alpha_d
        self._integral = 0.0; self._prev_error = 0.0
        self._d_filtered = 0.0; self._prev_time = None

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0
        self._d_filtered = 0.0; self._prev_time = None

    def compute(self, error, dead_zone=0.0):
        now = time.time()
        dt  = 0.02 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time = now
        if abs(error) < dead_zone:
            error = 0.0
        P     = self.Kp * error
        raw_d = (error - self._prev_error) / dt
        self._d_filtered = self.alpha_d * raw_d + (1 - self.alpha_d) * self._d_filtered
        D     = self.Kd * self._d_filtered
        output_pd = P + D
        clamped   = float(np.clip(output_pd, self.out_min, self.out_max))
        if (output_pd == clamped) or (error * self._integral < 0):
            self._integral += error * dt
        max_i = 0.5 / max(self.Ki, 1e-6)
        self._integral = float(np.clip(self._integral, -max_i, max_i))
        self._prev_error = error
        return float(np.clip(P + self.Ki * self._integral + D, self.out_min, self.out_max))


# ══════════════════════════════════════════════════════════════════════════
class LineFollowerController(Node):
    """
    PuzzleBot controller — line following + traffic light (CV) +
    traffic sign detection (YOLO model via /signal_detection/detections).

    State machine:
      FOLLOWING    → normal PID line following
      SLOW_DOWN    → reduced speed (yellow light or roadwork sign)
      STOP_SIGN    → stop 5s then continue
      WAIT_RED     → stop until green light
      TURN_LEFT    → turn left 1.5s
      TURN_RIGHT   → turn right 1.5s
      GO_STRAIGHT  → straight 1.5s ignoring line
      CROSSWALK    → stop 3s at pedestrian crossing
    """

    CAM_W = 640
    CAM_H = 480

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── Calibration ───────────────────────────────────────────────────
        self._map1 = None
        self._map2 = None
        calib_path = os.path.join(_PKG_DIR, "calibracion_jetson.npz")
        if os.path.isfile(calib_path):
            data  = np.load(calib_path)
            K     = data['K']
            dist  = data['dist']
            new_K, _ = cv.getOptimalNewCameraMatrix(
                K, dist, (self.CAM_W, self.CAM_H), 0, (self.CAM_W, self.CAM_H))
            self._map1, self._map2 = cv.initUndistortRectifyMap(
                K, dist, None, new_K, (self.CAM_W, self.CAM_H), cv.CV_16SC2)
            self.get_logger().info(f"Calibration loaded — undistortion ENABLED")
        else:
            self.get_logger().warn("Calibration file not found — undistortion DISABLED")

        # ── PID ───────────────────────────────────────────────────────────
        self.pid_angular = PIDController(
            Kp=1.1, Ki=0.025, Kd=0.25,
            output_min=-1.8, output_max=1.8, alpha_d=0.2)
        self.pid_dead_zone  = 0.03
        self.normal_speed   = 0.08
        self.slow_speed     = 0.05
        self.speed_reduction = 0.55

        # ── State machine ─────────────────────────────────────────────────
        self._state        = ST_FOLLOWING
        self._state_start  = time.time()
        self._last_signal_t: dict = {}   # cooldown per sign class

        # ── Shared data (camera → control thread) ─────────────────────────
        self._cmd_linear   = 0.0
        self._cmd_angular  = 0.0
        self._error_norm   = 0.0
        self._line_lost    = False
        self.traffic_state = "none"
        self.lock          = threading.Lock()

        # ── Latest sign detections for overlay ────────────────────────────
        self._sign_detections: list = []   # list of dicts from YOLO

        # ── Detectors ─────────────────────────────────────────────────────
        self.traffic_detector = TrafficLightDetection()
        self.line_detector    = CenterLineDetector(
            alpha=0.35, lost_timeout=2.0,
            roi_top_frac=0.25,
            roi_left_frac=0.15, roi_right_frac=0.85,
            lookahead_weight=0.30,
        )

        # ── Camera — same Argus pipeline as dataset capture tool ──────────
        argus_pipeline = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), "
            f"width={self.CAM_W}, height={self.CAM_H}, "
            f"format=NV12, framerate=15/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.CAM_W}, height={self.CAM_H}, "
            f"format=BGRx ! videoconvert ! "
            f"video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
        )
        self.get_logger().info(f"Opening camera ({self.CAM_W}x{self.CAM_H})...")
        self.cap = cv.VideoCapture(argus_pipeline, cv.CAP_GSTREAMER)
        if self.cap.isOpened():
            self.get_logger().info("Camera opened!")
        else:
            self.get_logger().error("Cannot open camera!")

        # ── Video recording ───────────────────────────────────────────────
        ts    = time.strftime("%Y%m%d_%H%M%S")
        vpath = os.path.expanduser(f"~/puzzlebot_run_{ts}.mp4")
        self._video_writer = cv.VideoWriter(
            vpath, cv.VideoWriter_fourcc(*'mp4v'), 10, (self.CAM_W, self.CAM_H))
        self._video_path = vpath
        self.get_logger().info(f"Recording to: {vpath}")

        self._log_count = 0

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_vel   = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/puzzlebot/state', 10)

        # ── Subscriber — YOLO sign detections ────────────────────────────
        self.create_subscription(
            String, '/signal_detection/detections',
            self._detections_cb, 10)

        # ── Timers ────────────────────────────────────────────────────────
        self.cam_timer  = self.create_timer(0.10, self.camera_callback)
        self.ctrl_timer = self.create_timer(0.02, self.control_callback)

        self.get_logger().info("LineFollowerController started.")

    # ══════════════════════════════════════════════════════════════════════
    def _undistort(self, image):
        if self._map1 is None:
            return image
        return cv.remap(image, self._map1, self._map2, cv.INTER_LINEAR)

    # ══════════════════════════════════════════════════════════════════════
    def _transition(self, new_state):
        """Move to a new state and reset timers."""
        with self.lock:
            self._state = new_state
        self._state_start = time.time()
        self.pid_angular.reset()
        self.get_logger().info(f"State → {new_state}")

    # ══════════════════════════════════════════════════════════════════════
    def _detections_cb(self, msg):
        """
        Receive YOLO detections from /signal_detection/detections.
        Expected JSON: {"detections": [{"class_name": "stop", "bbox_area": 25000, ...}]}
        Updates the sign overlay and triggers state transitions.
        """
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        detections = data.get("detections", [])

        # Always update overlay list (even if no action taken)
        with self.lock:
            self._sign_detections = detections
            current_state = self._state
            tl_state      = self.traffic_state

        if not detections:
            return

        now   = time.time()
        valid = [d for d in detections if d.get("bbox_area", 0) >= BBOX_AREA_MIN]
        if not valid:
            return

        # Largest bbox = closest sign
        valid.sort(key=lambda d: d.get("bbox_area", 0), reverse=True)
        best  = valid[0]
        clase = best.get("class_name", "")

        # Cooldown
        if now - self._last_signal_t.get(clase, 0) < SIGNAL_COOLDOWN:
            return

        # Traffic light takes priority
        if tl_state in ("red", "yellow"):
            return

        # Don't interrupt active manoeuvres
        if current_state in (ST_TURN_LEFT, ST_TURN_RIGHT, ST_STOP_SIGN,
                              ST_GO_STRAIGHT, ST_WAIT_RED, ST_CROSSWALK):
            return

        new_state = {
            "stop":       ST_STOP_SIGN,
            "turn_left":  ST_TURN_LEFT,
            "turn_right": ST_TURN_RIGHT,
            "go_straight":ST_GO_STRAIGHT,
            "roadwork":   ST_SLOW_DOWN,
            "crosswalk":  ST_CROSSWALK,
        }.get(clase)

        if new_state:
            self._last_signal_t[clase] = now
            self._transition(new_state)

    # ══════════════════════════════════════════════════════════════════════
    def camera_callback(self):
        if not self.cap.isOpened():
            return
        ret, image = self.cap.read()
        if not ret:
            return

        image = self._undistort(image)

        # ── Traffic light (OpenCV) ────────────────────────────────────────
        tl_state, r_area, y_area, g_area = self.traffic_detector.detect_state(image)

        # ── Traffic light → state transitions ────────────────────────────
        with self.lock:
            current_state = self._state

        if tl_state == "red" and current_state not in (ST_WAIT_RED,):
            self._transition(ST_WAIT_RED)
        elif tl_state == "green" and current_state == ST_WAIT_RED:
            self._transition(ST_FOLLOWING)
        elif tl_state == "yellow" and current_state == ST_FOLLOWING:
            self._transition(ST_SLOW_DOWN)
        elif tl_state in ("green", "none") and current_state == ST_SLOW_DOWN:
            self._transition(ST_FOLLOWING)

        # ── Line detection ────────────────────────────────────────────────
        result    = self.line_detector.detect_center_line(image)
        line_lost = (result is None)

        with self.lock:
            state = self._state

        if line_lost:
            with self.lock:
                prev_wc = self._cmd_angular
            Vc = self.slow_speed * 0.4
            Wc = prev_wc * 0.5
            error_norm = self._error_norm
        else:
            error_norm = self.line_detector.smooth_error

            if state == ST_FOLLOWING:
                Wc = self.pid_angular.compute(-error_norm, self.pid_dead_zone)
                sf = max(1.0 - abs(error_norm) * self.speed_reduction, 0.30)
                Vc = self.normal_speed * sf
            elif state == ST_SLOW_DOWN:
                Wc = self.pid_angular.compute(-error_norm, self.pid_dead_zone)
                Vc = self.slow_speed
            elif state in (ST_STOP_SIGN, ST_WAIT_RED, ST_CROSSWALK):
                Vc = 0.0; Wc = 0.0
            elif state == ST_TURN_LEFT:
                Vc = self.slow_speed; Wc = 0.8
            elif state == ST_TURN_RIGHT:
                Vc = self.slow_speed; Wc = -0.8
            elif state == ST_GO_STRAIGHT:
                Vc = self.normal_speed; Wc = 0.0
            else:
                Wc = self.pid_angular.compute(-error_norm, self.pid_dead_zone)
                sf = max(1.0 - abs(error_norm) * self.speed_reduction, 0.30)
                Vc = self.normal_speed * sf

        with self.lock:
            self.traffic_state = tl_state
            self._cmd_linear   = Vc
            self._cmd_angular  = Wc
            self._error_norm   = error_norm
            self._line_lost    = line_lost

        msg      = String()
        msg.data = state
        self.pub_state.publish(msg)

        # ── Debug overlay ─────────────────────────────────────────────────
        debug = image.copy()
        debug = self._draw(debug, result, error_norm, tl_state,
                           line_lost, Vc, Wc, state)
        self._video_writer.write(debug)
        cv.imshow('PuzzleBot', debug)
        cv.waitKey(1)

        self._log_count += 1
        if self._log_count >= 10:
            self._log_count = 0
            calib = "undistort=ON" if self._map1 is not None else "undistort=OFF"
            self.get_logger().info(
                f"TL={tl_state} | state={state} | "
                f"err={error_norm:+.3f} | Vc={Vc:.3f} Wc={Wc:+.3f} | {calib}"
            )

    # ══════════════════════════════════════════════════════════════════════
    def control_callback(self):
        """Publish cmd_vel and handle timed state transitions."""
        with self.lock:
            state = self._state
            Vc    = self._cmd_linear
            Wc    = self._cmd_angular

        elapsed = time.time() - self._state_start

        # Timed transitions
        if state == ST_STOP_SIGN and elapsed >= T_STOP_SIGN:
            self._transition(ST_FOLLOWING)
        elif state == ST_TURN_LEFT and elapsed >= T_TURN:
            self._transition(ST_FOLLOWING)
        elif state == ST_TURN_RIGHT and elapsed >= T_TURN:
            self._transition(ST_FOLLOWING)
        elif state == ST_GO_STRAIGHT and elapsed >= T_STRAIGHT:
            self._transition(ST_FOLLOWING)
        elif state == ST_CROSSWALK and elapsed >= T_CROSSWALK:
            self._transition(ST_FOLLOWING)

        self._publish(Vc, Wc)

    # ══════════════════════════════════════════════════════════════════════
    def _publish(self, linear, angular):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.pub_vel.publish(cmd)

    # ══════════════════════════════════════════════════════════════════════
    def _draw(self, image, result, error_norm, tl_state,
              line_lost, Vc, Wc, state) -> np.ndarray:
        h, w = image.shape[:2]

        # Center line
        cv.line(image, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Line detector ROI + dot
        self.line_detector.draw_debug(image, result)

        # ── YOLO sign detections overlay ──────────────────────────────────
        with self.lock:
            detections = list(self._sign_detections)

        sign_colors = {
            "stop":        (0, 0, 255),
            "turn_left":   (255, 128, 0),
            "turn_right":  (255, 128, 0),
            "go_straight": (0, 255, 128),
            "roadwork":    (0, 128, 255),
            "crosswalk":   (255, 0, 128),
            "give_way":    (0, 200, 255),
            "semaforo":    (200, 200, 0),
        }
        for det in detections:
            bbox  = det.get("bbox", None)   # [x1,y1,x2,y2] if available
            clase = det.get("class_name", "?")
            conf  = det.get("confidence", 0.0)
            color = sign_colors.get(clase, (200, 200, 200))

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv.putText(image, f"{clase} {conf:.0%}",
                           (x1, max(y1 - 6, 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── HUD ───────────────────────────────────────────────────────────
        tl_color = {
            "red": (0,0,255), "yellow": (0,255,255),
            "green": (0,255,0), "none": (255,255,255)
        }.get(tl_state, (255,255,255))

        state_color = {
            ST_FOLLOWING:   (0, 255, 0),
            ST_SLOW_DOWN:   (0, 255, 255),
            ST_STOP_SIGN:   (0, 0, 255),
            ST_WAIT_RED:    (0, 0, 255),
            ST_TURN_LEFT:   (255, 128, 0),
            ST_TURN_RIGHT:  (255, 128, 0),
            ST_GO_STRAIGHT: (0, 255, 128),
            ST_CROSSWALK:   (255, 0, 128),
        }.get(state, (255,255,255))

        overlay = image.copy()
        cv.rectangle(overlay, (10, 10), (440, 185), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        cv.putText(image, f"Traffic: {tl_state.upper()} "
                   f"({self.traffic_detector.confidence:.0%})",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.75, tl_color, 2)
        cv.putText(image, f"State: {state}",
                   (20, 68), cv.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv.putText(image, f"Err: {error_norm:+.3f}",
                   (20, 94), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv.putText(image, f"Vc={Vc:.3f} m/s   Wc={Wc:+.3f} rad/s",
                   (20, 118), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200,230,255), 1)
        cv.putText(image, f"PID Kp={self.pid_angular.Kp} "
                   f"Ki={self.pid_angular.Ki} Kd={self.pid_angular.Kd}",
                   (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.4, (170,170,170), 1)

        calib_label = "undistort=ON" if self._map1 is not None else "undistort=OFF"
        cv.putText(image, calib_label,
                   (20, 158), cv.FONT_HERSHEY_SIMPLEX, 0.38,
                   (0,255,100) if self._map1 is not None else (80,80,80), 1)

        # Sign detections summary on HUD
        if detections:
            signs_txt = "  ".join(
                f"{d.get('class_name','?')}({d.get('confidence',0):.0%})"
                for d in detections[:3]
            )
            cv.putText(image, f"Signs: {signs_txt}",
                       (20, 176), cv.FONT_HERSHEY_SIMPLEX, 0.38,
                       (255, 200, 100), 1)

        if line_lost:
            cv.putText(image, "LINE LOST",
                       (w // 2 - 100, h // 2),
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