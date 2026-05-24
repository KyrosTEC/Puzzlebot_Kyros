"""
Camera calibration script for the PuzzleBot wide-angle camera.

Usage:
    1. Print a 9x6 chessboard pattern (or use a digital screen)
    2. Run:  python3 calibrate.py
    3. Press S to save a frame, Q when done (need at least 15 good frames)
    4. Calibration runs automatically and saves camera_params.npz

The output file camera_params.npz contains:
    camera_matrix  : 3x3 intrinsic matrix
    dist_coeffs    : distortion coefficients (k1,k2,p1,p2,k3)
    img_size       : (width, height) the calibration was done at

Copy camera_params.npz to ~/ros2_ws/src/half_term_challenge/
"""

import cv2
import numpy as np
import glob
import os
import sys

# ── Chessboard settings ───────────────────────────────────────────────────
# These must match your printed/displayed pattern.
# (cols-1) x (rows-1) inner corners — a standard 9x6 board has 8x5 inner corners
CHESS_COLS   = 8    # inner corners horizontally
CHESS_ROWS   = 5    # inner corners vertically
SQUARE_SIZE  = 1.0  # real-world size (use 1.0 for normalized units)

# ── Camera device ─────────────────────────────────────────────────────────
DEVICE_ID    = 0
FRAME_W      = 640
FRAME_H      = 360
MIN_FRAMES   = 15   # minimum good detections before calibrating

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_FILE  = os.path.expanduser("~/camera_params.npz")


def collect_and_calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points in real-world space (z=0 plane)
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []   # 3D points
    img_points = []   # 2D points in image

    cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        # fallback: try GStreamer CSI pipeline
        from half_term_challenge.traffic_light_detection import gstreamer_pipeline
        cap = cv2.VideoCapture(
            gstreamer_pipeline(flip_method=0,
                               capture_width=1280, capture_height=720,
                               display_width=FRAME_W, display_height=FRAME_H,
                               framerate=15),
            cv2.CAP_GSTREAMER
        )

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print(f"\n{'='*55}")
    print(f"  Chessboard: {CHESS_COLS}x{CHESS_ROWS} inner corners")
    print(f"  Need {MIN_FRAMES} good captures minimum")
    print(f"  S = save frame | Q = done & calibrate | ESC = quit")
    print(f"{'='*55}\n")

    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Try to find chessboard (show live feedback)
        found, corners = cv2.findChessboardCorners(
            gray, (CHESS_COLS, CHESS_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            cv2.drawChessboardCorners(display, (CHESS_COLS, CHESS_ROWS),
                                      corners, found)
            status_color = (0, 255, 0)
            status_text  = f"Board FOUND  |  saved={saved}/{MIN_FRAMES}  |  S to capture"
        else:
            status_color = (0, 100, 255)
            status_text  = f"Searching...  |  saved={saved}/{MIN_FRAMES}"

        cv2.putText(display, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
        cv2.imshow("Calibration — S=save  Q=calibrate  ESC=quit", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            saved += 1
            print(f"  Captured frame {saved}")

        elif key == ord('q'):
            if saved < MIN_FRAMES:
                print(f"  Need at least {MIN_FRAMES} frames (have {saved}). Keep going.")
            else:
                break

        elif key == 27:   # ESC
            print("Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # ── Run calibration ───────────────────────────────────────────────────
    print(f"\nCalibrating with {saved} frames...")
    img_size = (FRAME_W, FRAME_H)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"\nRMS reprojection error: {rms:.4f}  (good if < 1.0)")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")

    # Optimal new camera matrix (crops black borders after undistort)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, alpha=0.0
    )
    print(f"\nOptimal camera matrix (alpha=0, no black borders):\n{new_matrix}")

    np.savez(OUTPUT_FILE,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             new_camera_matrix=new_matrix,
             img_size=np.array(img_size))

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Copy this file to your ROS2 package source folder.")

    # ── Show undistortion preview ─────────────────────────────────────────
    print("\nShowing undistortion preview — press any key to close.")
    cap2 = cv2.VideoCapture(DEVICE_ID, cv2.CAP_V4L2)
    if not cap2.isOpened():
        return

    for _ in range(30):   # skip buffer frames
        cap2.read()

    ret, sample = cap2.read()
    cap2.release()

    if ret:
        undistorted = cv2.undistort(sample, camera_matrix, dist_coeffs,
                                    None, new_matrix)
        combined = np.hstack([sample, undistorted])
        cv2.putText(combined, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined, "Undistorted", (FRAME_W + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Before / After undistortion", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_and_calibrate()