import cv2
import numpy as np
import os
import glob
import time

# ============================================================
# CONFIGURACIÓN DEL PATRÓN
# ============================================================

# Número de esquinas INTERNAS del tablero.
# Si tu tablero tiene 8 cuadros x 6 cuadros → (7, 5)
CHESSBOARD_SIZE = (7, 5)

# Tamaño real de cada cuadro en mm.
# Mídelo con regla para que la matriz K tenga unidades reales.
SQUARE_SIZE = 25.0

# ============================================================
# CONFIGURACIÓN DE CÁMARA - JETSON NANO
# ============================================================

# Elige el modo según tu cámara:
#   "CSI"  → cámara Raspberry Pi / IMX219 conectada al conector CSI
#   "USB"  → cámara USB genérica (ej. Logitech)
CAMERA_MODE = "CSI"

# Solo para modo USB: índice del dispositivo (/dev/video0 → 0)
USB_INDEX = 0

# Resolución de captura
WIDTH  = 640
HEIGHT = 480

# Framerate para GStreamer (solo CSI)
FRAMERATE = 30

# ============================================================
# RUTAS — dentro del workspace ROS2 de la Jetson
# ============================================================

# Cambia esto si tu workspace tiene otro nombre o ubicación
ROS2_WS = os.path.expanduser("~/ros2_ws")

# Carpeta donde se guardarán las imágenes de calibración
CALIB_DIR = os.path.join(ROS2_WS, "calibracion", "imagenes")

# Archivo de salida con los parámetros de la cámara
OUTPUT_NPZ = os.path.join(ROS2_WS, "calibracion", "calibracion_jetson.npz")

# Mínimo recomendado de imágenes válidas
MIN_VALID_IMAGES = 10

# Cooldown entre capturas (segundos) para evitar fotos idénticas
CAPTURE_COOLDOWN = 1.0


# ============================================================
# UTILIDADES
# ============================================================

def abrir_camara():
    """
    Abre la cámara según CAMERA_MODE.
    Retorna un objeto cv2.VideoCapture listo para usar.
    """
    if CAMERA_MODE == "CSI":
        # Pipeline GStreamer para cámara CSI en Jetson Nano
        pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, "
            f"format=NV12, framerate={FRAMERATE}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={WIDTH}, height={HEIGHT}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # Cámara USB estándar
        cap = cv2.VideoCapture(USB_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    return cap


# ============================================================
# 1. CAPTURAR IMÁGENES DEL PATRÓN
# ============================================================

def capturar_imagenes():
    os.makedirs(CALIB_DIR, exist_ok=True)

    cap = abrir_camara()

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara en modo {CAMERA_MODE}.")
        if CAMERA_MODE == "CSI":
            print("  Verifica que nvarguscamerasrc esté disponible.")
            print("  Prueba: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink")
        else:
            print(f"  Prueba cambiando USB_INDEX a 1 o 2.")
        return

    print("\n=== CAPTURA DE IMÁGENES PARA CALIBRACIÓN (JETSON NANO) ===")
    print(f"  Modo cámara : {CAMERA_MODE}")
    print(f"  Resolución  : {WIDTH}x{HEIGHT}")
    print(f"  Guardando en: {CALIB_DIR}")
    print("")
    print("  Presiona 'c' para capturar (solo si el patrón está detectado).")
    print("  Presiona 'f' para forzar captura sin detección.")
    print("  Presiona 'q' para salir.")
    print("  Mueve el patrón: centro, esquinas, inclinado, cerca y lejos.\n")

    # Continuar numeración si ya hay imágenes previas
    existing = glob.glob(os.path.join(CALIB_DIR, "zhang_*.jpg"))
    existing = [f for f in existing if "_corners" not in f]
    count = len(existing)

    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] No se pudo leer frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()

        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
            cv2.putText(
                display,
                f"DETECTADO | c=guardar | imgs={count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                display,
                f"Buscando patron... | imgs={count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
            )

        cv2.imshow("Calibracion Jetson - Patron", display)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # Captura normal: solo si el patrón fue detectado
        if key == ord("c") and found and (now - last_capture_time) >= CAPTURE_COOLDOWN:
            filename = os.path.join(CALIB_DIR, f"zhang_{count:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"  Guardada: {filename}")
            count += 1
            last_capture_time = now

        # Captura forzada: aunque no detecte (útil para revisar iluminación)
        elif key == ord("f") and (now - last_capture_time) >= CAPTURE_COOLDOWN:
            filename = os.path.join(CALIB_DIR, f"zhang_{count:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"  Forzada (sin detección): {filename}")
            count += 1
            last_capture_time = now

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal de imágenes capturadas: {count}")


# ============================================================
# 2. CALIBRACIÓN CON MÉTODO DE ZHANG 2D
# ============================================================

def calibrar_zhang():
    images = sorted([
        f for f in glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
        if "_corners" not in f
    ])

    if len(images) == 0:
        print(f"[ERROR] No hay imágenes en: {CALIB_DIR}")
        print("  Primero ejecuta la opción 1 para capturar imágenes.")
        return None

    print(f"\nImágenes encontradas: {len(images)}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Puntos 3D del patrón (Z=0 porque es plano)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # Puntos 3D mundo real
    imgpoints = []  # Puntos 2D imagen
    image_size = None

    print("\n=== BUSCANDO ESQUINAS ===")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  [SKIP] No se pudo leer: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)

            # Guardar preview con esquinas dibujadas
            img_preview = img.copy()
            cv2.drawChessboardCorners(img_preview, CHESSBOARD_SIZE, corners_refined, found)
            preview_path = fname.replace(".jpg", "_corners.jpg")
            cv2.imwrite(preview_path, img_preview)

            print(f"  OK : {os.path.basename(fname)}")
        else:
            print(f"  NO : {os.path.basename(fname)}  ← patrón no detectado")

    print(f"\nImágenes válidas: {len(objpoints)} / {len(images)}")

    if len(objpoints) < MIN_VALID_IMAGES:
        print(f"[AVISO] Se recomiendan al menos {MIN_VALID_IMAGES} imágenes válidas.")
        print("  Captura más fotos del patrón desde distintas posiciones y ángulos.")
        return None

    # Flags: solo k1, k2 y p1, p2 (adecuado para cámaras del Puzzlebot)
    flags = (
        cv2.CALIB_ZERO_TANGENT_DIST |
        cv2.CALIB_FIX_K3 |
        cv2.CALIB_FIX_K4 |
        cv2.CALIB_FIX_K5 |
        cv2.CALIB_FIX_K6
    )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    print("\n=== RESULTADOS DE CALIBRACIÓN ===")
    print(f"  Error RMS : {ret:.4f}  (< 1.0 es bueno, < 0.5 es excelente)")
    print(f"\n  Matriz intrínseca K:")
    print(f"    fx = {K[0,0]:.2f} px")
    print(f"    fy = {K[1,1]:.2f} px")
    print(f"    cx = {K[0,2]:.2f} px")
    print(f"    cy = {K[1,2]:.2f} px")
    print(f"\n  Distorsión [k1, k2, p1, p2]:")
    print(f"    {dist.ravel()[:4]}")

    # Guardar en carpeta del workspace
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(
        OUTPUT_NPZ,
        K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, error_rms=ret
    )
    print(f"\n  Guardado en: {OUTPUT_NPZ}")

    return {
        "K": K, "dist": dist,
        "rvecs": rvecs, "tvecs": tvecs,
        "error_rms": ret,
        "objpoints": objpoints, "imgpoints": imgpoints
    }


# ============================================================
# 3. ERROR DE REPROYECCIÓN
# ============================================================

def calcular_error_reproyeccion(data):
    mean_error = 0.0
    for i in range(len(data["objpoints"])):
        projected, _ = cv2.projectPoints(
            data["objpoints"][i],
            data["rvecs"][i], data["tvecs"][i],
            data["K"], data["dist"]
        )
        error = cv2.norm(
            data["imgpoints"][i], projected, cv2.NORM_L2SQR
        ) / len(projected)
        mean_error += error

    rmse = np.sqrt(mean_error / len(data["objpoints"]))
    print(f"\n=== ERROR DE REPROYECCIÓN ===")
    print(f"  RMSE: {rmse:.4f} px")
    return rmse


# ============================================================
# 4. UNDISTORT - IMAGEN GUARDADA
# ============================================================

def undistort_imagen_guardada():
    if not os.path.exists(OUTPUT_NPZ):
        print(f"[ERROR] No existe: {OUTPUT_NPZ}")
        print("  Primero realiza la calibración (opción 2 o 4).")
        return

    data = np.load(OUTPUT_NPZ)
    K, dist = data["K"], data["dist"]

    # Tomar la primera imagen de calibración como ejemplo
    images = sorted([
        f for f in glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
        if "_corners" not in f
    ])
    if not images:
        print("[ERROR] No hay imágenes en la carpeta de calibración.")
        return

    img = cv2.imread(images[0])
    if img is None:
        print("[ERROR] No se pudo cargar la imagen.")
        return

    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    x, y, w_roi, h_roi = roi
    undistorted_crop = undistorted[y:y+h_roi, x:x+w_roi]

    out_dir = os.path.dirname(OUTPUT_NPZ)
    cv2.imwrite(os.path.join(out_dir, "imagen_original.jpg"), img)
    cv2.imwrite(os.path.join(out_dir, "imagen_sin_distorsion.jpg"), undistorted)
    cv2.imwrite(os.path.join(out_dir, "imagen_sin_distorsion_crop.jpg"), undistorted_crop)

    cv2.imshow("Original", img)
    cv2.imshow("Sin distorsion", undistorted)
    cv2.imshow("Sin distorsion (recortada)", undistorted_crop)
    print(f"\nImágenes guardadas en: {out_dir}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 5. UNDISTORT - TIEMPO REAL
# ============================================================

def undistort_tiempo_real():
    if not os.path.exists(OUTPUT_NPZ):
        print(f"[ERROR] No existe: {OUTPUT_NPZ}")
        print("  Primero realiza la calibración.")
        return

    data = np.load(OUTPUT_NPZ)
    K, dist = data["K"], data["dist"]

    cap = abrir_camara()
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara en modo {CAMERA_MODE}.")
        return

    print("\nMostrando cámara original y sin distorsión.")
    print("Presiona 'q' para salir.\n")

    new_K = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame.")
            break

        h, w = frame.shape[:2]
        if new_K is None:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        cv2.imshow("Original", frame)
        cv2.imshow("Sin distorsion", undistorted)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n==============================================")
    print(" Calibración Zhang 2D - Puzzlebot Jetson Nano")
    print(f" Cámara: {CAMERA_MODE} | {WIDTH}x{HEIGHT}")
    print(f" Workspace: {ROS2_WS}")
    print("==============================================")
    print("\nOpciones:")
    print("  1. Capturar imágenes del patrón")
    print("  2. Calibrar cámara (Zhang 2D)")
    print("  3. Ver undistort en imagen guardada")
    print("  4. Calibrar + error de reproyección")
    print("  5. Calibrar + error + undistort (todo)")
    print("  6. Undistort en tiempo real")

    opcion = input("\nSelecciona una opción: ").strip()

    if opcion == "1":
        capturar_imagenes()
    elif opcion == "2":
        calibrar_zhang()
    elif opcion == "3":
        undistort_imagen_guardada()
    elif opcion == "4":
        data = calibrar_zhang()
        if data:
            calcular_error_reproyeccion(data)
    elif opcion == "5":
        data = calibrar_zhang()
        if data:
            calcular_error_reproyeccion(data)
            undistort_imagen_guardada()
    elif opcion == "6":
        undistort_tiempo_real()
    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
