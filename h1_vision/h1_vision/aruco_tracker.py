#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker de pose basado en marcadores ArUco para el Unitree H1-2.

Responsabilidades principales:
- detectar un marcador ArUco configurado en la imagen RGB;
- utilizar los intrínsecos y coeficientes de distorsión de CameraInfo;
- estimar la pose a partir del tamaño físico conocido del marcador;
- evaluar la consistencia geométrica mediante error de reproyección;
- suavizar temporalmente la pose estimada (posición y orientación);
- publicar TF y PoseStamped en el marco de la cámara.

Este nodo proporciona una referencia geométrica asociada con la mesa y no realiza
calibración completa de la cámara ni estimación basada en profundidad. La transformación
hacia el marco de pelvis se realiza posteriormente en el supervisor de alto nivel.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from collections import deque
from scipy.spatial.transform import Rotation as R

def get_aruco_dictionary(name):
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    }
    if name not in dict_map:
        raise ValueError(f"Diccionario ArUco no soportado: {name}")

    dict_id = dict_map[name]

    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    elif hasattr(cv2.aruco, 'Dictionary_get'):
        return cv2.aruco.Dictionary_get(dict_id)
    else:
        return getattr(cv2.aruco, name)

# ============================================================
# Responsabilidad general del detector ArUco
# ============================================================
class ArucoTrackerNode(Node):
    def __init__(self):
        # ============================================================
        # Inicialización y parámetros
        # ============================================================

        super().__init__('aruco_tracker_node')

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV no incluye el modulo aruco. Instalar opencv-contrib-python o usar una version de OpenCV con aruco.")

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('marker_frame', 'aruco_mesa')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size_m', 0.093)
        self.declare_parameter('aruco_dictionary', 'DICT_4X4_50')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', '/vision/aruco_mesa/pose')
        self.declare_parameter('debug_visual', False)
        self.declare_parameter('show_main_window', True)
        self.declare_parameter('show_debug_overlay', False)
        self.declare_parameter('show_axes', False)
        self.declare_parameter('axis_length_m', 0.04)
        self.declare_parameter('alpha_pos', 0.25)
        self.declare_parameter('alpha_rot', 0.25)
        self.declare_parameter('expected_camera_fps', 30.0)
        self.declare_parameter('fps_window_size', 30)
        self.declare_parameter('reject_if_ambiguous', True)
        self.declare_parameter('max_reprojection_error_px', 5.0)

        self.color_topic = str(self.get_parameter('color_topic').value).strip()
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value).strip()
        self.camera_frame = str(self.get_parameter('camera_frame').value).strip()
        self.marker_frame = str(self.get_parameter('marker_frame').value).strip()
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.aruco_dictionary_name = str(self.get_parameter('aruco_dictionary').value).strip()
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.publish_pose = bool(self.get_parameter('publish_pose').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value).strip()
        self.debug_visual = bool(self.get_parameter('debug_visual').value)
        self.show_main_window = bool(self.get_parameter('show_main_window').value)
        self.show_debug_overlay = bool(self.get_parameter('show_debug_overlay').value)
        self.show_axes = bool(self.get_parameter('show_axes').value)
        self.axis_length_m = float(self.get_parameter('axis_length_m').value)
        self.alpha_pos = float(self.get_parameter('alpha_pos').value)
        self.alpha_rot = float(self.get_parameter('alpha_rot').value)
        self.expected_camera_fps = float(self.get_parameter('expected_camera_fps').value)
        self.fps_window_size = int(self.get_parameter('fps_window_size').value)
        self.reject_if_ambiguous = bool(self.get_parameter('reject_if_ambiguous').value)
        self.max_reprojection_error_px = float(self.get_parameter('max_reprojection_error_px').value)

        if self.debug_visual:
            self.show_main_window = True
            self.show_debug_overlay = True
            self.show_axes = True

        if self.marker_id < 0:
            raise ValueError("marker_id debe ser mayor o igual que 0")
        if self.marker_size_m <= 0.0:
            raise ValueError("marker_size_m debe ser mayor que 0.0")
        if self.axis_length_m <= 0.0:
            raise ValueError("axis_length_m debe ser mayor que 0.0")
        if not (0.0 < self.alpha_pos <= 1.0):
            raise ValueError("alpha_pos debe estar en el rango (0.0, 1.0]")
        if not (0.0 < self.alpha_rot <= 1.0):
            raise ValueError("alpha_rot debe estar en el rango (0.0, 1.0]")
        if self.expected_camera_fps <= 0.0:
            raise ValueError("expected_camera_fps debe ser mayor que 0.0")
        if self.fps_window_size <= 1:
            raise ValueError("fps_window_size debe ser mayor que 1")
        if not self.marker_frame:
            raise ValueError("marker_frame no debe estar vacio")

        self.aruco_dict = get_aruco_dictionary(self.aruco_dictionary_name)

        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        self.main_window_name = self.marker_frame

        self.frame_timestamps = deque(maxlen=self.fps_window_size)
        self.last_processing_time_ms = 0.0
        self.camera_fps_estimate = 0.0

        self.last_tvec = None
        self.last_quat = None
        self.last_reprojection_error_px = None
        self.lost_frame_count = 0
        self.last_detection_stamp = None
        self.last_pose_log_time = 0.0

        # ============================================================
        # Interfaces ROS 2
        # ============================================================
        self.subscription_color = self.create_subscription(Image, self.color_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self.bridge = CvBridge()
        self.intrinsics_loaded = False
        self.camera_matrix = None
        self.dist_coeffs = None

        self.get_logger().info(
            f"[DETECTOR ARUCO INICIALIZADO]\n"
            f"  Topico RGB: '{self.color_topic}'\n"
            f"  Topico CameraInfo: '{self.camera_info_topic}'\n"
            f"  Frame de camara: '{self.camera_frame}'\n"
            f"  Frame del marcador: '{self.marker_frame}'\n"
            f"  ID marcador: {self.marker_id}\n"
            f"  Tamano marcador [m]: {self.marker_size_m:.4f}\n"
            f"  Diccionario ArUco: '{self.aruco_dictionary_name}'\n"
            f"  Publicar TF: {self.publish_tf}\n"
            f"  Publicar PoseStamped: {self.publish_pose}\n"
            f"  Topico de pose: '{self.pose_topic}'\n"
            f"  Debug visual: {self.debug_visual}"
        )

    def stamp_to_seconds(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def update_fps_estimate(self, msg_stamp, processing_time_ms):
        timestamp = self.stamp_to_seconds(msg_stamp)
        if timestamp <= 0.0:
            timestamp = time.perf_counter()

        if len(self.frame_timestamps) == 0 or timestamp > self.frame_timestamps[-1]:
            self.frame_timestamps.append(timestamp)

        if len(self.frame_timestamps) >= 2:
            duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if duration > 1e-6:
                fps = (len(self.frame_timestamps) - 1) / duration
            else:
                fps = 0.0
        else:
            fps = 0.0

        self.camera_fps_estimate = min(fps, self.expected_camera_fps)
        self.last_processing_time_ms = processing_time_ms

    # ============================================================
    # Recepción de CameraInfo
    # ============================================================
    def camera_info_callback(self, msg):

        # La matriz K contiene las distancias focales fx y fy en píxeles,
        # además del punto principal cx y cy.
        self.camera_matrix = np.array([
            [msg.k[0], msg.k[1], msg.k[2]],
            [msg.k[3], msg.k[4], msg.k[5]],
            [msg.k[6], msg.k[7], msg.k[8]]
        ], dtype=np.float64)

        if msg.d and len(msg.d) > 0:
        # D contiene los coeficientes de distorsión proporcionados
        # por el modelo de cámara.
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.intrinsics_loaded = True
        self.get_logger().info("Intrinsecos de camara cargados para ArUco.")
        self.destroy_subscription(self.info_sub)

    # ============================================================
    # Interpolación lineal normalizada de la orientación
    # ============================================================
    # Se alinea primero el signo para mantener continuidad entre q y -q.
    # Luego se aplica una interpolación lineal y se normaliza el resultado.
    def interpolate_quaternion(self, q1, q2, alpha):

        dot = np.dot(q1, q2)
        # Se alinea el signo del cuaternión con la estimación anterior para
        # evitar discontinuidades numéricas entre q y -q.

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        q_out = (1.0 - alpha) * q1 + alpha * q2
        norm = np.linalg.norm(q_out)
        if norm > 1e-6:
            return q_out / norm
        return q1

    # ============================================================
    # Publicación de resultados
    # ============================================================
    def publish_aruco_tf_and_pose(self, tvec, quat_xyzw, stamp, source_frame="", reprojection_error_px=None):
        if tvec is None or quat_xyzw is None:
            return

        tvec = np.array(tvec, dtype=float).flatten()
        quat_xyzw = np.array(quat_xyzw, dtype=float).flatten()

        if np.any(np.isnan(tvec)) or np.any(np.isinf(tvec)):
            return
        if np.any(np.isnan(quat_xyzw)) or np.any(np.isinf(quat_xyzw)):
            return

        norm_quat = np.linalg.norm(quat_xyzw)
        if norm_quat < 1e-6:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat_xyzw = quat_xyzw / norm_quat

        frame_id = self.camera_frame if self.camera_frame else source_frame
        if not frame_id:
            now = time.time()
            if not hasattr(self, '_last_frame_warn_time'):
                self._last_frame_warn_time = 0.0
            if now - self._last_frame_warn_time >= 5.0:
                self._last_frame_warn_time = now
                self.get_logger().warn("[DETECTOR ARUCO] Frame de camara vacio; se omite publicacion TF/Pose.")
            return

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = frame_id
            t.child_frame_id = self.marker_frame
            t.transform.translation.x = tvec[0]
            t.transform.translation.y = tvec[1]
            t.transform.translation.z = tvec[2]
            t.transform.rotation.x = quat_xyzw[0]
            t.transform.rotation.y = quat_xyzw[1]
            t.transform.rotation.z = quat_xyzw[2]
            t.transform.rotation.w = quat_xyzw[3]
            self.tf_broadcaster.sendTransform(t)

        if self.publish_pose:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = tvec[0]
            pose_msg.pose.position.y = tvec[1]
            pose_msg.pose.position.z = tvec[2]
            pose_msg.pose.orientation.x = quat_xyzw[0]
            pose_msg.pose.orientation.y = quat_xyzw[1]
            pose_msg.pose.orientation.z = quat_xyzw[2]
            pose_msg.pose.orientation.w = quat_xyzw[3]
            self.pose_pub.publish(pose_msg)

        now = time.time()
        if now - self.last_pose_log_time >= 1.0:
            self.last_pose_log_time = now
            err_str = f"{reprojection_error_px:.4f}" if reprojection_error_px is not None else "None"
            self.get_logger().info(
                f"POSE ARUCO -> centro={tvec.tolist()} frame={frame_id} objeto={self.marker_frame} "
                f"id={self.marker_id} error_reproyeccion={err_str} px"
            )

    # ============================================================
    # Procesamiento principal de la imagen
    # ============================================================
    def image_callback(self, msg):
        if not self.intrinsics_loaded:
            now = time.time()
            if not hasattr(self, '_last_info_warn_time'):
                self._last_info_warn_time = 0.0
            if now - self._last_info_warn_time >= 5.0:
                self._last_info_warn_time = now
                self.get_logger().warn("[DETECTOR ARUCO] No se recibio CameraInfo; esperando intrinsecos.")
            return

        start_proc = time.perf_counter()
        try:

            # Conversión del mensaje ROS a imagen OpenCV.
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(cv_image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

            marker_found = False
            rvec, tvec = None, None
            corners_of_interest = None

            if ids is not None:

                for idx, mid in enumerate(ids.flatten()):
                    if mid == self.marker_id:
                        marker_found = True
                        corners_of_interest = corners[idx]
                        break

            if not marker_found:
                self.lost_frame_count += 1
                self.last_reprojection_error_px = None

                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            self.lost_frame_count = 0
            self.last_detection_stamp = msg.header.stamp

            s = self.marker_size_m / 2.0

            # Selección del marcador y definición de sus coordenadas métricas 3D.
            # El tamaño físico conocido del marcador define las coordenadas métricas
            # de sus cuatro esquinas.
            object_points = np.array([
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0]
            ], dtype=np.float32)

            image_points = corners_of_interest.reshape((4, 2)).astype(np.float32)

            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE') else cv2.SOLVEPNP_ITERATIVE

            # ============================================================
            # Estimación de pose
            # ============================================================
            # PnP estima la transformación del marcador respecto a la cámara
            # utilizando correspondencias entre puntos 3D y esquinas 2D.
            # rvec representa la rotación en formato eje–ángulo.
            # tvec representa la posición del origen del marcador respecto
            # al marco de la cámara, expresada en metros.
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=pnp_flag
            )

            if not success:

                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            current_tvec = tvec.flatten()
            if current_tvec[2] <= 0.0:
                now = time.time()
                if not hasattr(self, '_last_z_warn_time'):
                    self._last_z_warn_time = 0.0
                if now - self._last_z_warn_time >= 1.0:
                    self._last_z_warn_time = now
                    self.get_logger().warn("[DETECTOR ARUCO] Pose rechazada: marcador detras de la camara o Z invalida.")

                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # El error de reproyección compara las esquinas detectadas con las
            # proyectadas nuevamente a partir de la pose estimada.
            # Se expresa en píxeles y mide consistencia geométrica en la imagen.
            # Un error reducido no garantiza por sí solo exactitud métrica en 3D.
            projected_points, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            projected_points = projected_points.reshape((4, 2))

            reproj_error = float(np.mean(np.linalg.norm(image_points - projected_points, axis=1)))
            self.last_reprojection_error_px = reproj_error

            # La estimación se rechaza cuando supera el umbral configurado.
            if self.reject_if_ambiguous and reproj_error > self.max_reprojection_error_px:
                now = time.time()
                if not hasattr(self, '_last_reproj_warn_time'):
                    self._last_reproj_warn_time = 0.0
                if now - self._last_reproj_warn_time >= 1.0:
                    self._last_reproj_warn_time = now
                    self.get_logger().warn("[DETECTOR ARUCO] Error de reproyeccion alto; se rechaza deteccion.")

                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # Suavizado lineal de la posición estimada.
            if self.last_tvec is None:
                self.last_tvec = current_tvec.copy()
            else:
                self.last_tvec = (1.0 - self.alpha_pos) * self.last_tvec + self.alpha_pos * current_tvec

            # Rodrigues convierte el vector eje–ángulo rvec en una
            # matriz de rotación de 3 x 3.
            rmat, _ = cv2.Rodrigues(rvec)
            rot = R.from_matrix(rmat)
            # La matriz se convierte a cuaternión para suavizar y publicar
            # la orientación mediante ROS 2.
            current_quat = rot.as_quat()

            # Suavizado N-LERP de la orientación después de alinear su signo.
            if self.last_quat is None:
                self.last_quat = current_quat.copy()
            else:
                self.last_quat = self.interpolate_quaternion(self.last_quat, current_quat, self.alpha_rot)

            self.publish_aruco_tf_and_pose(
                self.last_tvec,
                self.last_quat,
                msg.header.stamp,
                source_frame=msg.header.frame_id,
                reprojection_error_px=self.last_reprojection_error_px
            )

            if self.show_main_window:

                pts_cnt = image_points.astype(np.int32).reshape((-1, 1, 2))

                # Visualización de depuración.
                cv2.polylines(cv_image, [pts_cnt], True, (0, 255, 0), 2)

                cx_img = int(np.mean(image_points[:, 0]))
                cy_img = int(np.mean(image_points[:, 1]))
                cv2.circle(cv_image, (cx_img, cy_img), 5, (0, 0, 255), -1)

                if self.show_debug_overlay:

                    if self.show_axes:
                        if hasattr(cv2, 'drawFrameAxes'):
                            cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.axis_length_m)
                        elif hasattr(cv2.aruco, 'drawAxis'):
                            cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.axis_length_m)

                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                    err_val = f"{self.last_reprojection_error_px:.2f}" if self.last_reprojection_error_px is not None else "N/A"
                    frame_id_res = self.camera_frame if self.camera_frame else msg.header.frame_id

                    cv2.putText(cv_image, f"ID marcador: {self.marker_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Z [m]: {self.last_tvec[2]:.3f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Error reproyeccion [px]: {err_val}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"FPS camara: {self.camera_fps_estimate:.1f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Proc [ms]: {self.last_processing_time_ms:.1f}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frames perdidos: {self.lost_frame_count}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frame camara: {frame_id_res}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frame marcador: {self.marker_frame}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:

                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                cv2.imshow(self.main_window_name, cv_image)
                cv2.setWindowTitle(self.main_window_name, titulo)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error en image_callback: {e}")


# ============================================================
# Punto de entrada
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = ArucoTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
