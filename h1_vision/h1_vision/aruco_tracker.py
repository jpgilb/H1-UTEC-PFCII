#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

class ArucoTrackerNode(Node):
    def __init__(self):
        super().__init__('aruco_tracker_node')
        
        # Validar modulo ArUco de OpenCV
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV no incluye el modulo aruco. Instalar opencv-contrib-python o usar una version de OpenCV con aruco.")

        # Declarar parámetros
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

        # Cargar parámetros
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

        # Regla de activación de debug visual
        if self.debug_visual:
            self.show_main_window = True
            self.show_debug_overlay = True
            self.show_axes = True

        # Validaciones
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

        # Cargar diccionario ArUco
        self.aruco_dict = get_aruco_dictionary(self.aruco_dictionary_name)
        
        # Parámetros del detector ArUco
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        # Nombres de ventana estandarizados
        self.main_window_name = self.marker_frame

        # Estado de FPS
        self.frame_timestamps = deque(maxlen=self.fps_window_size)
        self.last_processing_time_ms = 0.0
        self.camera_fps_estimate = 0.0

        # Histórico de tracking y errores
        self.last_tvec = None
        self.last_quat = None
        self.last_reprojection_error_px = None
        self.lost_frame_count = 0
        self.last_detection_stamp = None
        self.last_pose_log_time = 0.0

        # Suscripciones
        self.subscription_color = self.create_subscription(Image, self.color_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # Publicadores
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self.bridge = CvBridge()
        self.intrinsics_loaded = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # Log de inicio en español
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

    def camera_info_callback(self, msg):
        # Matriz K
        self.camera_matrix = np.array([
            [msg.k[0], msg.k[1], msg.k[2]],
            [msg.k[3], msg.k[4], msg.k[5]],
            [msg.k[6], msg.k[7], msg.k[8]]
        ], dtype=np.float64)
        
        # Coeficientes de distorsión
        if msg.d and len(msg.d) > 0:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            
        self.intrinsics_loaded = True
        self.get_logger().info("Intrinsecos de camara cargados para ArUco.")
        self.destroy_subscription(self.info_sub)

    def interpolate_quaternion(self, q1, q2, alpha):
        # Asegurar el camino mas corto (short path)
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
            
        # Interpolacion lineal (LERP) y normalizacion
        q_out = (1.0 - alpha) * q1 + alpha * q2
        norm = np.linalg.norm(q_out)
        if norm > 1e-6:
            return q_out / norm
        return q1

    def publish_aruco_tf_and_pose(self, tvec, quat_xyzw, stamp, source_frame="", reprojection_error_px=None):
        if tvec is None or quat_xyzw is None:
            return

        tvec = np.array(tvec, dtype=float).flatten()
        quat_xyzw = np.array(quat_xyzw, dtype=float).flatten()

        # Validar NaN/Inf
        if np.any(np.isnan(tvec)) or np.any(np.isinf(tvec)):
            return
        if np.any(np.isnan(quat_xyzw)) or np.any(np.isinf(quat_xyzw)):
            return

        # Normalizar quaternion
        norm_quat = np.linalg.norm(quat_xyzw)
        if norm_quat < 1e-6:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat_xyzw = quat_xyzw / norm_quat

        # Determinar frame_id
        frame_id = self.camera_frame if self.camera_frame else source_frame
        if not frame_id:
            now = time.time()
            if not hasattr(self, '_last_frame_warn_time'):
                self._last_frame_warn_time = 0.0
            if now - self._last_frame_warn_time >= 5.0:
                self._last_frame_warn_time = now
                self.get_logger().warn("[DETECTOR ARUCO] Frame de camara vacio; se omite publicacion TF/Pose.")
            return

        # Publicar TransformStamped
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

        # Publicar PoseStamped
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

        # Logs Throttled (1 por segundo)
        now = time.time()
        if now - self.last_pose_log_time >= 1.0:
            self.last_pose_log_time = now
            err_str = f"{reprojection_error_px:.4f}" if reprojection_error_px is not None else "None"
            self.get_logger().info(
                f"POSE ARUCO -> centro={tvec.tolist()} frame={frame_id} objeto={self.marker_frame} "
                f"id={self.marker_id} error_reproyeccion={err_str} px"
            )

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
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detectar marcadores ArUco
            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(cv_image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

            marker_found = False
            rvec, tvec = None, None
            corners_of_interest = None

            if ids is not None:
                # Buscar el marker_id configurado
                for idx, mid in enumerate(ids.flatten()):
                    if mid == self.marker_id:
                        marker_found = True
                        corners_of_interest = corners[idx]
                        break

            if not marker_found:
                self.lost_frame_count += 1
                self.last_reprojection_error_px = None
                # No publicar nueva pose si se pierde el marcador
                
                # Mostrar ventana debug en modo original si corresponde
                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)
                    
                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # Si se encuentra el marcador:
            self.lost_frame_count = 0
            self.last_detection_stamp = msg.header.stamp

            # Puntos 3D del marcador en su propio frame, centrados en el marcador
            s = self.marker_size_m / 2.0
            object_points = np.array([
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0]
            ], dtype=np.float32)

            # solvePnP requiere los puntos de la imagen en formato float32 y shape (4, 2)
            image_points = corners_of_interest.reshape((4, 2)).astype(np.float32)

            # PnP Solver Flags
            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE') else cv2.SOLVEPNP_ITERATIVE

            # El ArUco se utiliza como referencia geometrica de mesa. La orientacion completa se publica para permitir futuros offsets relativos al yaw del marcador, aunque la primera integracion con mover_brazo_single.py debe usar table_target_offset_mode='base'.
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=pnp_flag
            )

            if not success:
                # Mostrar ventana debug en modo original si corresponde
                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)
                    
                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # Validar profundidad
            current_tvec = tvec.flatten()
            if current_tvec[2] <= 0.0:
                now = time.time()
                if not hasattr(self, '_last_z_warn_time'):
                    self._last_z_warn_time = 0.0
                if now - self._last_z_warn_time >= 1.0:
                    self._last_z_warn_time = now
                    self.get_logger().warn("[DETECTOR ARUCO] Pose rechazada: marcador detras de la camara o Z invalida.")
                
                # Mostrar ventana debug en modo original si corresponde
                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)
                    
                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # Calcular error de reproyección
            projected_points, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            projected_points = projected_points.reshape((4, 2))
            
            # Error medio de reproyección en pixeles
            reproj_error = float(np.mean(np.linalg.norm(image_points - projected_points, axis=1)))
            self.last_reprojection_error_px = reproj_error

            if self.reject_if_ambiguous and reproj_error > self.max_reprojection_error_px:
                now = time.time()
                if not hasattr(self, '_last_reproj_warn_time'):
                    self._last_reproj_warn_time = 0.0
                if now - self._last_reproj_warn_time >= 1.0:
                    self._last_reproj_warn_time = now
                    self.get_logger().warn("[DETECTOR ARUCO] Error de reproyeccion alto; se rechaza deteccion.")
                
                # Mostrar ventana debug en modo original si corresponde
                if self.show_main_window:
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)
                    
                    titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                    cv2.imshow(self.main_window_name, cv_image)
                    cv2.setWindowTitle(self.main_window_name, titulo)
                    cv2.waitKey(1)
                return

            # Estabilización temporal
            # Posicion
            if self.last_tvec is None:
                self.last_tvec = current_tvec.copy()
            else:
                self.last_tvec = (1.0 - self.alpha_pos) * self.last_tvec + self.alpha_pos * current_tvec

            # Orientación
            rmat, _ = cv2.Rodrigues(rvec)
            rot = R.from_matrix(rmat)
            current_quat = rot.as_quat() # [x, y, z, w]

            if self.last_quat is None:
                self.last_quat = current_quat.copy()
            else:
                self.last_quat = self.interpolate_quaternion(self.last_quat, current_quat, self.alpha_rot)

            # Publicar
            self.publish_aruco_tf_and_pose(
                self.last_tvec,
                self.last_quat,
                msg.header.stamp,
                source_frame=msg.header.frame_id,
                reprojection_error_px=self.last_reprojection_error_px
            )

            # Interfaz Visual
            if self.show_main_window:
                # Modo normal: dibujar contorno del ArUco detectado y el centro del marcador
                pts_cnt = image_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(cv_image, [pts_cnt], True, (0, 255, 0), 2) # Contorno en verde
                
                # Calcular centro en imagen
                cx_img = int(np.mean(image_points[:, 0]))
                cy_img = int(np.mean(image_points[:, 1]))
                cv2.circle(cv_image, (cx_img, cy_img), 5, (0, 0, 255), -1) # Centro en rojo

                if self.show_debug_overlay:
                    # Mostrar ejes 3D
                    if self.show_axes:
                        if hasattr(cv2, 'drawFrameAxes'):
                            cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.axis_length_m)
                        elif hasattr(cv2.aruco, 'drawAxis'):
                            cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.axis_length_m)
                    
                    # Dibujar HUD debug
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)
                    
                    err_val = f"{self.last_reprojection_error_px:.2f}" if self.last_reprojection_error_px is not None else "N/A"
                    frame_id_res = self.camera_frame if self.camera_frame else msg.header.frame_id
                    
                    # Textos del HUD traducidos sin acentos
                    cv2.putText(cv_image, f"ID marcador: {self.marker_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Z [m]: {self.last_tvec[2]:.3f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Error reproyeccion [px]: {err_val}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"FPS camara: {self.camera_fps_estimate:.1f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Proc [ms]: {self.last_processing_time_ms:.1f}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frames perdidos: {self.lost_frame_count}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frame camara: {frame_id_res}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(cv_image, f"Frame marcador: {self.marker_frame}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # En modo normal, igual calcular FPS
                    proc_ms = (time.perf_counter() - start_proc) * 1000.0
                    self.update_fps_estimate(msg.header.stamp, proc_ms)

                titulo = f"{self.marker_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
                cv2.imshow(self.main_window_name, cv_image)
                cv2.setWindowTitle(self.main_window_name, titulo)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error en image_callback: {e}")

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
