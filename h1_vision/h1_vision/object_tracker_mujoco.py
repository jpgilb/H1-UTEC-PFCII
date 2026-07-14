#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracker RGB-D del cubo de manipulación para el Unitree H1-2.

Responsabilidades principales:
- segmentar por color la cara visible del cubo en el espacio HSV;
- obtener puntos 3D mediante la profundidad y los intrínsecos de la cámara;
- calcular el centro medio y la normal de la cara visible;
- construir una orientación combinando PCA y la proyección 3D de las esquinas detectadas;
- suavizar temporalmente el centro de la cara y la normal estimada;
- calcular el centro geométrico del cubo y publicar su pose mediante TF y PoseStamped.

La pose se publica en el marco óptico de la cámara. Su transformación hacia
el marco del robot se realiza posteriormente mediante TF2 en el supervisor.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
from collections import deque

class VisionNode(Node):
    def __init__(self):
        # ============================================================
        # Declaración y lectura de parámetros
        # ============================================================
        super().__init__('vision_node')

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('object_frame', 'objeto_cubo')
        self.declare_parameter('object_dimension', 0.055)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', '/vision/objeto_cubo/pose')

        self.declare_parameter('color_topic', '/camera/color')
        self.declare_parameter('depth_topic', '/camera/depth')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')

        self.declare_parameter('object_yaw_correction_deg', 0.0)

        self.declare_parameter('debug_visual', False)
        self.declare_parameter('use_gui_trackbars', False)
        self.declare_parameter('show_hsv_mask_debug', False)
        self.declare_parameter('show_depth_mask_debug', False)
        self.declare_parameter('show_debug_overlay', False)
        self.declare_parameter('expected_camera_fps', 30.0)
        self.declare_parameter('fps_window_size', 30)
        self.declare_parameter('h_min', 50)
        self.declare_parameter('s_min', 100)
        self.declare_parameter('v_min', 0)
        self.declare_parameter('h_max', 132)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)

        self.camera_frame = str(self.get_parameter('camera_frame').value).strip()
        self.object_frame = str(self.get_parameter('object_frame').value).strip()
        self.object_dimension = float(self.get_parameter('object_dimension').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.publish_pose = bool(self.get_parameter('publish_pose').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value).strip()
        self.color_topic = str(self.get_parameter('color_topic').value).strip()
        self.depth_topic = str(self.get_parameter('depth_topic').value).strip()
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value).strip()
        self.object_yaw_correction_deg = float(self.get_parameter('object_yaw_correction_deg').value)

        self.debug_visual = bool(self.get_parameter('debug_visual').value)
        self.use_gui_trackbars = bool(self.get_parameter('use_gui_trackbars').value)
        self.show_hsv_mask_debug = bool(self.get_parameter('show_hsv_mask_debug').value)
        self.show_depth_mask_debug = bool(self.get_parameter('show_depth_mask_debug').value)
        self.show_debug_overlay = bool(self.get_parameter('show_debug_overlay').value)
        self.expected_camera_fps = float(self.get_parameter('expected_camera_fps').value)
        self.fps_window_size = int(self.get_parameter('fps_window_size').value)
        self.h_min = int(self.get_parameter('h_min').value)
        self.s_min = int(self.get_parameter('s_min').value)
        self.v_min = int(self.get_parameter('v_min').value)
        self.h_max = int(self.get_parameter('h_max').value)
        self.s_max = int(self.get_parameter('s_max').value)
        self.v_max = int(self.get_parameter('v_max').value)

        if self.debug_visual:
            self.use_gui_trackbars = True
            self.show_hsv_mask_debug = True
            self.show_depth_mask_debug = True
            self.show_debug_overlay = True

        if self.object_dimension <= 0.0:
            raise ValueError("object_dimension debe ser mayor que 0.0")
        if self.expected_camera_fps <= 0.0:
            raise ValueError("expected_camera_fps debe ser mayor que 0.0")
        if self.fps_window_size <= 0:
            raise ValueError("fps_window_size debe ser mayor que 0")

        self.main_window_name = self.object_frame
        self.hsv_control_window_name = f"Ajuste HSV - {self.object_frame}"

        self.frame_timestamps = deque(maxlen=self.fps_window_size)
        self.last_processing_time_ms = 0.0
        self.camera_fps_estimate = 0.0

        # ============================================================
        # Publicación de TF, PoseStamped y configuración de depuración visual
        # ============================================================
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.last_pose_log_time = 0.0

        self.get_logger().info(
            f"[DETECTOR DE CUBO INICIALIZADO]\n"
            f"  Frame de cámara: '{self.camera_frame}'\n"
            f"  Frame del objeto: '{self.object_frame}'\n"
            f"  Dimensión del objeto [m]: {self.object_dimension:.4f}\n"
            f"  Publicar TF: {self.publish_tf}\n"
            f"  Publicar PoseStamped: {self.publish_pose}\n"
            f"  Tópico de pose: '{self.pose_topic}'\n"
            f"  Tópico color: '{self.color_topic}'\n"
            f"  Tópico depth: '{self.depth_topic}'\n"
            f"  Tópico CameraInfo: '{self.camera_info_topic}'\n"
            f"  Corrección yaw cubo [deg]: {self.object_yaw_correction_deg:.2f}\n"
            f"  Debug visual: {self.debug_visual}\n"
            f"  Trackbars HSV: {self.use_gui_trackbars}"
        )

        # ============================================================
        # Suscripciones RGB, profundidad y CameraInfo
        # ============================================================
        self.subscription_color = self.create_subscription(Image, self.color_topic, self.image_callback, 10)
        self.subscription_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.latest_depth_encoding = ""

        self.intrinsics_loaded = False
        self.fx = self.fy = self.cx = self.cy = 0.0

        self.last_tvec = np.zeros(3)
        self.stable_normal = None
        self.alpha_pos = 0.2
        self.alpha_normal = 0.1
        self.alpha_yaw = 0.4
        self.last_time = time.time()

        if self.use_gui_trackbars:
            self.setup_trackbars()
        self.get_logger().info("Detector de cubo listo. Publicacion TF/Pose activa.")

    def setup_trackbars(self):
        cv2.namedWindow(self.hsv_control_window_name)
        cv2.createTrackbar("Min H", self.hsv_control_window_name, self.h_min, 179, lambda x: None)
        cv2.createTrackbar("Min S", self.hsv_control_window_name, self.s_min, 255, lambda x: None)
        cv2.createTrackbar("Min V", self.hsv_control_window_name, self.v_min, 255, lambda x: None)
        cv2.createTrackbar("Max H", self.hsv_control_window_name, self.h_max, 179, lambda x: None)
        cv2.createTrackbar("Max S", self.hsv_control_window_name, self.s_max, 255, lambda x: None)
        cv2.createTrackbar("Max V", self.hsv_control_window_name, self.v_max, 255, lambda x: None)

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
    # Publicación de la pose estimada en el marco óptico
    # ============================================================
    def publish_cube_tf_and_pose(self, face_center, normal, quat, stamp, source_frame=""):
        if face_center is None or normal is None or quat is None:
            return

        face_center = np.array(face_center, dtype=float)
        normal = np.array(normal, dtype=float)
        quat = np.array(quat, dtype=float)

        if np.any(np.isnan(face_center)) or np.any(np.isinf(face_center)):
            return
        if np.any(np.isnan(normal)) or np.any(np.isinf(normal)):
            return
        if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
            return

        frame_id = self.camera_frame if self.camera_frame else source_frame
        if not frame_id:
            self.get_logger().warn("[DETECTOR CUBO] Frame vacío; se omite publicación TF/Pose.")
            return

        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-6:
            return
        normal = normal / norm_normal

        norm_quat = np.linalg.norm(quat)
        if norm_quat < 1e-6:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat = quat / norm_quat

        # El centro geométrico se estima desplazando el centro de la cara
        # media dimensión en sentido opuesto a la normal.
        cube_center = face_center - normal * (self.object_dimension / 2.0)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = frame_id
            t.child_frame_id = self.object_frame
            t.transform.translation.x = cube_center[0]
            t.transform.translation.y = cube_center[1]
            t.transform.translation.z = cube_center[2]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            self.tf_broadcaster.sendTransform(t)

        if self.publish_pose:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = cube_center[0]
            pose_msg.pose.position.y = cube_center[1]
            pose_msg.pose.position.z = cube_center[2]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            self.pose_pub.publish(pose_msg)

        now = time.time()
        if now - self.last_pose_log_time >= 1.0:
            self.last_pose_log_time = now
            self.get_logger().info(
                f"POSE CUBO -> centro_cara={face_center.tolist()} "
                f"centro_cubo={cube_center.tolist()} "
                f"normal={normal.tolist()} "
                f"frame={frame_id} "
                f"objeto={self.object_frame}"
            )

    # ============================================================
    # Recepción y almacenamiento de intrínsecos de la cámara
    # ============================================================
    def camera_info_callback(self, msg):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.intrinsics_loaded = True
        self.get_logger().info("Intrínsecos cargados.")
        self.destroy_subscription(self.info_sub)

    def depth_to_meters(self, z_raw):
        try:
            z = float(z_raw)
        except Exception:
            return None

        if not np.isfinite(z) or z <= 0.0:
            return None

        if self.latest_depth_encoding == '32FC1':
            return z

        if self.latest_depth_encoding in ('16UC1', 'mono16'):
            return z / 1000.0

        if z > 20.0:
            return z / 1000.0
        return z

    # ============================================================
    # Desproyección de píxeles hacia puntos 3D
    # ============================================================
    def deproject(self, u, v, z_raw):
        z_m = self.depth_to_meters(z_raw)
        if z_m is None:
            return None
        return [(u - self.cx) * z_m / self.fx, (v - self.cy) * z_m / self.fy, z_m]

    # ============================================================
    # Filtrado de valores de profundidad inválidos
    # ============================================================
    def valid_depth(self, z_raw):
        z_m = self.depth_to_meters(z_raw)
        return z_m is not None and 0.05 <= z_m <= 5.0

    def local_yaw_correction_matrix(self):
        theta = np.deg2rad(self.object_yaw_correction_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    def ray_from_pixel(self, u, v):
        """Rayo 3D en frame de cámara a partir de pixel usando intrínsecos."""
        if abs(self.fx) < 1e-9 or abs(self.fy) < 1e-9:
            return None
        return np.array([
            (float(u) - self.cx) / self.fx,
            (float(v) - self.cy) / self.fy,
            1.0
        ], dtype=float)

    def intersect_ray_with_plane(self, u, v, plane_point, plane_normal):
        """Intersección del rayo de cámara con el plano estimado de la cara visible."""
        ray = self.ray_from_pixel(u, v)
        if ray is None:
            return None

        plane_point = np.asarray(plane_point, dtype=float)
        plane_normal = np.asarray(plane_normal, dtype=float)

        denom = float(np.dot(plane_normal, ray))
        if abs(denom) < 1e-9:
            return None

        t = float(np.dot(plane_normal, plane_point) / denom)
        if not np.isfinite(t) or t <= 0.0:
            return None

        p = t * ray
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            return None
        return p

    # ============================================================
    # Estimación del eje X a partir de la geometría del contorno
    # ============================================================
    def estimate_x_axis_from_projected_corners(self, corners_2d, plane_point, plane_normal):
        """
        Estima el eje X del cubo usando las esquinas 2D proyectadas al plano 3D.

        Mejora respecto al yaw 2D:
        - TL/TR/BR/BL ya vienen ordenadas por sort_corners().
        - Cada esquina 2D se intersecta con el plano 3D de la cara.
        - El eje X se obtiene promediando el borde superior y el borde inferior.
        """
        try:
            pts_3d = []
            for pt in corners_2d:
                p = self.intersect_ray_with_plane(pt[0], pt[1], plane_point, plane_normal)
                if p is None:
                    return None
                pts_3d.append(p)

            pts_3d = np.asarray(pts_3d, dtype=float)
            p_tl, p_tr, p_br, p_bl = pts_3d

            edge_top = p_tr - p_tl
            edge_bottom = p_br - p_bl
            x_raw = edge_top + edge_bottom

            n = float(np.linalg.norm(x_raw))
            if not np.isfinite(n) or n < 1e-9:
                return None

            return x_raw / n

        except Exception as exc:
            self.get_logger().warn(f"[CUBE ORIENTATION 3D] Falló estimación 3D de eje X: {exc}")
            return None

    # ============================================================
    # Conversión de matriz de rotación a cuaternión
    # ============================================================
    def rotation_matrix_to_quaternion(self, m):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            return [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            return [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            return [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            return [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]

    def sort_corners(self, pts):
        pts = pts.reshape((4, 2))
        new_pts = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        new_pts[0] = pts[np.argmin(s)]
        new_pts[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        new_pts[1] = pts[np.argmin(diff)]
        new_pts[3] = pts[np.argmax(diff)]
        return new_pts

    # ============================================================
    # Recepción y almacenamiento de la última imagen de profundidad
    # ============================================================
    def depth_callback(self, msg):
        self.latest_depth_encoding = msg.encoding
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ============================================================
    # Procesamiento del frame RGB con la profundidad disponible
    # ============================================================
    # La profundidad se recibe de forma asíncrona y se utiliza la última
    # imagen almacenada. No existe sincronización explícita por timestamp.
    def image_callback(self, msg):
        if not self.intrinsics_loaded or self.latest_depth_image is None: return

        start_proc = time.perf_counter()
        try:
            # Conversión y segmentación del frame mediante HSV.
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.use_gui_trackbars:
                l_h = cv2.getTrackbarPos("Min H", self.hsv_control_window_name)
                l_s = cv2.getTrackbarPos("Min S", self.hsv_control_window_name)
                l_v = cv2.getTrackbarPos("Min V", self.hsv_control_window_name)
                u_h = cv2.getTrackbarPos("Max H", self.hsv_control_window_name)
                u_s = cv2.getTrackbarPos("Max S", self.hsv_control_window_name)
                u_v = cv2.getTrackbarPos("Max V", self.hsv_control_window_name)
            else:
                l_h, l_s, l_v = self.h_min, self.s_min, self.v_min
                u_h, u_s, u_v = self.h_max, self.s_max, self.v_max
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))
            # Apertura morfológica y selección del contorno dominante.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.show_hsv_mask_debug:
                cv2.imshow(f"{self.object_frame} - mascara HSV", mask)

            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(best_cnt) > 500:
                    peri = cv2.arcLength(best_cnt, True)
                    approx = cv2.approxPolyDP(best_cnt, 0.04 * peri, True)

                    if len(approx) == 4:

                        cv2.polylines(cv_image, [approx], True, (0, 255, 0), 2)
                        # Aproximación cuadrilateral y referencia visual del borde superior.
                        corners_2d = self.sort_corners(approx)
                        for pt in corners_2d:
                            cv2.circle(cv_image, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

                        v_x_2d = corners_2d[1] - corners_2d[0]
                        angle_2d = np.rad2deg(np.arctan2(v_x_2d[1], v_x_2d[0]))
                        rad = np.deg2rad(angle_2d)
                        visual_yaw_vec = np.array([np.cos(rad), np.sin(rad), 0.0])

                        # Muestreo 3D dentro del contorno y cálculo del centro de la cara.
                        mask_cnt = np.zeros(mask.shape, dtype=np.uint8)
                        cv2.drawContours(mask_cnt, [best_cnt], -1, 255, -1)
                        indices = np.argwhere(mask_cnt == 255)[::15]
                        pts_3d = []
                        for v, u in indices:
                            z = self.latest_depth_image[v, u]
                            if self.valid_depth(z):
                                p = self.deproject(u, v, z)
                                if p: pts_3d.append(p)

                        if len(pts_3d) > 15:
                            pts_3d = np.array(pts_3d)

                            # La media de la nube representa el centro estimado de la cara visible,
                            # no todavía el centro geométrico completo del cubo.
                            mean_center = np.mean(pts_3d, axis=0)

                            # PCA para estimar la normal y seleccionar su sentido hacia la cámara.
                            # La nube se centra antes de calcular su covarianza.
                            # El autovector con menor autovalor representa la dirección de menor
                            # dispersión y se utiliza como normal de la cara visible.
                            cov = np.cov((pts_3d - mean_center).T)
                            evals, evecs = np.linalg.eigh(cov)
                            raw_normal = evecs[:, np.argmin(evals)]

                            # La normal obtenida por PCA puede aparecer con cualquiera de sus signos.
                            # Se selecciona el sentido dirigido hacia la cámara.
                            if raw_normal[2] > 0: raw_normal = -raw_normal

                            # Suavizado temporal de la normal estimada.
                            if self.stable_normal is None: self.stable_normal = raw_normal
                            else: self.stable_normal = (1 - self.alpha_normal) * self.stable_normal + self.alpha_normal * raw_normal
                            self.stable_normal /= np.linalg.norm(self.stable_normal)

                            z_axis = self.stable_normal

                            # El eje X principal se obtiene proyectando las esquinas detectadas
                            # sobre el plano 3D de la cara. La dirección 2D se conserva como fallback.
                            x_axis_raw_3d = self.estimate_x_axis_from_projected_corners(
                                corners_2d=corners_2d,
                                plane_point=mean_center,
                                plane_normal=z_axis
                            )

                            if x_axis_raw_3d is None:
                                x_axis_raw = visual_yaw_vec
                                self.get_logger().warn(
                                    "[CUBE ORIENTATION 3D CORNER FIX] Fallback a yaw 2D TL->TR."
                                )
                            else:
                                x_axis_raw = x_axis_raw_3d

                            # Construcción de la base ortonormal y corrección local del frame.
                            # Se elimina del eje X su componente paralela a la normal y luego
                            # se normaliza para formar una base ortogonal.
                            # x_ortogonal = x_inicial - dot(x_inicial, z) * z
                            x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
                            x_norm = np.linalg.norm(x_axis)
                            if x_norm < 1e-9:
                                x_axis = visual_yaw_vec - np.dot(visual_yaw_vec, z_axis) * z_axis
                                x_norm = np.linalg.norm(x_axis)

                            x_axis /= max(x_norm, 1e-9)

                            y_axis = np.cross(z_axis, x_axis)
                            y_axis /= max(np.linalg.norm(y_axis), 1e-9)

                            R_final = np.column_stack((x_axis, y_axis, z_axis))

                            R_publish = R_final @ self.local_yaw_correction_matrix()

                            # Suavizado temporal del centro de la cara antes de publicar.
                            self.last_tvec = (1 - self.alpha_pos) * self.last_tvec + self.alpha_pos * mean_center
                            quat = self.rotation_matrix_to_quaternion(R_publish)

                            self.publish_cube_tf_and_pose(
                                face_center=self.last_tvec,
                                normal=self.stable_normal,
                                quat=quat,
                                stamp=msg.header.stamp,
                                source_frame=msg.header.frame_id
                            )

                            rvec_draw, _ = cv2.Rodrigues(R_publish)
                            # Actualización de la visualización de depuración.
                            cv2.drawFrameAxes(cv_image, np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), None, rvec_draw, self.last_tvec, 0.05)

                            if self.show_debug_overlay:
                                cube_center_pos = self.last_tvec - self.stable_normal * (self.object_dimension / 2.0)
                                cv2.putText(cv_image, f"Centro cara: {[round(c, 3) for c in self.last_tvec.tolist()]} m", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Centro cubo: {[round(c, 3) for c in cube_center_pos.tolist()]} m", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Profundidad: {self.last_tvec[2]:.3f} m", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"FPS camara: {self.camera_fps_estimate:.1f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Tiempo procesamiento: {self.last_processing_time_ms:.1f} ms", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            processing_time_ms = (time.perf_counter() - start_proc) * 1000.0
            self.update_fps_estimate(msg.header.stamp, processing_time_ms)

            title = f"{self.object_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
            cv2.imshow(self.main_window_name, cv_image)
            cv2.setWindowTitle(self.main_window_name, title)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error en image_callback: {e}")


# ============================================================
# Punto de entrada del nodo
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
