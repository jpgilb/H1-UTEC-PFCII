#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Remanente para retrocompatibilidad, no se usa para control principal
SHOW_HSV_MASK_DEBUG = False
SHOW_DEPTH_MASK_DEBUG = False

class SphereTrackerNode(Node):
    def __init__(self):
        super().__init__('sphere_tracker_node')
        
        # Declarar y cargar parámetros
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('object_frame', 'objeto_esfera')
        self.declare_parameter('object_dimension', 0.055)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', '/vision/objeto_esfera/pose')
        self.declare_parameter('min_area', 300)
        self.declare_parameter('use_gui_trackbars', False)
        self.declare_parameter('alpha_pos', 0.2)
        self.declare_parameter('min_valid_depth_points', 10) # default: 10
        self.declare_parameter('circularity_threshold', 0.45) # default: 0.45
        self.declare_parameter('centroid_depth_window_px', 15) # default: 15
        self.declare_parameter('h_min', 1)
        self.declare_parameter('s_min', 93)
        self.declare_parameter('v_min', 35)
        self.declare_parameter('h_max', 30)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)
        self.declare_parameter('min_radius_px', 8)
        self.declare_parameter('max_radius_px', 200)
        self.declare_parameter('min_fill_ratio', 0.35)
        self.declare_parameter('erosion_iterations', 1)
        
        # Filtro Temporal y ROI
        self.declare_parameter('use_temporal_filter', True)
        self.declare_parameter('max_2d_jump_px', 80.0)
        self.declare_parameter('max_3d_jump_m', 0.10)
        self.declare_parameter('lost_frame_tolerance', 3)
        self.declare_parameter('use_temporal_roi', True)
        self.declare_parameter('roi_margin_px', 100)
        self.declare_parameter('publish_last_valid_during_short_loss', False)
        self.declare_parameter('reset_roi_after_lost', True)

        # Nuevos parámetros de depuración y FPS
        self.declare_parameter('debug_visual', False)
        self.declare_parameter('show_hsv_mask_debug', False)
        self.declare_parameter('show_depth_mask_debug', False)
        self.declare_parameter('show_debug_overlay', False)
        self.declare_parameter('expected_camera_fps', 30.0)
        self.declare_parameter('fps_window_size', 30)

        self.color_topic = str(self.get_parameter('color_topic').value).strip()
        self.depth_topic = str(self.get_parameter('depth_topic').value).strip()
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value).strip()
        self.camera_frame = str(self.get_parameter('camera_frame').value).strip()
        self.object_frame = str(self.get_parameter('object_frame').value).strip()
        self.object_dimension = float(self.get_parameter('object_dimension').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.publish_pose = bool(self.get_parameter('publish_pose').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value).strip()
        self.min_area = int(self.get_parameter('min_area').value)
        self.use_gui_trackbars = bool(self.get_parameter('use_gui_trackbars').value)
        self.alpha_pos = float(self.get_parameter('alpha_pos').value)
        self.min_valid_depth_points = int(self.get_parameter('min_valid_depth_points').value)
        self.circularity_threshold = float(self.get_parameter('circularity_threshold').value)
        self.centroid_depth_window_px = int(self.get_parameter('centroid_depth_window_px').value)
        self.h_min = int(self.get_parameter('h_min').value)
        self.s_min = int(self.get_parameter('s_min').value)
        self.v_min = int(self.get_parameter('v_min').value)
        self.h_max = int(self.get_parameter('h_max').value)
        self.s_max = int(self.get_parameter('s_max').value)
        self.v_max = int(self.get_parameter('v_max').value)
        self.min_radius_px = int(self.get_parameter('min_radius_px').value)
        self.max_radius_px = int(self.get_parameter('max_radius_px').value)
        self.min_fill_ratio = float(self.get_parameter('min_fill_ratio').value)
        self.erosion_iterations = int(self.get_parameter('erosion_iterations').value)

        self.use_temporal_filter = bool(self.get_parameter('use_temporal_filter').value)
        self.max_2d_jump_px = float(self.get_parameter('max_2d_jump_px').value)
        self.max_3d_jump_m = float(self.get_parameter('max_3d_jump_m').value)
        self.lost_frame_tolerance = int(self.get_parameter('lost_frame_tolerance').value)
        self.use_temporal_roi = bool(self.get_parameter('use_temporal_roi').value)
        self.roi_margin_px = int(self.get_parameter('roi_margin_px').value)
        self.publish_last_valid_during_short_loss = bool(self.get_parameter('publish_last_valid_during_short_loss').value)
        self.reset_roi_after_lost = bool(self.get_parameter('reset_roi_after_lost').value)

        self.debug_visual = bool(self.get_parameter('debug_visual').value)
        self.show_hsv_mask_debug = bool(self.get_parameter('show_hsv_mask_debug').value)
        self.show_depth_mask_debug = bool(self.get_parameter('show_depth_mask_debug').value)
        self.show_debug_overlay = bool(self.get_parameter('show_debug_overlay').value)
        self.expected_camera_fps = float(self.get_parameter('expected_camera_fps').value)
        self.fps_window_size = int(self.get_parameter('fps_window_size').value)

        # Regla de activación del modo debug visual
        if self.debug_visual:
            self.use_gui_trackbars = True
            self.show_hsv_mask_debug = True
            self.show_depth_mask_debug = True
            self.show_debug_overlay = True

        # Validación
        if self.object_dimension <= 0.0:
            raise ValueError("object_dimension debe ser mayor que 0.0")
        if self.min_area <= 0:
            raise ValueError("min_area debe ser mayor que 0")
        if self.max_2d_jump_px <= 0.0:
            raise ValueError("max_2d_jump_px debe ser mayor que 0.0")
        if self.max_3d_jump_m <= 0.0:
            raise ValueError("max_3d_jump_m debe ser mayor que 0.0")
        if self.lost_frame_tolerance < 0:
            raise ValueError("lost_frame_tolerance debe ser mayor o igual que 0")
        if self.roi_margin_px <= 0:
            raise ValueError("roi_margin_px debe ser mayor que 0")
        if self.expected_camera_fps <= 0.0:
            raise ValueError("expected_camera_fps debe ser mayor que 0.0")
        if self.fps_window_size <= 0:
            raise ValueError("fps_window_size debe ser mayor que 0")

        # Nombres de ventana estandarizados
        self.main_window_name = self.object_frame
        self.hsv_control_window_name = f"Ajuste HSV - {self.object_frame}"

        # Suscripciones
        self.subscription_color = self.create_subscription(Image, self.color_topic, self.image_callback, 10)
        self.subscription_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
            
        self.bridge = CvBridge()
        self.latest_depth_image = None
        
        # Parámetros de Cámara
        self.intrinsics_loaded = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        
        # Estabilización temporal y estado de filtro
        self.last_tvec = None
        self.last_time = time.time()
        self.last_pose_log_time = 0.0

        # Estado de FPS
        self.frame_timestamps = deque(maxlen=self.fps_window_size)
        self.last_processing_time_ms = 0.0
        self.camera_fps_estimate = 0.0

        self.last_valid_center_2d = None
        self.last_valid_center_3d = None
        self.last_valid_stamp = None
        self.lost_frame_count = 0
        self.temporal_reject_count = 0
        self.last_jump_2d = None
        self.last_jump_3d = None
        
        # Publicadores
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        # Setup GUI Control Trackbars
        if self.use_gui_trackbars:
            self.setup_trackbars()

        self.get_logger().info("Nodo de deteccion de esfera iniciado")
        self.get_logger().info(
            f"[DETECTOR DE ESFERA INICIALIZADO]\n"
            f"  Frame de cámara: '{self.camera_frame}'\n"
            f"  Frame del objeto: '{self.object_frame}'\n"
            f"  Diámetro del objeto [m]: {self.object_dimension:.4f}\n"
            f"  Publicar TF: {self.publish_tf}\n"
            f"  Publicar PoseStamped: {self.publish_pose}\n"
            f"  Tópico de pose: '{self.pose_topic}'\n"
            f"  Área mínima: {self.min_area}\n"
            f"  Umbral de circularidad: {self.circularity_threshold:.4f}\n"
            f"  Rango de radio [px]: [{self.min_radius_px}, {self.max_radius_px}]\n"
            f"  Razón mínima de relleno: {self.min_fill_ratio:.4f}\n"
            f"  Filtro temporal: {self.use_temporal_filter}\n"
            f"  ROI temporal: {self.use_temporal_roi}\n"
            f"  Debug visual: {self.debug_visual}\n"
            f"  Trackbars HSV: {self.use_gui_trackbars}"
        )

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

    def camera_info_callback(self, msg):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.intrinsics_loaded = True
        self.get_logger().info("Intrinsecos cargados.")
        self.destroy_subscription(self.info_sub)

    def normalize_depth_to_meters(self, z_raw):
        if isinstance(z_raw, np.ndarray):
            if z_raw.dtype == np.uint16:
                return z_raw.astype(float) / 1000.0
            else:
                z_float = z_raw.astype(float)
                positives = z_float[z_float > 0]
                if len(positives) > 0 and np.median(positives) > 10.0:
                    return z_float / 1000.0
                return z_float
        else:
            if isinstance(z_raw, (int, np.uint16, np.integer)):
                return float(z_raw) / 1000.0
            else:
                z_val = float(z_raw)
                if z_val > 10.0:
                    return z_val / 1000.0
                return z_val

    def deproject(self, u, v, z_m):
        if z_m <= 0: return None
        return [(u - self.cx) * z_m / self.fx, (v - self.cy) * z_m / self.fy, z_m]

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def select_best_sphere_contour(self, contours):
        if not contours:
            return None
        
        best_cnt = None
        best_circularity = 0.0
        best_fill_ratio = 0.0
        best_radius_enc = 0.0
        best_cx = None
        best_cy = None
        best_score = -1e9
        
        # Determinar si la restricción ROI está activa
        roi_active = (
            self.use_temporal_roi and 
            self.last_valid_center_2d is not None and 
            (self.lost_frame_count <= self.lost_frame_tolerance or not self.reset_roi_after_lost)
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.circularity_threshold:
                continue
            hull = cv2.convexHull(contour)
            (x_enc, y_enc), radius_enc = cv2.minEnclosingCircle(hull)
            if not (self.min_radius_px <= radius_enc <= self.max_radius_px):
                continue
            circle_area = np.pi * radius_enc * radius_enc
            fill_ratio = area / circle_area if circle_area > 1e-6 else 0.0
            if fill_ratio < self.min_fill_ratio:
                continue
            
            cx_candidate = float(x_enc)
            cy_candidate = float(y_enc)
            candidate_center_2d = np.array([cx_candidate, cy_candidate], dtype=float)
            
            score = area
            if roi_active:
                dist_2d = float(np.linalg.norm(candidate_center_2d - self.last_valid_center_2d))
                if dist_2d > self.roi_margin_px:
                    continue
                else:
                    score = area + 1000.0 - 5.0 * dist_2d
            
            if score > best_score:
                best_score = score
                best_cnt = contour
                best_circularity = circularity
                best_fill_ratio = fill_ratio
                best_radius_enc = radius_enc
                best_cx = cx_candidate
                best_cy = cy_candidate
                
        if best_cnt is None:
            return None
        return best_cnt, best_circularity, best_fill_ratio, best_radius_enc, best_cx, best_cy

    def accept_temporal_candidate(self, center_2d, center_3d):
        if not self.use_temporal_filter:
            return True
        if self.reset_roi_after_lost and self.lost_frame_count > self.lost_frame_tolerance:
            self.last_jump_2d = None
            self.last_jump_3d = None
            return True
        if self.last_valid_center_2d is None or self.last_valid_center_3d is None:
            return True
            
        jump_2d = float(np.linalg.norm(center_2d - self.last_valid_center_2d))
        jump_3d = float(np.linalg.norm(center_3d - self.last_valid_center_3d))
        self.last_jump_2d = jump_2d
        self.last_jump_3d = jump_3d
        
        if jump_2d > self.max_2d_jump_px:
            self.temporal_reject_count += 1
            now = time.time()
            if not hasattr(self, '_last_reject_log_time'):
                self._last_reject_log_time = 0.0
            if now - self._last_reject_log_time >= 1.0:
                self._last_reject_log_time = now
                self.get_logger().warn(
                    f"[FILTRO ESFERA] Rechazo por salto 2D excesivo: {jump_2d:.2f}px (max {self.max_2d_jump_px:.1f}px)"
                )
            return False
            
        if jump_3d > self.max_3d_jump_m:
            self.temporal_reject_count += 1
            now = time.time()
            if not hasattr(self, '_last_reject_log_time'):
                self._last_reject_log_time = 0.0
            if now - self._last_reject_log_time >= 1.0:
                self._last_reject_log_time = now
                self.get_logger().warn(
                    f"[FILTRO ESFERA] Rechazo por salto 3D excesivo: {jump_3d:.4f}m (max {self.max_3d_jump_m:.3f}m)"
                )
            return False
            
        return True

    def handle_lost_detection(self, stamp, source_frame=""):
        self.lost_frame_count += 1
        if self.publish_last_valid_during_short_loss:
            if self.lost_frame_count <= self.lost_frame_tolerance and self.last_valid_center_3d is not None:
                self.publish_sphere_tf_and_pose(
                    self.last_valid_center_3d,
                    stamp,
                    source_frame=source_frame,
                    circularity=None
                )

    def publish_sphere_tf_and_pose(self, center, stamp, source_frame="", circularity=None):
        if center is None:
            return

        center = np.array(center, dtype=float)
        if np.any(np.isnan(center)) or np.any(np.isinf(center)):
            return

        frame_id = self.camera_frame if self.camera_frame else source_frame
        if not frame_id:
            self.get_logger().warn("[DETECTOR ESFERA] Frame vacio; se omite publicacion TF/Pose.")
            return

        # Publicar TF (identidad quaternion)
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = frame_id
            t.child_frame_id = self.object_frame
            t.transform.translation.x = center[0]
            t.transform.translation.y = center[1]
            t.transform.translation.z = center[2]
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

        # Publicar Pose (identidad quaternion)
        if self.publish_pose:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = center[0]
            pose_msg.pose.position.y = center[1]
            pose_msg.pose.position.z = center[2]
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            self.pose_pub.publish(pose_msg)

        # Logs Throttled (1 por segundo)
        now = time.time()
        if now - self.last_pose_log_time >= 1.0:
            self.last_pose_log_time = now
            z_surface = center[2] - (self.object_dimension / 2.0)
            circ_str = f"{circularity:.4f}" if circularity is not None else "None"
            self.get_logger().info(
                f"POSE ESFERA -> centro={center.tolist()} frame={frame_id} objeto={self.object_frame} "
                f"circularidad={circ_str} z_superficie={z_surface:.4f}"
            )

    def image_callback(self, msg):
        if not self.intrinsics_loaded or self.latest_depth_image is None: return
        
        start_proc = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Segmentacion HSV
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
            
            # Morfologia robusta
            kernel_open = np.ones((3, 3), np.uint8)
            kernel_close = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            mask = cv2.medianBlur(mask, 5)

            # Validar dimensiones RGB-D
            depth_shape = self.latest_depth_image.shape[:2]
            mask_shape = mask.shape[:2]
            
            if depth_shape != mask_shape:
                now = time.time()
                if not hasattr(self, '_last_shape_warn_time'):
                    self._last_shape_warn_time = 0.0
                if now - self._last_shape_warn_time >= 5.0:
                    self._last_shape_warn_time = now
                    self.get_logger().warn("[DETECTOR ESFERA] Las dimensiones de la mascara RGB y la de profundidad difieren; verificar alineacion RGB-D.")

            if self.show_hsv_mask_debug:
                cv2.imshow(f"{self.object_frame} - mascara HSV", mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Inicializar variables para depuracion visual y profundidad
            depth_source = "None"
            valid_window_pts_count = 0
            valid_mask_pts_count = 0
            radius_enc = 0.0
            fill_ratio = 0.0
            z_m = 0.0
            cx = cy = None
            x_enc = y_enc = 0.0
            best_cnt = None
            circularity = None

            # Crear mascaras de depuracion de profundidad vacias por defecto (solo si esta activo el debug)
            if self.show_depth_mask_debug and depth_shape == mask_shape:
                depth_valid_mask = np.zeros_like(mask, dtype=np.uint8)
                depth_valid_mask[self.latest_depth_image > 0] = 255
                depth_roi_mask = np.zeros_like(mask, dtype=np.uint8)
            
            res = self.select_best_sphere_contour(contours)
            
            if res is None:
                # No hay deteccion aceptable en este frame
                self.handle_lost_detection(msg.header.stamp, source_frame=msg.header.frame_id)
            else:
                best_cnt, circularity, fill_ratio, radius_enc, cx, cy = res
                x_enc, y_enc = cx, cy
                
                # OPTIMIZACION: Cropping a Bounding Box para dibujo y erosion local
                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(best_cnt)
                margin = 5 * self.erosion_iterations
                
                h_mask, w_mask = mask.shape[:2]
                x_start = max(0, x_bb - margin)
                y_start = max(0, y_bb - margin)
                x_end = min(w_mask, x_bb + w_bb + margin)
                y_end = min(h_mask, y_bb + h_bb + margin)
                
                local_w = x_end - x_start
                local_h = y_end - y_start
                
                if local_w > 0 and local_h > 0:
                    mask_cnt_local = np.zeros((local_h, local_w), dtype=np.uint8)
                    
                    # Adaptar contorno a coordenadas de la subimagen
                    local_cnt = best_cnt - np.array([x_start, y_start])
                    cv2.drawContours(mask_cnt_local, [local_cnt], -1, 255, -1)
                    
                    # Erosionar la mascara localmente
                    kernel = np.ones((5,5), np.uint8)
                    mask_eroded_local = cv2.erode(mask_cnt_local, kernel, iterations=self.erosion_iterations)
                    
                    if depth_shape == mask_shape:
                        depth_crop = self.latest_depth_image[y_start:y_end, x_start:x_end]
                        depth_pixels = depth_crop[mask_eroded_local == 255]
                        valid_depths = depth_pixels[depth_pixels > 0]
                        valid_mask_pts_count = len(valid_depths)
                        
                        if self.show_depth_mask_debug:
                            mask_eroded = np.zeros_like(mask)
                            mask_eroded[y_start:y_end, x_start:x_end] = mask_eroded_local
                            depth_roi_mask = cv2.bitwise_and(depth_valid_mask, mask_eroded)
                    else:
                        valid_mask_pts_count = 0
                else:
                    valid_mask_pts_count = 0
                
                if valid_mask_pts_count >= self.min_valid_depth_points:
                    # Extraer ventana de profundidad alrededor de cx, cy
                    w_half = self.centroid_depth_window_px // 2
                    h_img, w_img = self.latest_depth_image.shape[:2]
                    r_min = max(0, int(cy - w_half))
                    r_max = min(h_img, int(cy + w_half + 1))
                    c_min = max(0, int(cx - w_half))
                    c_max = min(w_img, int(cx + w_half + 1))
                    
                    window_depths = self.latest_depth_image[r_min:r_max, c_min:c_max]
                    valid_window_depths = window_depths[window_depths > 0]
                    valid_window_pts_count = len(valid_window_depths)
                    
                    if valid_window_pts_count >= self.min_valid_depth_points:
                        z_raw_median = np.median(valid_window_depths)
                        depth_source = "centroid_window"
                    else:
                        z_raw_median = np.median(valid_depths)
                        depth_source = "eroded_mask_fallback"
                    
                    # Normalizar
                    z_m = self.normalize_depth_to_meters(z_raw_median)
                    
                    # Desproyectar
                    surface_center = self.deproject(cx, cy, z_m)
                    
                    if surface_center is not None:
                        # Centro geometrico corregido
                        sphere_center = np.array(surface_center, dtype=float)
                        sphere_center[2] += self.object_dimension / 2.0
                        
                        center_2d = np.array([cx, cy], dtype=float)
                        center_3d_raw = sphere_center.copy()
                        
                        is_global_reacquisition = (
                            self.reset_roi_after_lost and
                            self.lost_frame_count > self.lost_frame_tolerance
                        )
                        
                        # Validar alineacion temporal
                        if not self.accept_temporal_candidate(center_2d, center_3d_raw):
                            self.handle_lost_detection(msg.header.stamp, source_frame=msg.header.frame_id)
                        else:
                            self.last_valid_center_2d = center_2d.copy()
                            self.last_valid_center_3d = center_3d_raw.copy()
                            self.last_valid_stamp = msg.header.stamp
                            
                            # Estabilizacion temporal
                            if self.last_tvec is None or is_global_reacquisition:
                                self.last_tvec = center_3d_raw.copy()
                            else:
                                self.last_tvec = (1.0 - self.alpha_pos) * self.last_tvec + self.alpha_pos * center_3d_raw
                            
                            self.lost_frame_count = 0
                            
                            # Publicar
                            self.publish_sphere_tf_and_pose(
                                self.last_tvec,
                                msg.header.stamp,
                                source_frame=msg.header.frame_id,
                                circularity=circularity
                            )
                            
                            # Overlay visual del candidato aceptado
                            cv2.drawContours(cv_image, [best_cnt], -1, (0, 255, 0), 2) # Verde
                            cv2.circle(cv_image, (int(x_enc), int(y_enc)), int(radius_enc), (255, 0, 0), 2) # Azul (circulo envolvente)
                            cv2.circle(cv_image, (int(cx), int(cy)), 5, (0, 0, 255), -1) # Rojo (centro)
                            
                            # Textos de diagnostico de la deteccion
                            if self.show_debug_overlay:
                                depth_source_es = "ventana_centroide" if depth_source == "centroid_window" else "mascara_erosionada"
                                cv2.putText(cv_image, f"Circ: {circularity:.2f} (Umbral: {self.circularity_threshold:.2f})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Relleno: {fill_ratio:.2f} (Umbral: {self.min_fill_ratio:.2f})", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Radio: {radius_enc:.1f}px ({self.min_radius_px}-{self.max_radius_px})", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Profundidad: {z_m:.3f}m Origen: {depth_source_es}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Pts validos Ventana: {valid_window_pts_count} Mascara: {valid_mask_pts_count}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        self.handle_lost_detection(msg.header.stamp, source_frame=msg.header.frame_id)
                else:
                    self.handle_lost_detection(msg.header.stamp, source_frame=msg.header.frame_id)

            # Dibujar ROI temporal y general si corresponde
            if self.show_debug_overlay:
                if self.use_temporal_roi and self.last_valid_center_2d is not None:
                    roi_active = (self.lost_frame_count <= self.lost_frame_tolerance or not self.reset_roi_after_lost)
                    if roi_active:
                        cv2.circle(cv_image, (int(self.last_valid_center_2d[0]), int(self.last_valid_center_2d[1])), 
                                   self.roi_margin_px, (0, 255, 255), 1) # Amarillo suave
                        cv2.putText(cv_image, "ROI: temporal", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        cv2.putText(cv_image, "ROI: reacquisicion global", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

                # Dashboard general del filtro temporal
                cv2.putText(cv_image, f"Frames perdidos: {self.lost_frame_count} (Max: {self.lost_frame_tolerance})", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(cv_image, f"Rechazos: {self.temporal_reject_count}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                j2d_str = f"{self.last_jump_2d:.2f}px" if self.last_jump_2d is not None else "N/A"
                j3d_str = f"{self.last_jump_3d:.4f}m" if self.last_jump_3d is not None else "N/A"
                cv2.putText(cv_image, f"Salto 2D: {j2d_str} (Max: {self.max_2d_jump_px})", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(cv_image, f"Salto 3D: {j3d_str} (Max: {self.max_3d_jump_m})", (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Mostrar ventanas de depuracion de profundidad si corresponde
            if self.show_depth_mask_debug and depth_shape == mask_shape:
                cv2.imshow(f"{self.object_frame} - mascara profundidad", depth_valid_mask)
                cv2.imshow(f"{self.object_frame} - ROI profundidad", depth_roi_mask)

            # Actualizar FPS
            processing_time_ms = (time.perf_counter() - start_proc) * 1000.0
            self.update_fps_estimate(msg.header.stamp, processing_time_ms)
            
            # Dashboard principal
            title = f"{self.object_frame} - FPS camara: {self.camera_fps_estimate:.1f} | Proc: {self.last_processing_time_ms:.1f} ms"
            cv2.imshow(self.main_window_name, cv_image)
            cv2.setWindowTitle(self.main_window_name, title)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error en image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SphereTrackerNode()
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
