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
        super().__init__('vision_node')
        
        # Declarar y cargar parámetros
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('object_frame', 'objeto_cubo')
        self.declare_parameter('object_dimension', 0.055)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', '/vision/objeto_cubo/pose')
        
        # Nuevos parámetros para modo debug, trackbars y FPS
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

        # Regla de activación del modo debug visual
        if self.debug_visual:
            self.use_gui_trackbars = True
            self.show_hsv_mask_debug = True
            self.show_depth_mask_debug = True
            self.show_debug_overlay = True

        # Validación
        if self.object_dimension <= 0.0:
            raise ValueError("object_dimension debe ser mayor que 0.0")
        if self.expected_camera_fps <= 0.0:
            raise ValueError("expected_camera_fps debe ser mayor que 0.0")
        if self.fps_window_size <= 0:
            raise ValueError("fps_window_size debe ser mayor que 0")

        # Nombres de ventana estandarizados
        self.main_window_name = self.object_frame
        self.hsv_control_window_name = f"Ajuste HSV - {self.object_frame}"

        # Estado interno de FPS
        self.frame_timestamps = deque(maxlen=self.fps_window_size)
        self.last_processing_time_ms = 0.0
        self.camera_fps_estimate = 0.0

        # Publicadores y Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.last_pose_log_time = 0.0

        # Log al iniciar en español
        self.get_logger().info(
            f"[DETECTOR DE CUBO INICIALIZADO]\n"
            f"  Frame de cámara: '{self.camera_frame}'\n"
            f"  Frame del objeto: '{self.object_frame}'\n"
            f"  Dimensión del objeto [m]: {self.object_dimension:.4f}\n"
            f"  Publicar TF: {self.publish_tf}\n"
            f"  Publicar PoseStamped: {self.publish_pose}\n"
            f"  Tópico de pose: '{self.pose_topic}'\n"
            f"  Debug visual: {self.debug_visual}\n"
            f"  Trackbars HSV: {self.use_gui_trackbars}"
        )
        
        # Suscripciones
        self.subscription_color = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.subscription_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)
            
        self.bridge = CvBridge()
        self.latest_depth_image = None
        
        # Parámetros de Cámara
        self.intrinsics_loaded = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        
        # --- MOTOR DE ESTABILIDAD ANCLADA ---
        self.last_tvec = np.zeros(3)
        self.stable_normal = None  # Inclinación de la mesa
        self.alpha_pos = 0.2
        self.alpha_normal = 0.1   # Ultra-lento para congelar la inclinación
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

        # Calcular el centro geométrico del cubo
        cube_center = face_center - normal * (self.object_dimension / 2.0)

        # Publicar TF
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

        # Publicar Pose
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

        # Throttled logging (1 por segundo)
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

    def camera_info_callback(self, msg):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.intrinsics_loaded = True
        self.get_logger().info("Intrínsecos cargados.")
        self.destroy_subscription(self.info_sub)

    def deproject(self, u, v, z_mm):
        if z_mm <= 0: return None
        z_m = z_mm / 1000.0
        return [(u - self.cx) * z_m / self.fx, (v - self.cy) * z_m / self.fy, z_m]

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
        new_pts[0] = pts[np.argmin(s)] # TL
        new_pts[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis=1)
        new_pts[1] = pts[np.argmin(diff)] # TR
        new_pts[3] = pts[np.argmax(diff)] # BL
        return new_pts

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        if not self.intrinsics_loaded or self.latest_depth_image is None: return
        
        start_proc = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Segmentación HSV
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
                        # 1. Visual Debug: Contorno y Esquinas
                        cv2.polylines(cv_image, [approx], True, (0, 255, 0), 2)
                        corners_2d = self.sort_corners(approx)
                        for pt in corners_2d:
                            cv2.circle(cv_image, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        
                        # 2. Yaw Visual (TL -> TR)
                        v_x_2d = corners_2d[1] - corners_2d[0]
                        angle_2d = np.rad2deg(np.arctan2(v_x_2d[1], v_x_2d[0]))
                        rad = np.deg2rad(angle_2d)
                        visual_yaw_vec = np.array([np.cos(rad), np.sin(rad), 0.0])
                        
                        # 3. Normal (Inclinación)
                        mask_cnt = np.zeros(mask.shape, dtype=np.uint8)
                        cv2.drawContours(mask_cnt, [best_cnt], -1, 255, -1)
                        indices = np.argwhere(mask_cnt == 255)[::15]
                        pts_3d = []
                        for v, u in indices:
                            z = self.latest_depth_image[v, u]
                            if z > 200:
                                p = self.deproject(u, v, z)
                                if p: pts_3d.append(p)
                        
                        if len(pts_3d) > 15:
                            pts_3d = np.array(pts_3d)
                            mean_center = np.mean(pts_3d, axis=0)
                            cov = np.cov((pts_3d - mean_center).T)
                            evals, evecs = np.linalg.eigh(cov)
                            raw_normal = evecs[:, np.argmin(evals)]
                            if raw_normal[2] > 0: raw_normal = -raw_normal
                            
                            # Congelar Normal
                            if self.stable_normal is None: self.stable_normal = raw_normal
                            else: self.stable_normal = (1 - self.alpha_normal) * self.stable_normal + self.alpha_normal * raw_normal
                            self.stable_normal /= np.linalg.norm(self.stable_normal)
                            
                            # 4. Construir Matriz y Pose
                            z_axis = self.stable_normal
                            x_axis = visual_yaw_vec - np.dot(visual_yaw_vec, z_axis) * z_axis
                            x_axis /= np.linalg.norm(x_axis)
                            y_axis = np.cross(z_axis, x_axis)
                            R_final = np.column_stack((x_axis, y_axis, z_axis))
                            
                            self.last_tvec = (1 - self.alpha_pos) * self.last_tvec + self.alpha_pos * mean_center
                            quat = self.rotation_matrix_to_quaternion(R_final)
                            
                            # Publicar TF y Pose
                            self.publish_cube_tf_and_pose(
                                face_center=self.last_tvec,
                                normal=self.stable_normal,
                                quat=quat,
                                stamp=msg.header.stamp,
                                source_frame=msg.header.frame_id
                            )
                            
                            rvec_draw, _ = cv2.Rodrigues(R_final)
                            cv2.drawFrameAxes(cv_image, np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), None, rvec_draw, self.last_tvec, 0.05)
                            
                            if self.show_debug_overlay:
                                cube_center_pos = self.last_tvec - self.stable_normal * (self.object_dimension / 2.0)
                                cv2.putText(cv_image, f"Centro cara: {[round(c, 3) for c in self.last_tvec.tolist()]} m", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Centro cubo: {[round(c, 3) for c in cube_center_pos.tolist()]} m", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Profundidad: {self.last_tvec[2]:.3f} m", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"FPS camara: {self.camera_fps_estimate:.1f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(cv_image, f"Tiempo procesamiento: {self.last_processing_time_ms:.1f} ms", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
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
    node = VisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
