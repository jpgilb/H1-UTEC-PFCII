
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
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
        
        cv2.namedWindow("Control")
        self.setup_trackbars()
        self.get_logger().info("Motor de Pose Anclado (Fijo en Mesa) con Debug Visual Activo")

    def setup_trackbars(self):
        cv2.createTrackbar("Min H", "Control", 50, 179, lambda x: None)
        cv2.createTrackbar("Min S", "Control", 100, 255, lambda x: None)
        cv2.createTrackbar("Min V", "Control", 0, 255, lambda x: None)
        cv2.createTrackbar("Max H", "Control", 132, 179, lambda x: None)
        cv2.createTrackbar("Max S", "Control", 255, 255, lambda x: None)
        cv2.createTrackbar("Max V", "Control", 255, 255, lambda x: None)

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
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Segmentación HSV
            l_h, l_s, l_v = cv2.getTrackbarPos("Min H", "Control"), cv2.getTrackbarPos("Min S", "Control"), cv2.getTrackbarPos("Min V", "Control")
            u_h, u_s, u_v = cv2.getTrackbarPos("Max H", "Control"), cv2.getTrackbarPos("Max S", "Control"), cv2.getTrackbarPos("Max V", "Control")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
                            
                            # Logs y Ejes
                            self.get_logger().info(f"POSE ANCLADA -> P: [{self.last_tvec[0]:.3f}, {self.last_tvec[1]:.3f}, {self.last_tvec[2]:.3f}] | Q: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                            rvec_draw, _ = cv2.Rodrigues(R_final)
                            cv2.drawFrameAxes(cv_image, np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]), None, rvec_draw, self.last_tvec, 0.05)

            # Dashboard
            fps = 1.0 / (time.time() - self.last_time)
            self.last_time = time.time()
            cv2.setWindowTitle("Anchored Pose", f"Anchored Pose - FPS: {fps:.1f}")
            cv2.imshow("Anchored Pose", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

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
