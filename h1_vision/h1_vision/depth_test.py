
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthTestNode(Node):
    def __init__(self):
        super().__init__('depth_test_node')
        
        # Suscripción a la imagen de profundidad
        # Se asume que el tópico es /camera/depth/image_raw
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)
            
        self.bridge = CvBridge()
        self.get_logger().info("Nodo de prueba de profundidad iniciado. Midiendo el centro de la pantalla (ROI 3x3).")

    def depth_callback(self, msg):
        try:
            # Convertir imagen de ROS a OpenCV
            # Usamos passthrough para mantener la precisión original (usualmente mm en 16UC1)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Obtener dimensiones de la imagen
            h, w = depth_image.shape
            
            # Identificar el píxel central
            cx, cy = w // 2, h // 2
            
            # Extraer ROI de 3x3 alrededor del centro
            # Definimos los límites asegurándonos de no salirnos de la imagen
            x_start = max(0, cx - 1)
            x_end = min(w, cx + 2)
            y_start = max(0, cy - 1)
            y_end = min(h, cy + 2)
            
            roi = depth_image[y_start:y_end, x_start:x_end]
            
            # Calcular la mediana de los 9 píxeles para filtrar ruido puntual
            # Sin límites de distancia por requerimiento
            z_mm = np.median(roi)
            
            # --- Visualización ---
            # Normalizamos la imagen de profundidad para que sea visible (0-255)
            depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            
            # Dibujar un pequeño marcador en el centro (el ROI 3x3)
            cv2.rectangle(depth_vis, (x_start, y_start), (x_end-1, y_end-1), (0, 0, 255), 2)
            
            # Escribir el valor de distancia en la imagen
            text = f"Z Centro: {z_mm:.1f} mm"
            cv2.putText(depth_vis, text, (cx + 15, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar ventana de OpenCV
            cv2.imshow("Test de Profundidad - Centro", depth_vis)
            cv2.waitKey(1)
            
            # Log en terminal para monitoreo rápido
            # self.get_logger().info(f"Centro ({cx}, {cy}) -> Z: {z_mm:.1f} mm")
            
        except Exception as e:
            self.get_logger().error(f"Error en el procesamiento de profundidad: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthTestNode()
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
