#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_right_phase4_contact_variants.py

Diagnóstico para la falla del Phase 4 del brazo derecho.
Realiza un barrido en grilla de posiciones y orientaciones candidatas
alrededor de tcp_contact y ejecuta IK con colisiones y check_state_validity.
"""

import os
import sys
import math
import time
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from moveit_msgs.msg import RobotState

class RightArmPhase4Scanner(Node):
    def __init__(self):
        super().__init__('right_arm_phase4_scanner')
        self.cb_group = ReentrantCallbackGroup()
        
        self.ik_client = self.create_client(
            GetPositionIK, 'compute_ik', callback_group=self.cb_group
        )
        self.state_validity_client = self.create_client(
            GetStateValidity, 'check_state_validity', callback_group=self.cb_group
        )
        
        self.latest_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10, callback_group=self.cb_group
        )
        
        self.scan_thread = None
        
    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def start_scanning(self):
        import threading
        self.scan_thread = threading.Thread(target=self.run_scan)
        self.scan_thread.start()

    def run_scan(self):
        # 1. Esperar servicios
        self.get_logger().info('Esperando servicio /compute_ik...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info('Servicio /compute_ik listo.')

        self.get_logger().info('Esperando servicio /check_state_validity...')
        while not self.state_validity_client.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info('Servicio /check_state_validity listo.')

        # 2. Esperar seed state de joint_states
        self.get_logger().info('Esperando joint_states para usar como semilla...')
        while self.latest_joint_state is None:
            time.sleep(0.1)
        self.get_logger().info('Seed state recibido.')

        # 3. Rangos de búsqueda
        y_offsets = [-0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010]
        z_offsets = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030]
        pitch_candidates = [-179.9, -175.0, -170.0, -165.0, -160.0, -155.0, -150.0]
        yaw_candidates = [80.0, 85.0, 90.0, 95.0, 100.0]

        total_candidates = len(y_offsets) * len(z_offsets) * len(pitch_candidates) * len(yaw_candidates)
        self.get_logger().info(f'Iniciando barrido de {total_candidates} candidatos...')

        results = []
        count = 0

        # TCP de contacto original
        orig_x = 0.5
        orig_y = 0.09550003731562445
        orig_z = 0.10204276054496451

        for y_offset in y_offsets:
            for z_offset in z_offsets:
                distance = math.sqrt(y_offset**2 + z_offset**2)
                target_pos = [orig_x, orig_y + y_offset, orig_z + z_offset]

                for pitch in pitch_candidates:
                    for yaw in yaw_candidates:
                        count += 1
                        if count % 100 == 0:
                            self.get_logger().info(f'Progreso: {count}/{total_candidates} evaluados...')

                        # Calcular orientación usando scipy
                        quat = R.from_euler('xyz', [0.0, pitch, yaw], degrees=True).as_quat()

                        # Preparar request de IK
                        req = GetPositionIK.Request()
                        req.ik_request.group_name = "right_arm"
                        req.ik_request.ik_link_name = "R_palm_tcp"
                        req.ik_request.pose_stamped.header.frame_id = "pelvis"
                        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
                        
                        req.ik_request.pose_stamped.pose.position.x = float(target_pos[0])
                        req.ik_request.pose_stamped.pose.position.y = float(target_pos[1])
                        req.ik_request.pose_stamped.pose.position.z = float(target_pos[2])
                        req.ik_request.pose_stamped.pose.orientation.x = float(quat[0])
                        req.ik_request.pose_stamped.pose.orientation.y = float(quat[1])
                        req.ik_request.pose_stamped.pose.orientation.z = float(quat[2])
                        req.ik_request.pose_stamped.pose.orientation.w = float(quat[3])
                        
                        req.ik_request.avoid_collisions = True
                        req.ik_request.robot_state.joint_state = self.latest_joint_state

                        # Call IK
                        try:
                            ik_response = self.ik_client.call(req)
                        except Exception as e:
                            self.get_logger().error(f"Error llamando a compute_ik: {e}")
                            continue

                        ik_code = ik_response.error_code.val
                        state_valid = False
                        contacts_str = "IK_FAILED"

                        if ik_code == 1:
                            # IK exitosa, validar estado
                            check_req = GetStateValidity.Request()
                            check_req.robot_state = ik_response.solution
                            check_req.group_name = "right_arm"

                            try:
                                val_response = self.state_validity_client.call(check_req)
                                if val_response is not None:
                                    state_valid = val_response.valid
                                    if state_valid:
                                        contacts_str = ""
                                    else:
                                        contacts_list = []
                                        for c in val_response.contacts:
                                            contacts_list.append(f"{c.contact_body_1}<->{c.contact_body_2}")
                                        contacts_str = ";".join(sorted(list(set(contacts_list))))
                            except Exception as e:
                                self.get_logger().error(f"Error llamando a check_state_validity: {e}")
                                contacts_str = "STATE_VAL_ERROR"

                        results.append({
                            'y_offset': y_offset,
                            'z_offset': z_offset,
                            'pitch': pitch,
                            'yaw': yaw,
                            'ik_code': ik_code,
                            'state_valid': state_valid,
                            'contacts': contacts_str,
                            'distance': distance
                        })

        # 4. Ordenar candidatos según las prioridades
        def sort_key(item):
            ik_success = 0 if item['ik_code'] == 1 else 1
            state_valid_val = 0 if item['state_valid'] else 1
            distance = item['distance']
            pitch_diff = abs(item['pitch'] - (-179.9))
            yaw_diff = abs(item['yaw'] - 90.0)
            return (ik_success, state_valid_val, distance, pitch_diff, yaw_diff)

        results.sort(key=sort_key)

        # 5. Imprimir tabla formateada
        self.get_logger().info("=== RESULTADOS DEL BARRIDO DE CANDIDATOS (ORDENADOS POR CALIDAD) ===")
        print(f"\n{'y_offset':<10} | {'z_offset':<10} | {'pitch':<8} | {'yaw':<6} | {'IK code':<8} | {'Valid':<6} | {'Distance':<10} | {'Contacts'}")
        print("-" * 120)
        for item in results:
            valid_str = "True" if item['state_valid'] else "False"
            contacts_short = item['contacts'][:50] + "..." if len(item['contacts']) > 50 else item['contacts']
            print(f"{item['y_offset']:<10.4f} | {item['z_offset']:<10.4f} | {item['pitch']:<8.2f} | {item['yaw']:<6.2f} | {item['ik_code']:<8} | {valid_str:<6} | {item['distance']:<10.4f} | {contacts_short}")

        # 6. Guardar CSV
        csv_path = os.path.expanduser('~/ros2_ws/src/H1-UTEC-PFCII/right_phase4_contact_variant_scan.csv')
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['y_offset', 'z_offset', 'pitch', 'yaw', 'ik_code', 'state_valid', 'contacts', 'distance_from_original_contact'])
                for item in results:
                    writer.writerow([
                        item['y_offset'],
                        item['z_offset'],
                        item['pitch'],
                        item['yaw'],
                        item['ik_code'],
                        item['state_valid'],
                        item['contacts'],
                        item['distance']
                    ])
            self.get_logger().info(f"Resultados guardados exitosamente en: {csv_path}")
        except Exception as e:
            self.get_logger().error(f"Error guardando CSV: {e}")

        # Finalizar
        self.get_logger().info("Escaneo completado. Finalizando nodo...")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RightArmPhase4Scanner()
    node.start_scanning()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
