#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_handoff_reachability.py

ROS 2 reachability scanner node for H1-2 robot.
Tests candidate table handoff positions to find points reachable by both left_arm and right_arm.
"""

import csv
import math
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK, GetStateValidity


class HandoffReachabilityScanner(Node):
    def __init__(self):
        super().__init__('handoff_reachability_scanner')
        
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.state_validity_client = self.create_client(GetStateValidity, 'check_state_validity')
        
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio /compute_ik...')
        while not self.state_validity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio /check_state_validity...')
            
        self.get_logger().info('Servicios de MoveIt listos. Iniciando escaneo de alcanzabilidad...')

    def call_service_sync(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def check_pose_validity(self, pos, quat, group_name, ee_link):
        req = GetPositionIK.Request()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = ee_link
        req.ik_request.pose_stamped.header.frame_id = "pelvis"
        req.ik_request.pose_stamped.pose.position.x = float(pos[0])
        req.ik_request.pose_stamped.pose.position.y = float(pos[1])
        req.ik_request.pose_stamped.pose.position.z = float(pos[2])
        req.ik_request.pose_stamped.pose.orientation.x = float(quat[0])
        req.ik_request.pose_stamped.pose.orientation.y = float(quat[1])
        req.ik_request.pose_stamped.pose.orientation.z = float(quat[2])
        req.ik_request.pose_stamped.pose.orientation.w = float(quat[3])
        req.ik_request.avoid_collisions = True
        
        ik_res = self.call_service_sync(self.ik_client, req)
        if ik_res is None or ik_res.error_code.val != 1:
            return False
            
        val_req = GetStateValidity.Request()
        val_req.robot_state = ik_res.solution
        val_req.group_name = group_name
        
        val_res = self.call_service_sync(self.state_validity_client, val_req)
        if val_res is None or not val_res.valid:
            return False
            
        return True

    def run_scan(self):
        # 1. Definir los rangos de búsqueda
        x_vals = [0.35, 0.40, 0.45, 0.50]
        y_vals = [-0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12]
        z_vals = [0.082, 0.10, 0.12]

        # Parámetros geométricos (mano y aproximación)
        radius = 0.0125
        surface_clearance = 0.012
        approach_distance = 0.040
        dz_offset = 0.020
        hover_height = 0.120

        contact_offset = radius + surface_clearance # 0.0245

        # Orientaciones fijas de TCP (quaternions)
        tcp_quat_left = R.from_euler('xyz', [0.0, 0.0, 90.0], degrees=True).as_quat()
        tcp_quat_right = R.from_euler('xyz', [0.0, -179.9, 90.0], degrees=True).as_quat()

        results = []

        total_scans = len(x_vals) * len(y_vals) * len(z_vals)
        idx = 0

        self.get_logger().info(f"Total de candidatos a escanear: {total_scans}")

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    idx += 1
                    
                    # --- Left Arm ---
                    approach_dir_left = np.array([0.0, 1.0, 0.0])
                    tcp_place = np.array([x, y, z]) + approach_dir_left * contact_offset
                    tcp_place[2] += dz_offset
                    tcp_place_above = tcp_place + np.array([0.0, 0.0, hover_height])

                    left_pre_valid = self.check_pose_validity(tcp_place_above, tcp_quat_left, "left_arm", "L_palm_tcp")
                    left_place_valid = self.check_pose_validity(tcp_place, tcp_quat_left, "left_arm", "L_palm_tcp")
                    left_pre_place_valid = left_pre_valid and left_place_valid

                    # --- Right Arm ---
                    approach_dir_right = np.array([0.0, -1.0, 0.0])
                    tcp_contact = np.array([x, y, z]) + approach_dir_right * contact_offset
                    tcp_contact[2] += dz_offset
                    tcp_ready = tcp_contact + approach_dir_right * approach_distance
                    tcp_pregrasp = tcp_ready + np.array([0.0, 0.0, hover_height])

                    right_pregrasp_valid = self.check_pose_validity(tcp_pregrasp, tcp_quat_right, "right_arm", "R_palm_tcp")
                    right_ready_valid = self.check_pose_validity(tcp_ready, tcp_quat_right, "right_arm", "R_palm_tcp")
                    right_contact_valid = self.check_pose_validity(tcp_contact, tcp_quat_right, "right_arm", "R_palm_tcp")

                    all_valid = (
                        left_pre_place_valid and
                        right_pregrasp_valid and
                        right_ready_valid and
                        right_contact_valid
                    )

                    # Lateral transfer distance (Left arm moves from pick y=0.3745 to place y=y_place)
                    y_place_left_tcp = y + 0.0245
                    lat_dist = abs(y_place_left_tcp - 0.3745)

                    results.append({
                        'x': x,
                        'y': y,
                        'z': z,
                        'left_pre_place_valid': left_pre_place_valid,
                        'right_pregrasp_valid': right_pregrasp_valid,
                        'right_contact_valid': right_contact_valid,
                        'all_valid': all_valid,
                        'lateral_transfer_distance': lat_dist
                    })

                    if idx % 10 == 0 or idx == total_scans:
                        self.get_logger().info(f"Escaneo: {idx}/{total_scans} completado...")

        # 2. Ordenar candidatos
        # - all_valid=True primero
        # - menor distancia de transferencia lateral
        # - y más cercana a la línea central (y=0)
        results.sort(key=lambda r: (not r['all_valid'], r['lateral_transfer_distance'], abs(r['y'])))

        # 3. Imprimir tabla en consola
        print("\n" + "="*85)
        print(f"{'X':<6} | {'Y':<6} | {'Z':<6} | {'Left Pre/Place':<15} | {'Right Pregrasp':<15} | {'Right Contact':<15} | {'All Valid':<10} | {'Lat Dist':<10}")
        print("="*85)
        for r in results:
            print(
                f"{r['x']:<6.3f} | {r['y']:<6.3f} | {r['z']:<6.3f} | "
                f"{str(r['left_pre_place_valid']):<15} | {str(r['right_pregrasp_valid']):<15} | {str(r['right_contact_valid']):<15} | "
                f"{str(r['all_valid']):<10} | {r['lateral_transfer_distance']:<10.4f}"
            )
        print("="*85 + "\n")

        # 4. Guardar resultados en CSV
        csv_path = "/home/sebas/ros2_ws/src/H1-UTEC-PFCII/handoff_reachability_scan.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'x', 'y', 'z',
                    'left_pre_place_valid', 'right_pregrasp_valid', 'right_contact_valid',
                    'all_valid', 'lateral_transfer_distance'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            self.get_logger().info(f"Resultados guardados exitosamente en: {csv_path}")
        except Exception as e:
            self.get_logger().error(f"Error al escribir el archivo CSV: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = HandoffReachabilityScanner()
    try:
        node.run_scan()
    except Exception as e:
        node.get_logger().error(f"Error durante el escaneo: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
