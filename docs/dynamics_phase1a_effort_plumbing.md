# Fase 1A: Plomería de Esfuerzos y Torques (Effort Plumbing) para Brazos

Este documento resume la implementación de la Fase 1A del andamiaje dinámico, la cual habilita la interfaz física de esfuerzos/torques en los brazos izquierdo y derecho del robot H1-2 dentro del simulador MuJoCo y del framework de `ros2_control`, aislando los cambios de la versión de producción cinemática estable.

---

## 1. Archivos Modificados y Creados

### Archivos de Configuración y Lanzamiento (Modificados)
1.  **[h1_2_mujoco_model_env_v1_wide_shelves_camera_dynamics.xml](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/h1_2_mujoco_model_env_v1_wide_shelves_camera_dynamics.xml)**: Conversión de 14 actuadores de brazos de control de posición a motores de torque puro.
2.  **[h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro)**: Cambio de las interfaces de comando de los brazos a esfuerzo y adición de la interfaz de estado de esfuerzo.
3.  **[ros2_controllers_dynamics.yaml](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/config/ros2_controllers_dynamics.yaml)**: Reemplazo de los controladores de trayectoria `JointTrajectoryController` de los brazos por controladores directos de esfuerzo `ForwardCommandController`.
4.  **[h1_2_mujoco_wide_camera_dynamics.launch.py](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/launch/h1_2_mujoco_wide_camera_dynamics.launch.py)**: Ajuste de spawners de controladores en el arranque para levantar los controladores de esfuerzo en lugar del JTC nominal.

---

## 2. Lista de Actuadores Convertidos a Motores

Se cambiaron los siguientes 14 actuadores a la etiqueta `<motor>` en el archivo XML:

### Brazo Izquierdo (L)
*   `left_shoulder_pitch_motor` (joint: `left_shoulder_pitch_joint`, límites: $\pm 40\text{ N}\cdot\text{m}$)
*   `left_shoulder_roll_motor` (joint: `left_shoulder_roll_joint`, límites: $\pm 40\text{ N}\cdot\text{m}$)
*   `left_shoulder_yaw_motor` (joint: `left_shoulder_yaw_joint`, límites: $\pm 18\text{ N}\cdot\text{m}$)
*   `left_elbow_motor` (joint: `left_elbow_joint`, límites: $\pm 18\text{ N}\cdot\text{m}$)
*   `left_wrist_roll_motor` (joint: `left_wrist_roll_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)
*   `left_wrist_pitch_motor` (joint: `left_wrist_pitch_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)
*   `left_wrist_yaw_motor` (joint: `left_wrist_yaw_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)

### Brazo Derecho (R)
*   `right_shoulder_pitch_motor` (joint: `right_shoulder_pitch_joint`, límites: $\pm 40\text{ N}\cdot\text{m}$)
*   `right_shoulder_roll_motor` (joint: `right_shoulder_roll_joint`, límites: $\pm 40\text{ N}\cdot\text{m}$)
*   `right_shoulder_yaw_motor` (joint: `right_shoulder_yaw_joint`, límites: $\pm 18\text{ N}\cdot\text{m}$)
*   `right_elbow_motor` (joint: `right_elbow_joint`, límites: $\pm 18\text{ N}\cdot\text{m}$)
*   `right_wrist_roll_motor` (joint: `right_wrist_roll_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)
*   `right_wrist_pitch_motor` (joint: `right_wrist_pitch_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)
*   `right_wrist_yaw_motor` (joint: `right_wrist_yaw_joint`, límites: $\pm 19\text{ N}\cdot\text{m}$)

---

## 3. Interfaces de `ros2_control` Resultantes

Para las juntas de los brazos en `h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro`:
*   **Command Interfaces:**
    *   `<command_interface name="effort"/>`
*   **State Interfaces:**
    *   `<state_interface name="position"/>`
    *   `<state_interface name="velocity"/>`
    *   `<state_interface name="effort"/>`

Las manos (`left_hand_controller`, `right_hand_controller`) y el torso (`torso_controller`) conservan su configuración original basada en control de posición.

---

## 4. Controladores Activos en la Variante Dinámica

Configurados en `ros2_controllers_dynamics.yaml`:
1.  `joint_state_broadcaster` (`joint_state_broadcaster/JointStateBroadcaster`)
2.  `torso_controller` (`joint_trajectory_controller/JointTrajectoryController` por posición)
3.  `left_hand_controller` y `right_hand_controller` (`joint_trajectory_controller/JointTrajectoryController` por posición)
4.  `left_arm_effort_controller` (`forward_command_controller/ForwardCommandController` por esfuerzo)
5.  `right_arm_effort_controller` (`forward_command_controller/ForwardCommandController` por esfuerzo)

### Tópicos y Acciones Expuestos
*   **Esfuerzos de los brazos:**
    *   `/left_arm_effort_controller/commands` (tipo `std_msgs/msg/Float64MultiArray`)
    *   `/right_arm_effort_controller/commands` (tipo `std_msgs/msg/Float64MultiArray`)
*   **Manos, Dedos y Torso:**
    *   Servidores de acción tipo `FollowJointTrajectory` bajo `/left_hand_controller`, `/right_hand_controller`, `/torso_controller`.

---

## 5. Comandos de Validación Ejecutados

1.  **Parse de MJCF con la API de MuJoCo (Python):** exitoso (`MJCF OK`).
2.  **Generación de URDF con xacro:** exitoso sin advertencias ni errores.
3.  **Compilación con colcon:** exitosa.

---

## 6. Riesgos Técnicos

*   **Caída Inmediata del Brazo (Gravedad):** Al activar controladores de esfuerzo puros (`ForwardCommandController`), el comando por defecto enviado al arrancar el spawner es `0.0`. Debido a que la gravedad está activa en los brazos (`gravcomp="0"`), **ambos brazos caerán libremente por su propio peso al arrancar el simulador** si no se les inyecta torque de inmediato. Esto es normal en sistemas de torque directo y confirma que el plumbing de esfuerzos está funcionando de manera realista.
*   **Sintonización de Ganancias:** En la siguiente fase, el nodo Pinocchio deberá inyectar torques estables a alta frecuencia para evitar colisiones violentas por caída libre durante el arranque del simulador.

---

## 7. Qué NO se Implementó Todavía (Fases Futuras)

*   **Algoritmo de Control con Pinocchio:** El nodo de ROS 2 que cargará la cinemática/dinámica inversa para calcular $\tau_{grav}$ y $\tau_{pid}$ no ha sido creado.
*   **Action Server de MoveIt:** No se ha modificado la configuración de MoveIt; los nombres `left_arm_controller` y `right_arm_controller` permanecen inactivos esperando ser asumidos por nuestro custom dynamics controller.

---

## 8. Recomendación de Siguiente Fase

Proceder directamente a la **Fase 1B**: Creación del nodo ROS 2 `h1_2_dynamics_controller` (escrito en C++ o Python) que implementará el Action Server `FollowJointTrajectory`, cargará Pinocchio, resolverá la compensación de gravedad en lazo cerrado e inyectará los esfuerzos correspondientes al tópico `/left_arm_effort_controller/commands` y `/right_arm_effort_controller/commands`.
