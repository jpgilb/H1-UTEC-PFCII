# Plan de Auditoría y Preparación de Variante Dinámica para el H1-2

Este documento resume la auditoría técnica realizada sobre la cadena de control del humanoide H1-2 en simulación MuJoCo + ROS 2 Humble + MoveIt 2, y detalla la preparación del andamiaje cinemático y dinámico para la futura migración a control dinámico por torque/esfuerzo con Pinocchio.

---

## 1. Resumen del Estado Actual

*   **Cinemática (ROS 2/MoveIt):** MoveIt calcula la trayectoria geométrica y la discretiza temporalmente. ROS 2 envía consignas de posición pura ($q_{cmd}$) a $100\text{ Hz}$ usando controladores `JointTrajectoryController` sin ganancias PID definidas localmente en ROS.
*   **Dinámica (MuJoCo):** La física se integra a $1000\text{ Hz}$. Los brazos operan con gravedad real activa (`gravcomp="0"`), y la gravedad se contrarresta únicamente a través de servos de posición virtuales de MuJoCo (`<position kp="..."/>`) actuando como resortes locales. Esto genera un error de estado estable inherente (caída vertical por gravedad/sagging).
*   **Pinocchio:** Se encuentra instalado y operativo localmente (versión `4.0.0`), listo para cálculo dinámico inverso (gravedad, Coriolis, inercia).
*   **Modelo de Dinámica Generado:** El modelo URDF dinámico `h1_2_pinocchio_dynamics.urdf` ya ha sido exitosamente generado.

---

## 2. Archivos Encontrados y Auditados

### Modelos y Descripciones (en `h1_2_mujoco_bringup/description/`)
*   `h1_2_mujoco_model_env_v1_wide_shelves_camera_dynamics.xml` (Existe, correcto).
*   `h1_2_mujoco_wide_shelves_camera_dynamics.urdf.xacro` (Existe, corregido).
*   `h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro` (Existe, corregido).
*   `generated/h1_2_pinocchio_dynamics.urdf` (Existe, correcto).

### Archivos de Lanzamiento y Configuración
*   `launch/h1_2_mujoco_wide_camera_dynamics.launch.py` (Creado en esta fase).
*   `config/ros2_controllers_dynamics.yaml` (Creado en esta fase).

---

## 3. Inconsistencias Corregidas (Auditoría de Enlaces Internos)

Durante la auditoría, se detectaron y corrigieron las siguientes inconsistencias de direccionamiento en los archivos `_dynamics` para garantizar un entorno paralelo y aislado:

1.  **URDF Xacro a ros2_control:**
    *   *Inconsistencia:* [h1_2_mujoco_wide_shelves_camera_dynamics.urdf.xacro](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/h1_2_mujoco_wide_shelves_camera_dynamics.urdf.xacro#L27) incluía el ros2_control de gravedad nominal (`h1_2_mujoco_wide_shelves_camera_gravity_arms.ros2_control.xacro`).
    *   *Corrección:* Se modificó para apuntar al archivo dinámico: `h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro`.
2.  **ros2_control a XML de MuJoCo:**
    *   *Inconsistencia:* [h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/h1_2_mujoco_wide_shelves_camera_dynamics.ros2_control.xacro#L10) apuntaba al modelo XML nominal (`h1_2_mujoco_model_env_v1_wide_shelves_camera_gravity_arms.xml`).
    *   *Corrección:* Se modificó para cargar el XML dinámico: `h1_2_mujoco_model_env_v1_wide_shelves_camera_dynamics.xml`.
3.  **Lanzamiento (Launch) Dinámico (Creado):**
    *   *Inconsistencia:* No existía un launch específico para esta variante dinámica.
    *   *Solución:* Se creó `h1_2_mujoco_wide_camera_dynamics.launch.py` apuntando directamente a `h1_2_mujoco_wide_shelves_camera_dynamics.urdf.xacro` y utilizando el nuevo perfil de controladores `ros2_controllers_dynamics.yaml`.
4.  **Configuración de Controladores (Creado):**
    *   *Inconsistencia:* No existía un archivo de configuración de controladores aislado.
    *   *Solución:* Se creó `ros2_controllers_dynamics.yaml` en `h1_2_mujoco_bringup/config/`.

---

## 4. Inconsistencias Pendientes (Diseño de la Fase 1)

*   **Actuadores XML:** Actualmente, la variante `_dynamics.xml` conserva actuadores de posición (`<position>`). En la Fase 1, para permitir control dinámico por torque, estos deberán cambiarse a motores de torque puro (`<motor ctrlrange="-200 200" gear="1" .../>`).
*   **Hardware Interface (ros2_control):** `command_interface` de las juntas de los brazos y torso en el xacro dinámico continúan en modo `position`. Deberán cambiarse a `effort` en la Fase 1.
*   **Tipos de Controladores en YAML:** En `ros2_controllers_dynamics.yaml`, `left_arm_controller` y `right_arm_controller` siguen configurados como `joint_trajectory_controller` con comandos en posición. Deberán cambiarse a `effort_controllers/JointGroupEffortController` o a un controlador personalizado.

---

## 5. Reporte de Actuadores y Controladores Auditados

### Actuadores actuales en el XML (`_dynamics.xml`)
Todos los actuadores son servos de posición virtuales (`<position ... kp="..."/>`):
*   **Brazo Izquierdo (7 DOF):**
    - `left_shoulder_pitch_joint` (kp="160")
    - `left_shoulder_roll_joint` (kp="160")
    - `left_shoulder_yaw_joint` (kp="120")
    - `left_elbow_joint` (kp="220")
    - `left_wrist_roll_joint` (kp="65")
    - `left_wrist_pitch_joint` (kp="65")
    - `left_wrist_yaw_joint` (kp="65")
*   **Brazo Derecho (7 DOF):**
    - `right_shoulder_pitch_joint` (kp="160")
    - `right_shoulder_roll_joint` (kp="160")
    - `right_shoulder_yaw_joint` (kp="120")
    - `right_elbow_joint` (kp="220")
    - `right_wrist_roll_joint` (kp="65")
    - `right_wrist_pitch_joint` (kp="65")
    - `right_wrist_yaw_joint` (kp="65")
*   **Mano Izquierda (6 DOF):**
    - `L_index_proximal_joint` (kp="1")
    - `L_middle_proximal_joint` (kp="1")
    - `L_pinky_proximal_joint` (kp="1")
    - `L_ring_proximal_joint` (kp="1")
    - `L_thumb_proximal_yaw_joint` (kp="1")
    - `L_thumb_proximal_pitch_joint` (kp="1")
*   **Mano Derecha (6 DOF):**
    - `R_index_proximal_joint` (kp="1")
    - `R_middle_proximal_joint` (kp="1")
    - `R_pinky_proximal_joint` (kp="1")
    - `R_ring_proximal_joint` (kp="1")
    - `R_thumb_proximal_yaw_joint` (kp="1")
    - `R_thumb_proximal_pitch_joint` (kp="1")

### Controladores en `ros2_controllers_dynamics.yaml`
*   **Controladores cargados:** `left_arm_controller`, `right_arm_controller`, `torso_controller`, `left_hand_controller`, `right_hand_controller`, `joint_state_broadcaster`.
*   **Tipo de controlador:** Todos utilizan `joint_trajectory_controller/JointTrajectoryController` excepto el broadcaster.
*   **Interfaces expuestas por JTC:**
    - Action Server: `/<controller_name>/follow_joint_trajectory` (tipo `control_msgs::action::FollowJointTrajectory`).
    - Tópico de consigna: `/<controller_name>/joint_trajectory` (tipo `trajectory_msgs::msg::JointTrajectory`).

### Controladores en MoveIt (`moveit_controllers.yaml`)
MoveIt espera interactuar con los siguientes servidores de acción:
*   `left_arm_controller/follow_joint_trajectory` (Action tipo `FollowJointTrajectory`)
*   `right_arm_controller/follow_joint_trajectory` (Action tipo `FollowJointTrajectory`)
*   `left_hand_controller/follow_joint_trajectory` (Action tipo `FollowJointTrajectory`)
*   `right_hand_controller/follow_joint_trajectory` (Action tipo `FollowJointTrajectory`)

---

## 6. Propuesta de Arquitectura para Control Dinámico

Para implementar control dinámico por torque sin romper MoveIt ni modificar el script de alto nivel `mover_brazo_single.py`:

```
                           [mover_brazo_single.py]
                                     │
                                     ▼ (MoveGroup API)
                              [move_group Node]
                                     │
                                     ▼ (FollowJointTrajectory Action)
                       [custom_dynamics_controller Node]
                         - Carga URDF dinámico
                         - Instancia Pinocchio API (para G(q), M(q), C(q,q_dot))
                                     │
                                     ▼ (Publica consignas de torque / esfuerzo)
                     [/left_arm_controller/commands] (Effort Interface)
                                     │
                                     ▼
                        [mujoco_ros2_control plugin]
                                     │
                                     ▼ (Torque puro: d->ctrl)
                           [MuJoCo Simulator]
```

### Detalle de la Implementación Custom
1.  **Controlador Custom en ROS 2:** Crearemos un nodo ROS 2 que se registre como un Action Server de tipo `FollowJointTrajectory` bajo los nombres de `/left_arm_controller` y `/right_arm_controller`.
2.  **Sustitución Transparente:** Al usar los mismos nombres y tipos de acción esperados por MoveIt, este nodo interceptará el comando de trayectoria geométrico enviado por MoveIt.
3.  **Cálculo en Lazo Cerrado (Pinocchio):** En cada paso del bucle de control ($100\text{-}500\text{ Hz}$), el nodo leerá el estado actual del robot de `/joint_states` y calculará:
    *   Término feedforward de gravedad con Pinocchio: $\tau_{grav} = G(q)$
    *   Término de control de seguimiento PID en espacio articular: $\tau_{pid} = K_p (q_{des} - q) + K_d (\dot{q}_{des} - \dot{q})$
    *   Torque de salida total: $\tau = \tau_{grav} + \tau_{pid}$
4.  **Envío a MuJoCo:** Publicará $\tau$ a los tópicos de esfuerzo de `ros2_control`, los cuales serán aplicados directamente por MuJoCo a través de los actuadores `<motor>`.

---

## 7. Riesgos Técnicos

1.  **Frecuencia del Bucle de Control:** ROS 2 control corre usualmente a $100\text{ Hz}$. La sintonización de un PID en espacio de torque a esta frecuencia puede ser inestable si la inercia del robot varía rápido. Se recomienda forzar un `update_rate` del controlador de al menos $200\text{-}500\text{ Hz}$.
2.  **Exactitud del URDF frente al XML:** Si hay discrepancias de masa o centros de masa entre `h1_2_pinocchio_dynamics.urdf` y `h1_2_mujoco_model_env_v1_wide_shelves_camera_dynamics.xml`, la compensación de gravedad calculada por Pinocchio no anulará el peso de forma perfecta, resultando en un ligero desvío persistente.

---

## 8. Comandos de Validación Recomendados

Para validar la correcta inicialización del entorno dinámico en paralelo sin fallos:

1.  **Compilar y Sincronizar Workspace:**
    ```bash
    colcon build --symlink-install --packages-select h1_2_mujoco_bringup
    ```
2.  **Lanzar Simulación MuJoCo Dinámica (en Terminal 1):**
    ```bash
    ros2 launch h1_2_mujoco_bringup h1_2_mujoco_wide_camera_dynamics.launch.py
    ```
3.  **Verificar que los Controladores e Interfaces Cargan Correctamente (en Terminal 2):**
    ```bash
    ros2 control list_hardware_interfaces
    ros2 control list_controllers
    ```

---

## 9. Recomendación Final: Listo para la Fase 1

El andamiaje cinemático dinámico en paralelo (`_dynamics`) ha sido completamente auditado, las inconsistencias de rutas en Xacros han sido resueltas y el launch dinámico está configurado. El sistema se encuentra **completamente listo** para proceder con la Fase 1 (Sustitución de interfaces a `effort`, cambio de actuadores a `motor` en el XML y desarrollo del custom controller basado en Pinocchio).
