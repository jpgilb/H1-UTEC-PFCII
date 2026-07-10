# MUJOCO MOVEIT DEBUG SUMMARY

Este reporte documenta el estado actual de la integración entre MuJoCo, MoveIt 2 y el nodo de control `mover_brazo_single.py`, detallando los parámetros de diagnóstico, la configuración del controlador y las recomendaciones técnicas de estabilidad y seguimiento.

---

## 1. Estructura del Paquete y Launch Files Utilizados

### Estructura de Paquetes
* **`h1_2_mujoco_bringup`**: Contiene la descripción de hardware (URDF/Xacro, MJCF/XML), archivos de configuración para los controladores de simulación y los scripts de lanzamiento principales para MuJoCo y MoveIt.
* **`h1_2_moveit_config`**: Proporciona la configuración semántica del robot (SRDF), límites de articulaciones, cinemática e interfaces de planificación de MoveIt 2.
* **`H1-UTEC-PFCII`**: Contiene el código fuente de los scripts de manipulación de alto nivel, incluyendo `mover_brazo_single.py`.

### Archivos de Lanzamiento Utilizados
1. **`h1_2_mujoco.launch.py`** (en `h1_2_mujoco_bringup`):
   * Lanza `robot_state_publisher`.
   * Lanza `mujoco_ros2_control`'s `ros2_control_node` cargando el modelo URDF/Xacro y la configuración de controladores YAML con el parámetro `use_sim_time:=True`.
   * Lanza de forma secuencial y con retrasos mediante `TimerAction` los spawners de los controladores de ROS 2:
     * `joint_state_broadcaster` (retraso 6.0s)
     * `left_arm_controller` (retraso 9.0s)
     * `right_arm_controller` (retraso 11.0s)
     * `torso_controller` (retraso 13.0s)
     * `left_hand_controller` (retraso 15.0s)
     * `right_hand_controller` (retraso 17.0s)
2. **`moveit_on_mujoco.launch.py`** (en `h1_2_mujoco_bringup`):
   * Lanza el nodo `move_group` con las configuraciones de cinemática, OMPL, límites de articulaciones e interfaces de ejecución de trayectorias.

---

## 2. Configuración del Modelo de MuJoCo (MJCF)

Parámetros actuales en [h1_2_mujoco_model.xml](file:///Ubuntu-22.04/home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/h1_2_mujoco_model.xml):

* **Opción Timestep (`option timestep`)**: `0.01` (10 ms)
* **Estado de Gravedad**: Desactivada (`gravity="0 0 0"`)
* **Estado de Contacto**: Desactivados globalmente mediante la bandera `<flag contact="disable"/>`
* **Configuración de `torso_joint`**:
  * `damping`: `5.0`
  * `armature`: `0.02`
* **Gains de Actuadores (kp)**:
  * **Torso (`torso_joint`)**: `50.0`
  * **Brazo Izquierdo (`left_arm` joints)**: `10` para todas las articulaciones del hombro, codo y muñeca
  * **Brazo Derecho (`right_arm` joints)**: `10` para todas las articulaciones del hombro, codo y muñeca
  * **Actuadores de Manos (L/R hands)**: `1` para todas las articulaciones de flexión proximal de los dedos e índice/pulgar

---

## 3. Configuración de ros2_control

Parámetros en [ros2_controllers_mujoco.yaml](file:///Ubuntu-22.04/home/sebas/ros2_ws/src/h1_2_mujoco_bringup/config/ros2_controllers_mujoco.yaml):

* **Frecuencia de Actualización (`update_rate`)**: `20` Hz
* **Controladores e Interfaces de Control**:
  * **`joint_state_broadcaster`** (`joint_state_broadcaster/JointStateBroadcaster`)
  * **`left_arm_controller`** (`joint_trajectory_controller/JointTrajectoryController`):
    * Articulaciones: `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`
    * Interfaces: Command (`position`), State (`position`, `velocity`)
  * **`right_arm_controller`** (`joint_trajectory_controller/JointTrajectoryController`):
    * Articulaciones: `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint`, `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`
    * Interfaces: Command (`position`), State (`position`, `velocity`)
  * **`torso_controller`** (`joint_trajectory_controller/JointTrajectoryController`):
    * Articulación: `torso_joint`
    * Interfaces: Command (`position`), State (`position`, `velocity`)
  * **`left_hand_controller`** (`joint_trajectory_controller/JointTrajectoryController`):
    * Articulaciones: `L_index_proximal_joint`, `L_middle_proximal_joint`, `L_pinky_proximal_joint`, `L_ring_proximal_joint`, `L_thumb_proximal_yaw_joint`, `L_thumb_proximal_pitch_joint`
    * Interfaces: Command (`position`), State (`position`, `velocity`)
  * **`right_hand_controller`** (`joint_trajectory_controller/JointTrajectoryController`):
    * Articulaciones: `R_index_proximal_joint`, `R_middle_proximal_joint`, `R_pinky_proximal_joint`, `R_ring_proximal_joint`, `R_thumb_proximal_yaw_joint`, `R_thumb_proximal_pitch_joint`
    * Interfaces: Command (`position`), State (`position`, `velocity`)
* **Tolerancias y Restricciones de Ejecución**:
  * No hay tolerancias configuradas a nivel de joint_trajectory_controller en `ros2_controllers_mujoco.yaml`.
  * En `moveit_on_mujoco.launch.py`, el bloque `trajectory_execution` define:
    * `trajectory_execution.allowed_execution_duration_scaling`: `1.5`
    * `trajectory_execution.allowed_goal_duration_margin`: `1.0`
    * `trajectory_execution.allowed_start_tolerance`: `0.05`

---

## 4. Parámetros de Depuración Activos en `mover_brazo_single.py`

Valores actuales configurados en el nodo:

* **`debug_disable_object_collision`**: `True` (evita agregar el objeto manipulado a la escena de colisión de MoveIt, impidiendo el fallo de trayectoria por colisión inicial)
* **`post_motion_settle_time`**: `1.0` s (tiempo de espera programado tras finalizar un movimiento exitoso antes de avanzar de fase)
* **`debug_stop_after_phase`**: `0` (parámetro para suspender la secuencia tras completar una fase específica)
* **`phase4_motion_mode`**: `'cartesian'` (utiliza aproximación frontal cartesiana directa)
* **`phase8_motion_mode`**: `'cartesian_split'` (traslado segmentado con waypoints elevados)
* **`phase8_transit_z`**: `0.28` (cota Z mínima para el traslado cartesiano elevado)
* **`phase8_min_cartesian_fraction`**: `0.70` (porcentaje mínimo de trayectoria cartesiana exitosa requerida para Fase 8)

---

## 5. Resumen del Comportamiento Conocido

* **Fase 2 (Preagarre Elevado)**: Se completa con éxito al configurar `debug_disable_object_collision=true`.
* **Fase 3 (Descenso Vertical)**: Falla de manera intermitente con código de error MoveIt `-17` (`INVALID_ROBOT_STATE`), típicamente debido a una transición rápida antes de que `/joint_states` se sincronice tras la Fase 2.
* **Fase 4 (Aproximación frontal)**: Se ejecuta y completa satisfactoriamente operando en modo cartesiano local.
* **Fase 8 (Traslado sobre mesa)**: Falla de forma recurrente con error de MoveIt `-17` (`INVALID_ROBOT_STATE`) utilizando el modo por defecto de OMPL.
* **Oscilación y Jitter**: Se observa oscilación física notable en los eslabones del robot cuando se estabiliza cerca del setpoint de destino.
* **Deriva del Torso**: Con la sintonización aplicada (`kp=50.0`, `damping=5.0`, `armature=0.02`), el desplazamiento no deseado del torso se estabiliza por debajo del límite aceptable (~0.0038 rad máximo).

---

## 6. Puntos de Código donde se Gestionan Éxito/Fallo de Acciones

### Movimientos Articulares (OMPL / MoveGroup)
* **Registro de callbacks**:
  * En `planificar_a_pose(self, ...)` y `enviar_meta_articular_brazo(self, ...)` se hace la llamada asíncrona:
    ```python
    future = self.move_group_client.send_goal_async(goal_msg)
    future.add_done_callback(self.move_group_goal_response_callback)
    ```
  * En `move_group_goal_response_callback(self, future)`, al aceptarse la meta, se asigna el callback de resultado final:
    ```python
    result_future.add_done_callback(self.move_group_result_callback)
    ```
* **Manejo del Resultado**:
  * En [move_group_result_callback(self, future)](file:///Ubuntu-22.04/home/sebas/ros2_ws/src/H1-UTEC-PFCII/scripts/mover_brazo_single.py#L3620):
    * **Éxito** (`result.error_code.val == 1`): Se reporta fase completada y se deriva el flujo de transiciones (incluyendo `self.schedule_next_phase_after_settle()`).
    * **Fallo** (`result.error_code.val != 1`): Se llama a `self.abortar_objeto_actual()` registrando el código de error correspondiente.

### Movimientos Locales (Cartesianos / ExecuteTrajectory)
* **Registro de callbacks**:
  * En `ejecutar_movimiento_cartesiano_local(self, ...)` se hace la llamada asíncrona a la acción de ejecución de trayectorias de MoveIt:
    ```python
    future = self.execute_trajectory_client.send_goal_async(goal_msg)
    future.add_done_callback(self.cartesian_execute_goal_response_callback)
    ```
  * En `cartesian_execute_goal_response_callback(self, future)` se agrega el callback de resultado:
    ```python
    result_future.add_done_callback(self.cartesian_execute_result_callback)
    ```
* **Manejo del Resultado**:
  * En [cartesian_execute_result_callback(self, future)](file:///Ubuntu-22.04/home/sebas/ros2_ws/src/H1-UTEC-PFCII/scripts/mover_brazo_single.py#L2010):
    * **Éxito** (`result.error_code.val == 1`): Se reporta finalización exitosa y se bifurca según `self.cartesian_motion_context` para transicionar de fase (llamando a `self.schedule_next_phase_after_settle()`).
    * **Fallo** (`result.error_code.val != 1`): Llama a `self.abortar_objeto_actual()` reportando el fallo en la ejecución cartesiana.

---

## 7. Ejecución de Trayectorias sin Espera de Estabilización (Settle)

Las siguientes trayectorias y llamadas de secuencia omiten o no están programadas con la barrera de tiempo de estabilización (`post_motion_settle_time`):

1. **Bifurcaciones Internas de Transición en `move_group_result_callback`**:
   * **Fase 12 (Retirada de estante)**: Si `self.current_phase == 12` y `self.pending_phase12_shelf_retreat_motion` es verdadero, llama de forma inmediata a `self.ejecutar_fase_12_retirada_segura()`.
   * **Fase 12 (Retirada segmentada)**: Si `self.current_phase == 12` y `self.use_split_place_retreat` es verdadero, llama inmediatamente a `self.ejecutar_fase_12_retirada_segura()`.
   * **Fase 7 (Micro-lift de estante)**: Si `self.current_phase == 7` y `self.pending_phase7_lift_motion` es verdadero, ejecuta de inmediato `self.ejecutar_fase_7_retirada_vertical()`.
   * **Fase 9 (Inserción elevada de estante)**: Si `self.current_phase == 9` y `self.pending_phase9_shelf_insert_lifted_motion` es verdadero, planifica de inmediato el descenso cartesiano local.

2. **Bifurcaciones Internas de Transición en `cartesian_execute_result_callback`**:
   * **`phase7_lift_done`**: Llama inmediatamente a `self.ejecutar_fase_7_retirada_vertical()`.
   * **`phase9_shelf_insert_lifted_done`**: Llama inmediatamente a `self.ejecutar_movimiento_cartesiano_local(...)` para descender.
   * **`phase12_retreat_done`**: Llama inmediatamente a `self.ejecutar_fase_12_retirada_segura()`.

3. **Acciones de Apertura/Cierre de Mano**:
   * Los comandos de trayectoria articular de dedos enviados en la **Fase 1**, **Fase 5** y **Fase 10** a través de `self.hand_action_client` no están sujetos a la lógica del timer de estabilización del brazo (debido a que corresponden únicamente a articulaciones de los dedos y no implican desplazamiento del manipulador). Sin embargo, en la Fase 10 existe un retardo síncrono local `release_settle_time` previo a validar los joints de la mano.

---

## 8. Recomendación de Próximo Parche (Estabilidad y Seguimiento)

Para mitigar la inestabilidad de seguimiento y las oscilaciones físicas (que provocan desalineación con la pose planificada en MoveIt y el consecuente error `-17`), se recomienda aplicar los siguientes ajustes:

1. **Incrementar la frecuencia de control de ROS 2**:
   * Subir el parámetro `update_rate` de `controller_manager` en `ros2_controllers_mujoco.yaml` de **`20` Hz** a por lo menos **`100` Hz** o **`250` Hz** para reducir la latencia del lazo cerrado de control.

2. **Reforzar los parámetros dinámicos de los Actuadores en MuJoCo**:
   * Incrementar la ganancia proporcional (`kp`) de los actuadores de posición de las articulaciones de los brazos en `h1_2_mujoco_model.xml` a valores más rigurosos (p. ej., `kp="50"` o `kp="100"`), compensando con amortiguamiento físico directo en las articulaciones (`damping` de los joints correspondientes ajustado proporcionalmente a `5.0` o `10.0`) para evitar sobreoscilaciones destructivas.

3. **Agregar tolerancias a los controladores en ROS 2 Control**:
   * Declarar parámetros de tolerancia de seguimiento de trayectoria (`constraints` de posición) en cada articulación de los controladores de brazos en `ros2_controllers_mujoco.yaml` para asegurar que el controlador de trayectoria (`JointTrajectoryController`) no dé por finalizada la ejecución de forma abrupta si hay desfasamiento entre las poses de consigna y de estado.
