# Fase 1B: Controlador de Retención Dinámico (Hold Controller) con Pinocchio

Este documento detalla el diseño, la implementación, las validaciones y las correcciones de secuencia aplicadas al nodo de retención dinámica (`h1_2_dynamics_hold_controller.py`) para estabilizar los brazos del robot H1-2 mediante compensación de gravedad dinámica con Pinocchio.

---

## 1. Problema Detectado en el Arranque Inicial

En la versión original, al lanzar la simulación de la variante `_dynamics`:
1.  Los brazos del robot caían bruscamente bajo la gravedad durante los primeros segundos.
2.  *Causa:* El nodo hold capturaba la pose $q_{hold}$ inmediatamente al recibir el primer `/joint_states`. Sin embargo, los controladores de esfuerzo locales (`left_arm_effort_controller` y `right_arm_effort_controller`) aún no se habían terminado de registrar o no tenían suscriptores activos. Como resultado, las consignas de torque se perdían y el robot caía antes de que la retención pudiera ejercer fuerza.

---

## 2. Corrección de la Secuencia de Arranque

Para mitigar la caída del brazo y garantizar una captura limpia y sin perturbaciones, se implementaron dos medidas correctivas:

### A. Secuenciación y Minimización de Delays en el Lanzamiento
En el launch [h1_2_mujoco_wide_camera_dynamics_hold.launch.py](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/launch/h1_2_mujoco_wide_camera_dynamics_hold.launch.py), se restructuró la carga para que los controladores esenciales se levanten lo antes posible:
1.  **`joint_state_broadcaster`:** Carga a los $2.0\text{ s}$.
2.  **`left_arm_effort_controller` / `right_arm_effort_controller`:** Carga a los $3.0\text{ - }3.5\text{ s}$.
3.  **`h1_2_dynamics_hold_controller`:** Carga a los $4.5\text{ s}$ (justo después de que los controladores de esfuerzo están listos).
4.  **`torso` / `manos`:** Carga retrasada a los $5.5\text{ - }6.5\text{ s}$ (no interfieren en la retención inicial).

### B. Condición de Captura Resiliente en el Nodo
En el script [h1_2_dynamics_hold_controller.py](file:///home/sebas/ros2_ws/src/h1_2_mujoco_bringup/scripts/h1_2_dynamics_hold_controller.py#L160-L211), la pose $q_{hold}$ **solo se captura** cuando se cumplen las tres condiciones simultáneamente:
1.  El tópico `/joint_states` está activo y contiene la posición de las 14 juntas de los brazos.
2.  **Ambos publicadores de esfuerzos tienen suscriptores activos** (`get_subscription_count() >= 1`), lo cual garantiza que los controladores de bajo nivel de ROS están en línea y escuchando.
3.  Ha transcurrido el tiempo `hold_capture_delay_sec` ($0.5\text{ s}$) **únicamente después** de que se cumplieron las dos condiciones anteriores.

Si no hay suscriptores o no ha llegado la telemetría, el nodo publica torque cero de seguridad y loguea periódicamente:
`[HOLD CONTROL] Waiting for effort controller subscribers...`

---

## 3. Fase 1C-B — Reset automático después de hold inicial

Para erradicar por completo cualquier caída de los brazos (sagging) durante la inicialización de los controladores y del nodo ROS 2, se ha implementado un sistema de arranque y recuperación pausada.

### A. Diagnóstico de Falla de Pause-First (Deadlock)
Al intentar pausar MuJoCo antes de que ROS 2 control active los controladores (`pause-first` en el launch):
1.  La simulación física se congela en el frame 0, pero `controller_manager` (ROS 2 control) depende del avance del tiempo de simulación y de los pasos de física para inicializar y hacer el switch del hardware de los controladores.
2.  Dado que MuJoCo está pausado, el tiempo de simulación no avanza, lo cual provoca que `switch_controller` haga un *timeout* y falle.
3.  **Conclusión:** ROS 2 control necesita que la física y el tiempo de simulación avancen inicialmente para levantar los controladores.

### B. Solución: Reset de Mundo Post-Arranque (`--reset-after-hold-ready`)
Para evadir este deadlock, se diseñó la siguiente secuencia automatizada:
1.  **Arranque Libre:** MuJoCo y los controladores de ROS 2 se inicializan normalmente con la física corriendo. Los brazos caen ligeramente.
2.  **Captura y Estabilización Inicial:** El nodo hold dinámico espera a que la plomería de esfuerzos esté lista, captura $q_{hold}$, calcula y publica el primer torque dinámico con Pinocchio. En este momento, `hold_ready` pasa a ser `True`.
3.  **Pausa de Simulación:** El manager de pausa, al leer `hold_ready=True`, pausa inmediatamente la física llamando a `/mujoco_ros2_control_node/set_pause` (`paused=True`).
4.  **Reset de Mundo:** Llama al servicio `/mujoco_ros2_control_node/reset_world`. Esto devuelve instantáneamente a todo el entorno físico de MuJoCo (incluyendo los brazos y objetos) a sus poses iniciales del frame 0, pero manteniendo los controladores de ROS 2 en estado activo/excitado.
5.  **Recaptura de Hold (`recapture_hold`):** El manager llama al nuevo servicio `/h1_2_dynamics_hold_controller/recapture_hold`. Este servicio del hold controller lee las nuevas posiciones de junta (las posiciones nominales de MuJoCo post-reset), recaptura $q_{hold}$, publica temporalmente `hold_ready=False`, calcula los torques dinámicos iniciales sobre esta pose perfecta y los inyecta. Tras inyectar la primera muestra, publica `hold_ready=True`.
6.  **Despausa y Simulación Estable:** El manager detecta el segundo `hold_ready=True` (post-recaptura) y reanuda la simulación llamando a `paused=False`.
7.  **Resultado Visual:** Los brazos se sostienen suspendidos perfectamente en la pose inicial nominal del frame 0 sin caída alguna y sin tirones bruscos.

### C. Despausa Manual de Emergencia
Si por algún motivo excepcional (ej. retraso excesivo en el arranque) el sistema se queda en estado de pausa permanente, puedes reanudar la simulación manualmente en cualquier momento ejecutando el siguiente comando en tu terminal:
```bash
ros2 service call /mujoco_ros2_control_node/set_pause mujoco_ros2_control_msgs/srv/SetPause "{paused: false}"
```

---

## 4. Parámetros de Configuración del Nodo

El nodo expone los siguientes parámetros a través del sistema de parámetros de ROS 2:
*   `pinocchio_urdf_path`: Ruta absoluta del URDF dinámico utilizado para construir el modelo de Pinocchio.
*   `control_rate_hz` (default `100.0`): Frecuencia del lazo de control.
*   `gravity_scale` (default `1.0`): Factor multiplicativo para la compensación de gravedad.
*   `torque_sign` (default `1.0`): Dirección del torque generado (permite invertir el signo en caso de discrepancias de sentido).
*   `enable_pd` (default `True`): Activa/desactiva el bucle PD (permite probar compensación de gravedad pura sin retroalimentación).
*   `hold_capture_delay_sec` (default `0.5`): Retraso de asentamiento físico del robot antes de congelar $q_{hold}$.
*   `log_rate_hz` (default `1.0`): Frecuencia del log de telemetría a $1\text{ Hz}$.

### Ganancias Articulares y Límites Nominales
Las ganancias iniciales conservadoras sintonizadas son:
*   **Shoulder Pitch/Roll:** $K_p = 20.0$, $K_d = 2.0$ (Límite: $\pm 40\text{ N}\cdot\text{m}$)
*   **Shoulder Yaw:** $K_p = 12.0$, $K_d = 1.5$ (Límite: $\pm 18\text{ N}\cdot\text{m}$)
*   **Elbow:** $K_p = 15.0$, $K_d = 1.5$ (Límite: $\pm 18\text{ N}\cdot\text{m}$)
*   **Wrist Roll/Pitch/Yaw:** $K_p = 5.0$, $K_d = 0.5$ (Límite: $\pm 19\text{ N}\cdot\text{m}$)

---

## 5. Comandos de Ejecución y Verificación

### Lanzamiento de la Variante Completa (Simulador + Secuencia de Recuperación)
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch h1_2_mujoco_bringup h1_2_mujoco_wide_camera_dynamics_hold.launch.py
```

### Verificación de Telemetría
Puedes monitorear el estado del lazo y los torques publicados visualizando la consola de lanzamiento o bien escuchando los tópicos de esfuerzo:
```bash
# Ver torques comandados al brazo izquierdo
ros2 topic echo /left_arm_effort_controller/commands

# Ver torques comandados al brazo derecho
ros2 topic echo /right_arm_effort_controller/commands
```

---

## 6. Siguiente Fase Recomendada

Proceder a la **Fase 2**: Integrar el Action Server `FollowJointTrajectory` en el nodo de control. Esto permitirá que el nodo intercepte las trayectorias dinámicas de MoveIt y guíe suavemente al robot calculando $\tau_{grav} + \tau_{pd}$ sobre trayectorias dinámicas variables en lugar de una pose fija estática.
