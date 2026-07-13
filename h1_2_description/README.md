# h1_2_description

A ROS 2 package containing robot description files, including URDF models and launch configurations for the H1-2 humanoid robot. This model is adapted from the `unitreerobotics/unitree_ros` github repo.

## Package Contents

- `urdf/` - URDF (Unified Robot Description Format) files
- `mjcf/` - Model for Mujoco 
- `launch/` - Launch files for robot visualization
- `meshes/` - 3D mesh files for robot visualization

## Usage

The H1-2 robot without hands (27 dof: 6 per leg, 7 per arm, 1 torso) with sliders is visualized in RViz using:

```bash
ros2 launch h1_2_description display-handless-sliders.launch.py
```

The H1-2 robot (with hands) is visualized with sliders in RViz using:

```bash
ros2 launch h1_2_description display-sliders.launch.py
```

## Dependencies

- ROS 2 (Humble or later)
- urdfdom
- joint_state_publisher_gui

## Notes

- The `h1_2.urdf` and `h1_2_handless.urdf` are the original unitree models. Only a name for each material was added for compatibility with ROS2.
- The `h1_2_bright.urdf` and `h1_2_handless_bright.urdf` are the same models as the `h1_2.urdf` and `h1_2_handless.urdf`, respectively, but with a brighter material for a better visualization in RViz