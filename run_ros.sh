T1:

source /opt/ros/humble/setup.bash
source ~/fr3_ws/install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=dont-care use_fake_hardware:=true

T2:
source /opt/ros/humble/setup.bash
source ~/fr3_ws/install/setup.bash
source ~/Documents/frankaSim/franka_ws/install/setup.bash

ros2 run franka_basics ee_delta_to_joint_command --ros-args \
  -p group_name:=fr3_arm \
  -p ee_link:=fr3_link8 \
  -p base_frame:=fr3_link0 \
  -p cmd_topic:=/joint_command \
  -p fixed_orientation_xyzw:="[0.0,1.0,0.0,0.0]" \
  -p max_step_m:=0.05 \
  -p xyz_min:="[0.2,-0.5,0.1]" -p xyz_max:="[0.8,0.5,0.7]"

  T3:
source /opt/ros/humble/setup.bash
source ~/Documents/frankaSim/franka_ws/install/setup.bash

ros2 run franka_basics publish_action_stream --ros-args \
  -p input_path:=/tmp/actions.txt \
  -p topic:=/ee_delta \
  -p hz:=20.0 \
  -p scale_m_per_unit:=0.01 \
  -p xyz_indices:="[0,1,2]" \
  -p skip_bad_lines:=true