#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <unitree_hg/msg/low_state.hpp>

#include <string>
#include <vector>

class LowStateToJointStates : public rclcpp::Node
{
public:
  LowStateToJointStates()
  : Node("lowstate_to_jointstates")
  {
    const std::string lowstate_topic = "lowstate";     // lf/lowstate
    const std::string joint_states_topic = "/joint_states";

    // Joint name mapping
    joint_names_ = {
        "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    };

    pub_ = this->create_publisher<sensor_msgs::msg::JointState>(joint_states_topic, rclcpp::QoS(10));
    sub_ = this->create_subscription<unitree_hg::msg::LowState>(
      lowstate_topic,
      rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
      std::bind(&LowStateToJointStates::on_lowstate, this, std::placeholders::_1)
    );

    joint_msg_.name = joint_names_;
    joint_msg_.position.resize(NumJoints);
    joint_msg_.velocity.resize(NumJoints);
    joint_msg_.effort.resize(NumJoints);

    RCLCPP_INFO(this->get_logger(),
      "Subscribed to '%s', publishing '%s' with %zu joints",
      lowstate_topic.c_str(), joint_states_topic.c_str(), joint_names_.size()
    );
  }

private:
  static constexpr size_t NumJoints = 27;

  void on_lowstate(const unitree_hg::msg::LowState::SharedPtr msg)
  {
    if (msg->motor_state.size() < NumJoints) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "LowState motor_state has %zu entries (< %zu). Not publishing.",
        msg->motor_state.size(), NumJoints
      );
      return;
    }

    joint_msg_.header.stamp = this->now();

    for (size_t i = 0; i < NumJoints; ++i) {
      joint_msg_.position[i] = msg->motor_state[i].q;        // rad
      joint_msg_.velocity[i] = msg->motor_state[i].dq;       // rad/s
      joint_msg_.effort[i]   = msg->motor_state[i].tau_est;  // Nm
    }

    pub_->publish(joint_msg_);
  }

  std::vector<std::string> joint_names_;
  sensor_msgs::msg::JointState joint_msg_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr sub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LowStateToJointStates>());
  rclcpp::shutdown();
  return 0;
}
