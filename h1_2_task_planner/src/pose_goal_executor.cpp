#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

class PoseGoalExecutor : public rclcpp::Node
{
public:
  PoseGoalExecutor(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("pose_goal_executor", options)
  {
    this->declare_parameter<std::string>("planning_group", "left_arm");
    this->declare_parameter<std::string>("base_frame", "world");
    this->declare_parameter<std::string>("ee_link", "L_hand_base_link");
    this->declare_parameter<double>("position_tolerance", 0.01);
    this->declare_parameter<double>("orientation_tolerance", 0.05);
    this->declare_parameter<double>("planning_time", 5.0);
    this->declare_parameter<int>("num_planning_attempts", 10);
    this->declare_parameter<bool>("execute", true);

    planning_group_ = this->get_parameter("planning_group").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    ee_link_ = this->get_parameter("ee_link").as_string();
    position_tolerance_ = this->get_parameter("position_tolerance").as_double();
    orientation_tolerance_ = this->get_parameter("orientation_tolerance").as_double();
    planning_time_ = this->get_parameter("planning_time").as_double();
    num_planning_attempts_ = this->get_parameter("num_planning_attempts").as_int();
    execute_ = this->get_parameter("execute").as_bool();

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);

    sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/target_pose", 10,
      std::bind(&PoseGoalExecutor::targetCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "PoseGoalExecutor listo.");
    RCLCPP_INFO(this->get_logger(), "Grupo: %s | EE link: %s", planning_group_.c_str(), ee_link_.c_str());
  }

  void initMoveGroup()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);

    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setEndEffectorLink(ee_link_);
    move_group_->setGoalPositionTolerance(position_tolerance_);
    move_group_->setGoalOrientationTolerance(orientation_tolerance_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setNumPlanningAttempts(num_planning_attempts_);
    move_group_->setMaxVelocityScalingFactor(0.2);
    move_group_->setMaxAccelerationScalingFactor(0.2);

    RCLCPP_INFO(this->get_logger(), "MoveGroup inicializado.");
  }

private:
  void publishMarker(const geometry_msgs::msg::PoseStamped& target)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = target.header;
    marker.ns = "goal_pose";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    auto marker_pose = target.pose;
    marker_pose.position.z += 0.05;   // para que no quede tapado por la mano/brazo
    marker.pose = marker_pose;

    marker.scale.x = 0.18;  // largo de la flecha
    marker.scale.y = 0.035; // diámetro del cuerpo
    marker.scale.z = 0.035; // diámetro de la punta

    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker.lifetime = rclcpp::Duration::from_seconds(0.0);

    marker_pub_->publish(marker);
  }

  void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!move_group_) {
      RCLCPP_ERROR(this->get_logger(), "MoveGroup no inicializado.");
      return;
    }

    geometry_msgs::msg::PoseStamped target = *msg;

    if (target.header.frame_id.empty()) {
      target.header.frame_id = base_frame_;
    }

    publishMarker(target);

    RCLCPP_INFO(
      this->get_logger(),
      "Objetivo recibido: frame=%s pos=(%.3f, %.3f, %.3f)",
      target.header.frame_id.c_str(),
      target.pose.position.x,
      target.pose.position.y,
      target.pose.position.z);

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(target, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_WARN(this->get_logger(), "No se encontró plan válido.");
      move_group_->clearPoseTargets();
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Plan encontrado.");

    if (execute_) {
      auto result = move_group_->execute(plan);
      if (result == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "Trayectoria ejecutada.");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Falló la ejecución.");
      }
    }

    move_group_->clearPoseTargets();
  }

  std::string planning_group_;
  std::string base_frame_;
  std::string ee_link_;
  double position_tolerance_;
  double orientation_tolerance_;
  double planning_time_;
  int num_planning_attempts_;
  bool execute_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PoseGoalExecutor>();
  node->initMoveGroup();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
