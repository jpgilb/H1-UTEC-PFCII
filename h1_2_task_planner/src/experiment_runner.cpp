#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

using namespace std::chrono_literals;

struct Condition
{
  std::string name;
  double dx;
  double dy_abs;
  double dz;
};

static void ensureDirectory(const std::string& path)
{
  mkdir(path.c_str(), 0775);
}

static double distanceMeters(
  const geometry_msgs::msg::Point& a,
  const geometry_msgs::msg::Point& b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

class ExperimentRunner
{
public:
  ExperimentRunner(const rclcpp::Node::SharedPtr& node)
  : node_(node)
  {
    node_->declare_parameter<std::string>("planning_group", "left_arm");
    node_->declare_parameter<std::string>("base_frame", "world");
    node_->declare_parameter<std::string>("ee_link", "L_hand_base_link");
    node_->declare_parameter<std::string>("output_dir", "/tmp/h1_2_results");
    node_->declare_parameter<int>("repetitions", 3);
    node_->declare_parameter<bool>("execute", true);
    node_->declare_parameter<double>("planning_time", 5.0);
    node_->declare_parameter<int>("num_planning_attempts", 10);
    node_->declare_parameter<double>("position_tolerance", 0.01);
    node_->declare_parameter<double>("orientation_tolerance", 0.05);
    node_->declare_parameter<double>("velocity_scaling", 0.2);
    node_->declare_parameter<double>("acceleration_scaling", 0.2);
    node_->declare_parameter<double>("offset_scale", 1.0);

    planning_group_ = node_->get_parameter("planning_group").as_string();
    base_frame_ = node_->get_parameter("base_frame").as_string();
    ee_link_ = node_->get_parameter("ee_link").as_string();
    output_dir_ = node_->get_parameter("output_dir").as_string();
    repetitions_ = node_->get_parameter("repetitions").as_int();
    execute_ = node_->get_parameter("execute").as_bool();
    planning_time_ = node_->get_parameter("planning_time").as_double();
    num_planning_attempts_ = node_->get_parameter("num_planning_attempts").as_int();
    position_tolerance_ = node_->get_parameter("position_tolerance").as_double();
    orientation_tolerance_ = node_->get_parameter("orientation_tolerance").as_double();
    velocity_scaling_ = node_->get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = node_->get_parameter("acceleration_scaling").as_double();
    offset_scale_ = node_->get_parameter("offset_scale").as_double();

    ensureDirectory(output_dir_);

    event_pub_ = node_->create_publisher<std_msgs::msg::String>("/experiment_event", 10);
    marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);

    summary_csv_.open(output_dir_ + "/experiment_summary.csv", std::ios::out);
    trajectory_csv_.open(output_dir_ + "/planned_joint_trajectory.csv", std::ios::out);

    summary_csv_
      << "condition,trial,success,planning_time_s,execution_time_s,"
      << "planned_duration_s,trajectory_points,cartesian_error_mm,joint_error_rad,"
      << "target_x,target_y,target_z,final_x,final_y,final_z\n";

    trajectory_csv_
      << "condition,trial,time_s,joint_name,desired_position_rad,desired_velocity_rad_s\n";

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      node_, planning_group_);

    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setEndEffectorLink(ee_link_);
    move_group_->setGoalPositionTolerance(position_tolerance_);
    move_group_->setGoalOrientationTolerance(orientation_tolerance_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setNumPlanningAttempts(num_planning_attempts_);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(node_->get_logger(), "ExperimentRunner inicializado.");
    RCLCPP_INFO(node_->get_logger(), "Grupo: %s | EE: %s", planning_group_.c_str(), ee_link_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Resultados en: %s", output_dir_.c_str());
  }

  void run()
  {
    rclcpp::sleep_for(2s);

    std::vector<double> initial_joints = move_group_->getCurrentJointValues();

    const bool is_right_arm = planning_group_.find("right") != std::string::npos;
    const double y_sign = is_right_arm ? -1.0 : 1.0;

    std::vector<Condition> conditions = {
      {"P1_cercana",     0.03 * offset_scale_, 0.02 * offset_scale_, 0.02 * offset_scale_},
      {"P2_intermedia",  0.07 * offset_scale_, 0.04 * offset_scale_, 0.04 * offset_scale_},
      {"P3_borde",      0.11 * offset_scale_, 0.06 * offset_scale_, 0.06 * offset_scale_}
    };

    for (const auto& condition : conditions) {
      for (int trial = 1; trial <= repetitions_; ++trial) {
        RCLCPP_INFO(
          node_->get_logger(),
          "Ejecutando condición %s | repetición %d",
          condition.name.c_str(),
          trial);

        resetToInitialState(initial_joints);

        geometry_msgs::msg::PoseStamped initial_pose = move_group_->getCurrentPose(ee_link_);
        geometry_msgs::msg::PoseStamped target = makeTarget(initial_pose, condition, y_sign);

        publishMarker(target);

        runSingleTrial(condition.name, trial, target);

        rclcpp::sleep_for(1s);
      }
    }

    summary_csv_.close();
    trajectory_csv_.close();

    RCLCPP_INFO(node_->get_logger(), "Experimento finalizado.");
  }

private:
  void resetToInitialState(const std::vector<double>& initial_joints)
  {
    if (!execute_) {
      return;
    }

    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(initial_joints);

    moveit::planning_interface::MoveGroupInterface::Plan reset_plan;
    bool reset_success = static_cast<bool>(move_group_->plan(reset_plan));

    if (reset_success) {
      auto result = move_group_->execute(reset_plan);
      if (result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(node_->get_logger(), "No se pudo ejecutar el retorno al estado inicial.");
      }
    } else {
      RCLCPP_WARN(node_->get_logger(), "No se pudo planificar el retorno al estado inicial.");
    }

    rclcpp::sleep_for(500ms);
  }

  geometry_msgs::msg::PoseStamped makeTarget(
    const geometry_msgs::msg::PoseStamped& initial_pose,
    const Condition& condition,
    double y_sign)
  {
    geometry_msgs::msg::PoseStamped target;
    target.header.frame_id = base_frame_;
    target.header.stamp = node_->now();

    target.pose = initial_pose.pose;
    target.pose.position.x += condition.dx;
    target.pose.position.y += y_sign * condition.dy_abs;
    target.pose.position.z += condition.dz;

    return target;
  }

  void publishMarker(const geometry_msgs::msg::PoseStamped& target)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = target.header;
    marker.ns = "experiment_goal_pose";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose = target.pose;
    marker.pose.position.z += 0.05;

    marker.scale.x = 0.18;
    marker.scale.y = 0.035;
    marker.scale.z = 0.035;

    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker.lifetime = rclcpp::Duration::from_seconds(0.0);

    marker_pub_->publish(marker);
  }

  void publishEvent(const std::string& event, const std::string& condition, int trial)
  {
    std_msgs::msg::String msg;
    msg.data = event + "," + condition + "," + std::to_string(trial);
    event_pub_->publish(msg);
  }

  double computeJointError(
    const moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    const auto& joint_traj = plan.trajectory_.joint_trajectory;

    if (joint_traj.points.empty()) {
      return 0.0;
    }

    const auto& final_point = joint_traj.points.back();
    const auto current_joint_values = move_group_->getCurrentJointValues();
    const auto current_joint_names = move_group_->getJointNames();

    std::map<std::string, double> current_map;
    for (size_t i = 0; i < current_joint_names.size() && i < current_joint_values.size(); ++i) {
      current_map[current_joint_names[i]] = current_joint_values[i];
    }

    double sq_error = 0.0;
    int count = 0;

    for (size_t i = 0; i < joint_traj.joint_names.size() && i < final_point.positions.size(); ++i) {
      const auto& joint_name = joint_traj.joint_names[i];

      if (current_map.find(joint_name) == current_map.end()) {
        continue;
      }

      const double error = final_point.positions[i] - current_map[joint_name];
      sq_error += error * error;
      count++;
    }

    if (count == 0) {
      return 0.0;
    }

    return std::sqrt(sq_error);
  }

  void writePlannedTrajectory(
    const std::string& condition,
    int trial,
    const moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    const auto& joint_traj = plan.trajectory_.joint_trajectory;

    for (const auto& point : joint_traj.points) {
      const double t = rclcpp::Duration(point.time_from_start).seconds();

      for (size_t j = 0; j < joint_traj.joint_names.size(); ++j) {
        const std::string& joint_name = joint_traj.joint_names[j];

        double position = 0.0;
        double velocity = 0.0;

        if (j < point.positions.size()) {
          position = point.positions[j];
        }

        if (j < point.velocities.size()) {
          velocity = point.velocities[j];
        }

        trajectory_csv_
          << condition << ","
          << trial << ","
          << std::fixed << std::setprecision(6) << t << ","
          << joint_name << ","
          << position << ","
          << velocity << "\n";
      }
    }

    trajectory_csv_.flush();
  }

  void runSingleTrial(
    const std::string& condition,
    int trial,
    const geometry_msgs::msg::PoseStamped& target)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(target, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const auto planning_start = node_->now();
    bool plan_success = static_cast<bool>(move_group_->plan(plan));
    const double planning_time_s = (node_->now() - planning_start).seconds();

    double execution_time_s = 0.0;
    double planned_duration_s = 0.0;
    int trajectory_points = 0;
    double cartesian_error_mm = 0.0;
    double joint_error_rad = 0.0;
    bool execution_success = false;

    geometry_msgs::msg::Pose final_pose;

    if (plan_success) {
      trajectory_points = static_cast<int>(plan.trajectory_.joint_trajectory.points.size());

      if (!plan.trajectory_.joint_trajectory.points.empty()) {
        planned_duration_s = rclcpp::Duration(
          plan.trajectory_.joint_trajectory.points.back().time_from_start).seconds();
      }

      writePlannedTrajectory(condition, trial, plan);

      if (execute_) {
        publishEvent("START", condition, trial);

        const auto execution_start = node_->now();
        auto result = move_group_->execute(plan);
        execution_time_s = (node_->now() - execution_start).seconds();

        execution_success = (result == moveit::core::MoveItErrorCode::SUCCESS);

        publishEvent("END", condition, trial);
      } else {
        execution_success = true;
      }

      rclcpp::sleep_for(500ms);

      final_pose = move_group_->getCurrentPose(ee_link_).pose;
      cartesian_error_mm = 1000.0 * distanceMeters(target.pose.position, final_pose.position);
      joint_error_rad = computeJointError(plan);
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "No se encontró plan válido para %s | repetición %d",
        condition.c_str(),
        trial);

      final_pose = move_group_->getCurrentPose(ee_link_).pose;
      cartesian_error_mm = 1000.0 * distanceMeters(target.pose.position, final_pose.position);
    }

    const bool success = plan_success && execution_success;

    summary_csv_
      << condition << ","
      << trial << ","
      << (success ? 1 : 0) << ","
      << std::fixed << std::setprecision(6)
      << planning_time_s << ","
      << execution_time_s << ","
      << planned_duration_s << ","
      << trajectory_points << ","
      << cartesian_error_mm << ","
      << joint_error_rad << ","
      << target.pose.position.x << ","
      << target.pose.position.y << ","
      << target.pose.position.z << ","
      << final_pose.position.x << ","
      << final_pose.position.y << ","
      << final_pose.position.z << "\n";

    summary_csv_.flush();

    move_group_->clearPoseTargets();

    RCLCPP_INFO(
      node_->get_logger(),
      "Resultado %s | rep %d | success=%d | plan=%.3fs | exec=%.3fs | error=%.3f mm",
      condition.c_str(),
      trial,
      success ? 1 : 0,
      planning_time_s,
      execution_time_s,
      cartesian_error_mm);
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  std::ofstream summary_csv_;
  std::ofstream trajectory_csv_;

  std::string planning_group_;
  std::string base_frame_;
  std::string ee_link_;
  std::string output_dir_;

  int repetitions_;
  bool execute_;

  double planning_time_;
  int num_planning_attempts_;
  double position_tolerance_;
  double orientation_tolerance_;
  double velocity_scaling_;
  double acceleration_scaling_;
  double offset_scale_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("experiment_runner");

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() {
    executor.spin();
  });

  ExperimentRunner runner(node);
  runner.run();

  executor.cancel();

  if (spinner.joinable()) {
    spinner.join();
  }

  rclcpp::shutdown();
  return 0;
}
