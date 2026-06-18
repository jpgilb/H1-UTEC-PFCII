#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

using namespace std::chrono_literals;

static void ensureDirectory(const std::string& path)
{
  mkdir(path.c_str(), 0775);
}

static double jointErrorNorm(
  const std::map<std::string, double>& target,
  const std::vector<std::string>& current_names,
  const std::vector<double>& current_values)
{
  std::map<std::string, double> current;

  for (size_t i = 0; i < current_names.size() && i < current_values.size(); ++i) {
    current[current_names[i]] = current_values[i];
  }

  double sq = 0.0;
  int n = 0;

  for (const auto& [joint, target_value] : target) {
    if (current.find(joint) == current.end()) {
      continue;
    }

    const double error = target_value - current[joint];
    sq += error * error;
    n++;
  }

  if (n == 0) {
    return -1.0;
  }

  return std::sqrt(sq);
}

class BimanualCoordinatedExecutor
{
public:
  BimanualCoordinatedExecutor(const rclcpp::Node::SharedPtr& node)
  : node_(node)
  {
    node_->declare_parameter<std::string>("planning_group", "both_arms");
    node_->declare_parameter<std::string>("output_dir", "/tmp/h1_2_bimanual_results");
    node_->declare_parameter<int>("repetitions", 3);
    node_->declare_parameter<bool>("execute", true);
    node_->declare_parameter<double>("planning_time", 8.0);
    node_->declare_parameter<int>("num_planning_attempts", 20);
    node_->declare_parameter<double>("velocity_scaling", 0.15);
    node_->declare_parameter<double>("acceleration_scaling", 0.15);

    planning_group_ = node_->get_parameter("planning_group").as_string();
    output_dir_ = node_->get_parameter("output_dir").as_string();
    repetitions_ = node_->get_parameter("repetitions").as_int();
    execute_ = node_->get_parameter("execute").as_bool();
    planning_time_ = node_->get_parameter("planning_time").as_double();
    num_planning_attempts_ = node_->get_parameter("num_planning_attempts").as_int();
    velocity_scaling_ = node_->get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = node_->get_parameter("acceleration_scaling").as_double();

    ensureDirectory(output_dir_);

    event_pub_ = node_->create_publisher<std_msgs::msg::String>("/experiment_event", 10);

    csv_.open(output_dir_ + "/bimanual_joint_summary.csv", std::ios::out);
    csv_ << "condition,trial,success,planning_time_s,execution_time_s,"
         << "planned_duration_s,trajectory_points,joint_error_norm_rad\n";

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      node_, planning_group_);

    move_group_->setPlanningTime(planning_time_);
    move_group_->setNumPlanningAttempts(num_planning_attempts_);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(node_->get_logger(), "Ejecutor bimanual articular inicializado.");
    RCLCPP_INFO(node_->get_logger(), "Grupo: %s", planning_group_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Resultados en: %s", output_dir_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Publicando eventos en /experiment_event.");
  }

  void run()
  {
    rclcpp::sleep_for(2s);

    std::vector<std::string> conditions = {
      "B1_apertura_lateral",
      "B2_apertura_con_flexion",
      "B3_apertura_flexion_y_avance"
    };

    for (const auto& condition : conditions) {
      for (int trial = 1; trial <= repetitions_; ++trial) {
        RCLCPP_INFO(
          node_->get_logger(),
          "Condición %s | repetición %d",
          condition.c_str(),
          trial);

        runSingleTrial(condition, trial);
        rclcpp::sleep_for(1s);
      }
    }

    csv_.close();

    RCLCPP_INFO(node_->get_logger(), "Pruebas bimanuales articulares finalizadas.");
  }

private:
  void publishEvent(const std::string& event, const std::string& condition, int trial)
  {
    std_msgs::msg::String msg;
    msg.data = event + "," + condition + "," + std::to_string(trial);
    event_pub_->publish(msg);

    RCLCPP_INFO(
      node_->get_logger(),
      "Evento publicado: %s",
      msg.data.c_str());
  }

  std::map<std::string, double> getCurrentJointMap()
  {
    std::map<std::string, double> joint_map;

    auto names = move_group_->getJointNames();
    auto values = move_group_->getCurrentJointValues();

    for (size_t i = 0; i < names.size() && i < values.size(); ++i) {
      joint_map[names[i]] = values[i];
    }

    return joint_map;
  }

  std::map<std::string, double> makeTarget(
    const std::string& condition,
    const std::map<std::string, double>& current)
  {
    auto target = current;

    /*
      Movimiento coordinado seguro:
      - ambos brazos se mueven de forma simultánea;
      - los hombros se abren hacia lados opuestos;
      - los codos se flexionan levemente;
      - no se cruzan los brazos frente al torso.

      Nota: los signos pueden requerir ajuste si la convención articular del modelo
      genera una dirección diferente a la esperada.
    */

    if (condition == "B1_apertura_lateral") {
      addIfExists(target, "left_shoulder_roll_joint",  0.10);
      addIfExists(target, "right_shoulder_roll_joint", -0.10);
    }

    if (condition == "B2_apertura_con_flexion") {
      addIfExists(target, "left_shoulder_roll_joint",  0.14);
      addIfExists(target, "right_shoulder_roll_joint", -0.14);

      addIfExists(target, "left_elbow_joint",  0.18);
      addIfExists(target, "right_elbow_joint", 0.18);
    }

    if (condition == "B3_apertura_flexion_y_avance") {
      addIfExists(target, "left_shoulder_roll_joint",  0.18);
      addIfExists(target, "right_shoulder_roll_joint", -0.18);

      addIfExists(target, "left_shoulder_pitch_joint",  -0.10);
      addIfExists(target, "right_shoulder_pitch_joint", -0.10);

      addIfExists(target, "left_elbow_joint",  0.25);
      addIfExists(target, "right_elbow_joint", 0.25);
    }

    return target;
  }

  void addIfExists(
    std::map<std::string, double>& target,
    const std::string& joint_name,
    double delta)
  {
    if (target.find(joint_name) != target.end()) {
      target[joint_name] += delta;
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "La articulación %s no existe en el grupo %s",
        joint_name.c_str(),
        planning_group_.c_str());
    }
  }

  void runSingleTrial(const std::string& condition, int trial)
  {
    move_group_->setStartStateToCurrentState();

    auto current_map = getCurrentJointMap();
    auto target_map = makeTarget(condition, current_map);

    move_group_->clearPoseTargets();

    bool target_ok = move_group_->setJointValueTarget(target_map);

    if (!target_ok) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "No se pudo asignar objetivo articular para %s | rep %d",
        condition.c_str(),
        trial);

      writeResult(condition, trial, false, 0.0, 0.0, 0.0, 0, -1.0);
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const auto planning_start = node_->now();
    bool plan_success = static_cast<bool>(move_group_->plan(plan));
    const double planning_time_s = (node_->now() - planning_start).seconds();

    double execution_time_s = 0.0;
    double planned_duration_s = 0.0;
    int trajectory_points = 0;
    bool execution_success = false;
    double joint_error_rad = -1.0;

    if (plan_success) {
      trajectory_points =
        static_cast<int>(plan.trajectory_.joint_trajectory.points.size());

      if (!plan.trajectory_.joint_trajectory.points.empty()) {
        planned_duration_s = rclcpp::Duration(
          plan.trajectory_.joint_trajectory.points.back().time_from_start).seconds();
      }

      if (execute_) {
        publishEvent("START", condition, trial);
        rclcpp::sleep_for(100ms);

        const auto execution_start = node_->now();
        auto result = move_group_->execute(plan);
        execution_time_s = (node_->now() - execution_start).seconds();

        execution_success =
          (result == moveit::core::MoveItErrorCode::SUCCESS);

        publishEvent("END", condition, trial);
        rclcpp::sleep_for(100ms);
      } else {
        execution_success = true;
      }

      rclcpp::sleep_for(500ms);

      auto current_names = move_group_->getJointNames();
      auto current_values = move_group_->getCurrentJointValues();

      joint_error_rad =
        jointErrorNorm(target_map, current_names, current_values);
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "No se encontró plan válido para %s | repetición %d",
        condition.c_str(),
        trial);
    }

    const bool success = plan_success && execution_success;

    writeResult(
      condition,
      trial,
      success,
      planning_time_s,
      execution_time_s,
      planned_duration_s,
      trajectory_points,
      joint_error_rad);

    move_group_->clearPoseTargets();

    RCLCPP_INFO(
      node_->get_logger(),
      "Resultado %s | rep %d | success=%d | plan=%.3f s | exec=%.3f s | puntos=%d | err_joint=%.6f rad",
      condition.c_str(),
      trial,
      success ? 1 : 0,
      planning_time_s,
      execution_time_s,
      trajectory_points,
      joint_error_rad);
  }

  void writeResult(
    const std::string& condition,
    int trial,
    bool success,
    double planning_time_s,
    double execution_time_s,
    double planned_duration_s,
    int trajectory_points,
    double joint_error_rad)
  {
    csv_ << condition << ","
         << trial << ","
         << (success ? 1 : 0) << ","
         << planning_time_s << ","
         << execution_time_s << ","
         << planned_duration_s << ","
         << trajectory_points << ","
         << joint_error_rad << "\n";

    csv_.flush();
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub_;

  std::ofstream csv_;

  std::string planning_group_;
  std::string output_dir_;

  int repetitions_;
  bool execute_;

  double planning_time_;
  int num_planning_attempts_;
  double velocity_scaling_;
  double acceleration_scaling_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("bimanual_coordinated_executor");

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() {
    executor.spin();
  });

  BimanualCoordinatedExecutor runner(node);
  runner.run();

  executor.cancel();

  if (spinner.joinable()) {
    spinner.join();
  }

  rclcpp::shutdown();
  return 0;
}
