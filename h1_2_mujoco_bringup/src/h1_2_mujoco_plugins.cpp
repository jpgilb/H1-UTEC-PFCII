#include <mutex>
#include <string>

#include <nlohmann/json.hpp>
#include <std_msgs/msg/string.hpp>
#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace h1_2_mujoco_plugins
{

class ObjectSyncPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override
  {
    node_ = node;

    subscription_ = node_->create_subscription<std_msgs::msg::String>(
      "/h1_demo/object_manipulation_state",
      10,
      std::bind(&ObjectSyncPlugin::topic_callback, this, std::placeholders::_1));

    const int body_id = mj_name2id(model, mjOBJ_BODY, "visual_objeto_cubo");
    const int geom_id = mj_name2id(model, mjOBJ_GEOM, "visual_objeto_cubo_geom");

    int geom_bodyid = -1;
    if (geom_id != -1) {
      geom_bodyid = model->geom_bodyid[geom_id];
    }

    RCLCPP_WARN(
      node_->get_logger(),
      "[ObjectSyncPlugin INIT] body_id=%d geom_id=%d geom_bodyid=%d | subscribed=/h1_demo/object_manipulation_state",
      body_id, geom_id, geom_bodyid);

    return true;
  }

  void update(const mjModel* model, mjData* data) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_new_state_) {
      return;
    }

    update_count_++;

    const int body_id = mj_name2id(model, mjOBJ_BODY, "visual_objeto_cubo");
    const int geom_id = mj_name2id(model, mjOBJ_GEOM, "visual_objeto_cubo_geom");

    if (body_id == -1) {
      if (update_count_ % 100 == 0) {
        RCLCPP_ERROR(node_->get_logger(), "[ObjectSyncPlugin] visual_objeto_cubo no existe en mjModel.");
      }
      return;
    }

    // Lectura antes de modificar el modelo.
    double geom_before_x = -999.0;
    double geom_before_y = -999.0;
    double geom_before_z = -999.0;

    if (geom_id != -1) {
      geom_before_x = data->geom_xpos[3 * geom_id + 0];
      geom_before_y = data->geom_xpos[3 * geom_id + 1];
      geom_before_z = data->geom_xpos[3 * geom_id + 2];
    }

    // Este plugin es visual-only: se modifica la pose base del body en mjModel.
    // Se usa const_cast porque la interfaz del plugin entrega const mjModel*,
    // pero para esta etapa el cubo no es físico ni dinámico.
    mjModel* mutable_model = const_cast<mjModel*>(model);

    mutable_model->body_pos[3 * body_id + 0] = x_;
    mutable_model->body_pos[3 * body_id + 1] = y_;
    mutable_model->body_pos[3 * body_id + 2] = z_;

    mutable_model->body_quat[4 * body_id + 0] = qw_;
    mutable_model->body_quat[4 * body_id + 1] = qx_;
    mutable_model->body_quat[4 * body_id + 2] = qy_;
    mutable_model->body_quat[4 * body_id + 3] = qz_;

    // Recalcular cinemática desde el modelo persistente.
    mj_forward(model, data);

    double body_x = data->xpos[3 * body_id + 0];
    double body_y = data->xpos[3 * body_id + 1];
    double body_z = data->xpos[3 * body_id + 2];

    double geom_after_x = -999.0;
    double geom_after_y = -999.0;
    double geom_after_z = -999.0;

    int geom_bodyid = -1;
    if (geom_id != -1) {
      geom_bodyid = model->geom_bodyid[geom_id];
      geom_after_x = data->geom_xpos[3 * geom_id + 0];
      geom_after_y = data->geom_xpos[3 * geom_id + 1];
      geom_after_z = data->geom_xpos[3 * geom_id + 2];
    }

    if (update_count_ % 100 == 0) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[ObjectSyncPlugin MODEL BODY] mode=%s msg_count=%d update_count=%d | "
        "target=[%.4f %.4f %.4f] | "
        "model_body_pos=[%.4f %.4f %.4f] | "
        "body_xpos=[%.4f %.4f %.4f] | "
        "geom_before=[%.4f %.4f %.4f] | "
        "geom_after=[%.4f %.4f %.4f] | "
        "body_id=%d geom_id=%d geom_bodyid=%d",
        mode_.c_str(), msg_count_, update_count_,
        x_, y_, z_,
        mutable_model->body_pos[3 * body_id + 0],
        mutable_model->body_pos[3 * body_id + 1],
        mutable_model->body_pos[3 * body_id + 2],
        body_x, body_y, body_z,
        geom_before_x, geom_before_y, geom_before_z,
        geom_after_x, geom_after_y, geom_after_z,
        body_id, geom_id, geom_bodyid);
    }
  }

  void cleanup() override
  {
    subscription_.reset();
  }

private:
  void topic_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    try {
      auto json_data = nlohmann::json::parse(msg->data);

      std::lock_guard<std::mutex> lock(mutex_);

      if (json_data.contains("mode")) {
        mode_ = json_data["mode"].get<std::string>();
      }

      if (json_data.contains("position") && json_data.contains("orientation")) {
        x_ = json_data["position"][0].get<double>();
        y_ = json_data["position"][1].get<double>();
        z_ = json_data["position"][2].get<double>();

        qw_ = json_data["orientation"][0].get<double>();
        qx_ = json_data["orientation"][1].get<double>();
        qy_ = json_data["orientation"][2].get<double>();
        qz_ = json_data["orientation"][3].get<double>();

        has_new_state_ = true;
        msg_count_++;

        if (msg_count_ <= 5 || msg_count_ % 50 == 0) {
          RCLCPP_WARN(
            node_->get_logger(),
            "[ObjectSyncPlugin CALLBACK] msg_count=%d mode=%s | pos=[%.4f %.4f %.4f] quat=[%.4f %.4f %.4f %.4f]",
            msg_count_, mode_.c_str(), x_, y_, z_, qw_, qx_, qy_, qz_);
        }
      } else {
        RCLCPP_ERROR(
          node_->get_logger(),
          "[ObjectSyncPlugin CALLBACK] JSON sin campos position/orientation: %s",
          msg->data.c_str());
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "Error parsing ObjectSync msg: %s", e.what());
    }
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;

  std::mutex mutex_;
  std::string mode_ = "WORLD";

  double x_ = 0.54;
  double y_ = 0.30;
  double z_ = 0.08;

  double qw_ = 1.0;
  double qx_ = 0.0;
  double qy_ = 0.0;
  double qz_ = 0.0;

  bool has_new_state_ = false;
  int msg_count_ = 0;
  int update_count_ = 0;
};

}  // namespace h1_2_mujoco_plugins

PLUGINLIB_EXPORT_CLASS(h1_2_mujoco_plugins::ObjectSyncPlugin, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
