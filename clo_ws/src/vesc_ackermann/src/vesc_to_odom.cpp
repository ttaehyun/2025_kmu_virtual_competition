#include "vesc_ackermann/vesc_to_odom.h"
#include <cmath>
#include <limits>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace vesc_ackermann
{

template <typename T>
inline bool getRequiredParam(const ros::NodeHandle& nh, const std::string& name, T* value)
{
  if (nh.getParam(name, *value)) return true;
  ROS_FATAL("VescToOdom: Parameter %s is required.", name.c_str());
  return false;
}

VescToOdom::VescToOdom(ros::NodeHandle nh, ros::NodeHandle private_nh)
  : odom_frame_("odom"), base_frame_("base_link"),
    use_servo_cmd_(true), publish_tf_(false),
    x_(0.0), y_(0.0), imu_yaw_(0.0), imu_received_(false),
    wheelbase_(0.0), center_of_mass_offset_(0.0)
{
  // load frames and flags
  private_nh.param("odom_frame", odom_frame_, odom_frame_);
  private_nh.param("base_frame", base_frame_, base_frame_);
  private_nh.param("use_servo_cmd_to_calc_angular_velocity", use_servo_cmd_, use_servo_cmd_);
  private_nh.param("publish_tf", publish_tf_, publish_tf_);

  // load vehicle geometry
  if (!getRequiredParam(nh, "wheelbase", &wheelbase_)) return;
  private_nh.param("center_of_mass_offset", center_of_mass_offset_, center_of_mass_offset_);

  // load conversion parameters
  if (!getRequiredParam(nh, "speed_to_erpm_gain", &speed_to_erpm_gain_)) return;
  if (!getRequiredParam(nh, "speed_to_erpm_offset", &speed_to_erpm_offset_)) return;
  if (use_servo_cmd_) {
    if (!getRequiredParam(nh, "steering_angle_to_servo_gain", &steering_to_servo_gain_)) return;
    if (!getRequiredParam(nh, "steering_angle_to_servo_offset", &steering_to_servo_offset_)) return;
  }

  // initialize publishers and subscribers
  odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 10);
  if (publish_tf_) tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());

  vesc_state_sub_ = nh.subscribe("sensors/core", 10, &VescToOdom::vescStateCallback, this);
  if (use_servo_cmd_)
    servo_sub_ = nh.subscribe("sensors/servo_position_command", 10,
                              &VescToOdom::servoCmdCallback, this);
  imu_sub_ = nh.subscribe("imu", 50, &VescToOdom::imuCallback, this);
}

inline double VescToOdom::computeTurnRadius(double steering) const
{
  if (steering == 0.0) return std::numeric_limits<double>::infinity();
  double r_center = wheelbase_ / std::tan(steering);
  double r = std::sqrt(r_center*r_center + center_of_mass_offset_*center_of_mass_offset_);
  return std::copysign(r, steering);
}

void VescToOdom::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
  // extract yaw from quaternion
  tf2::Quaternion q;
  tf2::fromMsg(imu_msg->orientation, q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  imu_yaw_ = yaw;
  imu_received_ = true;
}

void VescToOdom::vescStateCallback(const vesc_msgs::VescStateStamped::ConstPtr& state)
{
  if (!imu_received_) return;  // wait for first IMU
  if (use_servo_cmd_ && !last_servo_cmd_) return;
  if (!last_state_) { last_state_ = state; return; }

  double dt = (state->header.stamp - last_state_->header.stamp).toSec();
  if (dt <= 0.0) return;
  dt = std::min(dt, 0.1);

  // compute linear speed [m/s]
  double raw_speed = state->state.speed;
  double linear_speed = (raw_speed - speed_to_erpm_offset_) / speed_to_erpm_gain_;
  if (std::fabs(linear_speed) < 1e-3) linear_speed = 0.0;

  // compute steering angle [rad]
  double steering = 0.0;
  if (use_servo_cmd_) {
    steering = (last_servo_cmd_->data - steering_to_servo_offset_) / steering_to_servo_gain_;
  }

  // compute turn radius & angular velocity
  double R = computeTurnRadius(steering);
  double wz = std::isfinite(R) ? linear_speed / R : 0.0;

  // arc integration for position using imu_yaw_
  double yaw0 = imu_yaw_ - wz*dt; // approximate previous yaw
  double yaw1 = imu_yaw_;         // current yaw from IMU
  double dx_local = std::isfinite(R)
    ? R * (std::sin(wz*dt))
    : linear_speed * dt;
  double dy_local = std::isfinite(R)
    ? R * (1.0 - std::cos(wz*dt))
    : 0.0;

  // world frame integration using imu_yaw_
  double c = std::cos(yaw0);
  double s = std::sin(yaw0);
  x_ += c*dx_local - s*dy_local;
  y_ += s*dx_local + c*dy_local;

  // prepare quaternion from IMU
  geometry_msgs::Quaternion odom_quat = tf2::toMsg(tf2::Quaternion(0,0,0,1));
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, imu_yaw_);
  odom_quat = tf2::toMsg(q);

  // publish TF
  if (publish_tf_) {
    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = state->header.stamp;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = x_;
    tf_msg.transform.translation.y = y_;
    tf_msg.transform.rotation = odom_quat;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  // publish Odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = state->header.stamp;
  odom_msg.header.frame_id = odom_frame_;
  odom_msg.child_frame_id = base_frame_;
  odom_msg.pose.pose.position.x = x_;
  odom_msg.pose.pose.position.y = y_;
  odom_msg.pose.pose.orientation = odom_quat;
  odom_msg.twist.twist.linear.x = linear_speed;
  odom_msg.twist.twist.angular.z = wz;

  // covariance
  odom_msg.pose.covariance[0] = 1e-4;
  odom_msg.pose.covariance[7] = 1e-4;
  odom_msg.pose.covariance[35] = 1e-4;
  odom_msg.twist.covariance[0] = 1e-4;
  odom_msg.twist.covariance[7] = 1e-4;
  odom_msg.twist.covariance[35] = 1e-4;

  odom_pub_.publish(odom_msg);
  last_state_ = state;
}

void VescToOdom::servoCmdCallback(const std_msgs::Float64::ConstPtr& servo)
{
  last_servo_cmd_ = servo;
}

}  // namespace vesc_ackermann
