#ifndef VESC_ACKERMANN_VESC_TO_ODOM_H_
#define VESC_ACKERMANN_VESC_TO_ODOM_H_

#include <memory>
#include <string>
#include <limits>

#include <ros/ros.h>
#include <vesc_msgs/VescStateStamped.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_broadcaster.h>

namespace vesc_ackermann
{

class VescToOdom
{
public:
  VescToOdom(ros::NodeHandle nh, ros::NodeHandle private_nh);

private:
  // frames
  std::string odom_frame_;
  std::string base_frame_;
  bool use_servo_cmd_;
  bool publish_tf_;

  // conversion params
  double speed_to_erpm_gain_;
  double speed_to_erpm_offset_;
  double steering_to_servo_gain_;
  double steering_to_servo_offset_;

  // vehicle geometry
  double wheelbase_;
  double center_of_mass_offset_;

  // state
  double x_, y_;
  double imu_yaw_;             ///< fused orientation from IMU in radians
  bool imu_received_;

  // previous messages
  std_msgs::Float64::ConstPtr last_servo_cmd_;
  vesc_msgs::VescStateStamped::ConstPtr last_state_;

  // ROS interfaces
  ros::Publisher odom_pub_;
  ros::Subscriber vesc_state_sub_;
  ros::Subscriber servo_sub_;
  ros::Subscriber imu_sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // callbacks
  void vescStateCallback(const vesc_msgs::VescStateStamped::ConstPtr& state);
  void servoCmdCallback(const std_msgs::Float64::ConstPtr& servo);
  void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);

  // helpers
  inline double computeTurnRadius(double steering) const;
};

}  // namespace vesc_ackermann

#endif  // VESC_ACKERMANN_VESC_TO_ODOM_H_
