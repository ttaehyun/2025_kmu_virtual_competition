#include <ros/ros.h>
#include <cmath>
#include <algorithm>  // std::min, std::max
#include <geometry_msgs/Twist.h>

ros::Publisher pub;

// 차량 한계치 (m/s, rad)
constexpr double MAX_SPEED_MS     =  8.0 / 3.6;            // 8 km/h → m/s
constexpr double MIN_SPEED_MS     = -8.0 / 3.6;

void cmdCallback(const geometry_msgs::Twist::ConstPtr& data) {
  double linear_x     = data->linear.x;
  double angular_z = data->angular.z;

  double v = std::max(std::min(linear_x, MAX_SPEED_MS), MIN_SPEED_MS);

  double applied_angular_z = (v >= 0.0) ? -angular_z : angular_z;

  geometry_msgs::Twist msg;
  msg.linear.x          = v;
  msg.angular.z = applied_angular_z;
  pub.publish(msg);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "cmd_vel_to_commands_vel");
  ros::NodeHandle nh("~");

  std::string twist_cmd_topic, commands_vel_topic;
  nh.param("twist_cmd_topic",    twist_cmd_topic,    std::string("/cmd_vel"));
  nh.param("commands_vel_topic", commands_vel_topic,std::string("/commands/vel"));

  pub = nh.advertise<geometry_msgs::Twist>(commands_vel_topic, 1);
  ros::Subscriber sub = nh.subscribe(twist_cmd_topic, 1, cmdCallback);
  ros::spin();
  return 0;
}
