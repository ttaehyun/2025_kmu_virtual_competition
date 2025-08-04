#include <ros/ros.h>
#include <cmath>
#include <algorithm>  // std::min, std::max
#include <geometry_msgs/Twist.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

double wheelbase;
std::string frame_id;
ros::Publisher pub;

// 차량 한계치 (m/s, rad)
constexpr double MAX_SPEED_MS     =  8.0 / 3.6;            // 8 km/h → m/s
constexpr double MIN_SPEED_MS     = -8.0 / 3.6;
constexpr double MAX_STEERING_RAD =  19.5 * M_PI / 180.0;  // 19.5° → rad
constexpr double MIN_STEERING_RAD = -19.5 * M_PI / 180.0;

double convertTransRotVelToSteering(double v, double omega, double wheelbase) {
  if (omega == 0.0 || v == 0.0) {
    return 0.0;
  }
  return std::atan2(wheelbase * omega, v);
}

void cmdCallback(const geometry_msgs::Twist::ConstPtr& data) {
  // 1) 원 입력
  double v_raw     = data->linear.x;
  double omega_raw = data->angular.z;

  // 2) 속도 클램핑 (min/max)
  double v = std::max(std::min(v_raw, MAX_SPEED_MS), MIN_SPEED_MS);

  // // 최소 임계 속도 보정
  // const double MIN_EFFECTIVE_SPEED = 0.2;
  // if (std::fabs(v) > 0.0 && std::fabs(v) < MIN_EFFECTIVE_SPEED) {
  //   v = (v > 0.0) ? MIN_EFFECTIVE_SPEED : -MIN_EFFECTIVE_SPEED;
  // }

  // // 각속도만 들어온 경우, 후진 강제
  // if (std::fabs(v_raw) < 1e-3 && std::fabs(omega_raw) > 0.1) {
  //   v = -MIN_EFFECTIVE_SPEED;
  // }

  // 3) Ackermann 조향각 계산
  double steering = convertTransRotVelToSteering(v, omega_raw, wheelbase);

  // 4) 조향각 클램핑
  steering = std::max(std::min(steering, MAX_STEERING_RAD), MIN_STEERING_RAD);

  // 5) 전진/후진에 따라 부호 결정
  double applied_steering = (v >= 0.0) ? -steering : steering;

  // 6) 메시지 생성 및 발행
  ackermann_msgs::AckermannDriveStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_id;
  msg.drive.speed          = v;
  msg.drive.steering_angle = applied_steering;
  pub.publish(msg);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "cmd_vel_to_ackermann_drive");
  ros::NodeHandle nh("~");

  std::string twist_cmd_topic, ackermann_cmd_topic;
  nh.param("twist_cmd_topic",    twist_cmd_topic,    std::string("/cmd_vel"));
  nh.param("ackermann_cmd_topic",ackermann_cmd_topic,std::string("/ackermann_cmd"));
  nh.param("wheelbase",          wheelbase,          1.0);
  nh.param("frame_id",           frame_id,           std::string("odom"));

  pub = nh.advertise<ackermann_msgs::AckermannDriveStamped>(ackermann_cmd_topic, 1);
  ros::Subscriber sub = nh.subscribe(twist_cmd_topic, 1, cmdCallback);

  ROS_INFO("cmd_vel_to_ackermann_drive started. "
           "speed ∈ [%.2f, %.2f] m/s, steering ∈ [%.2f, %.2f] rad",
           MIN_SPEED_MS, MAX_SPEED_MS,
           MIN_STEERING_RAD, MAX_STEERING_RAD);

  ros::spin();
  return 0;
}
