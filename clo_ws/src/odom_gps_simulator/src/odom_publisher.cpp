#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>

inline double deg2rad(double deg) {
  return deg * M_PI / 180.0;
}

class OdomPublisher
{
public:
  OdomPublisher()
  {
    imu_received_ = false;
    ego_received_ = false;

    // 초기 오프셋: map 기준 초기 자세
    init_pos_.setX(0.0); // -19.0
    init_pos_.setY(0.0); // 4.5
    init_pos_.setZ(0.0); // 0.03

    // Euler 각도(deg) → rad → quaternion
    double roll  = deg2rad(0.0); // -359.97549438476563
    double pitch = deg2rad(0.0); // 0.61049038171768188
    double yaw   = deg2rad(0.0); // 0.0003662109375
    init_q_.setRPY(roll, pitch, yaw);

    imu_sub_  = nh_.subscribe("/imu",       10, &OdomPublisher::imuCallback,  this);
    ego_sub_  = nh_.subscribe("/Ego_topic", 10, &OdomPublisher::egoCallback,  this);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/ground_truth", 10);

    // 50Hz 타이머
    timer_ = nh_.createTimer(
      ros::Duration(0.02),
      &OdomPublisher::timerCallback,
      this
    );
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber    imu_sub_, ego_sub_;
  ros::Publisher     odom_pub_;
  ros::Timer         timer_;

  sensor_msgs::Imu             latest_imu_;
  morai_msgs::EgoVehicleStatus latest_ego_;
  bool imu_received_, ego_received_;

  tf2::Vector3    init_pos_;
  tf2::Quaternion init_q_;

  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    latest_imu_ = *msg;
    imu_received_ = true;
  }

  void egoCallback(const morai_msgs::EgoVehicleStatus::ConstPtr& msg) {
    latest_ego_ = *msg;
    ego_received_ = true;
  }

  void timerCallback(const ros::TimerEvent&) {
    if (!imu_received_ || !ego_received_) return;

    // 1) 현재 map 좌표에서 로봇 위치 추출
    tf2::Vector3 curr_pos(
      latest_ego_.position.x,
      latest_ego_.position.y,
      latest_ego_.position.z
    );

    // 2) 초기 자세 기준으로 상대 위치 계산 (quatRotate 이용)
    tf2::Vector3 diff = curr_pos - init_pos_;
    tf2::Vector3 rel_pos = tf2::quatRotate(init_q_.inverse(), diff);

    // 3) IMU orientation → quaternion
    tf2::Quaternion imu_q;
    tf2::fromMsg(latest_imu_.orientation, imu_q);

    // 4) 초기 orientation 기준으로 상대 회전 계산
    tf2::Quaternion rel_q = init_q_.inverse() * imu_q;
    rel_q.normalize();

    // odometry 메시지 작성
    nav_msgs::Odometry odom;
    odom.header.stamp    = ros::Time::now();
    odom.header.frame_id = "odom";
    odom.child_frame_id  = "base_link";

    // pose
    odom.pose.pose.position.x = rel_pos.x();
    odom.pose.pose.position.y = rel_pos.y();
    odom.pose.pose.position.z = rel_pos.z();
    odom.pose.pose.orientation = tf2::toMsg(rel_q);

    // 매우 낮은 공분산 (simulation)
    for (int i = 0; i < 36; ++i) {
      odom.pose.covariance[i] = 1e-6;
    }

    // twist
    odom.twist.twist.linear.x  = latest_ego_.velocity.x;
    odom.twist.twist.linear.y  = latest_ego_.velocity.y;
    odom.twist.twist.linear.z  = latest_ego_.velocity.z;
    odom.twist.twist.angular   = latest_imu_.angular_velocity;
    for (int i = 0; i < 36; ++i) {
      odom.twist.covariance[i] = 1e-6;
    }

    odom_pub_.publish(odom);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "odom_publisher");
  OdomPublisher node;
  ros::spin();
  return 0;
}
