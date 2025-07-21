#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

constexpr double PI = 3.14159265358979323846;

void publishStaticTF(const std::string& parent, const std::string& child,
                     double x, double y, double z,
                     double roll, double pitch, double yaw)
{
  static tf2_ros::StaticTransformBroadcaster static_broadcaster;
  geometry_msgs::TransformStamped static_transformStamped;

  static_transformStamped.header.stamp = ros::Time::now();
  static_transformStamped.header.frame_id = parent;
  static_transformStamped.child_frame_id = child;
  static_transformStamped.transform.translation.x = x;
  static_transformStamped.transform.translation.y = y;
  static_transformStamped.transform.translation.z = z;

  tf2::Quaternion quat;
  quat.setRPY(roll, pitch, yaw);
  static_transformStamped.transform.rotation.x = quat.x();
  static_transformStamped.transform.rotation.y = quat.y();
  static_transformStamped.transform.rotation.z = quat.z();
  static_transformStamped.transform.rotation.w = quat.w();

  static_broadcaster.sendTransform(static_transformStamped);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "static_tf2_broadcaster");
  ros::NodeHandle nh;

  // 센서들에 대한 고정 변환 등록
  publishStaticTF("base_link", "lidar", 0.11, 0.0, 0.13, 0, 0, PI);
  publishStaticTF("base_link", "camera", 0.3, 0.0, 0.11, -PI/2, 0, -PI/2);
  publishStaticTF("base_link", "imu", 0.0, 0.0, 0.0, 0, 0, 0);

  ros::spin();  // 노드 유지
  return 0;
}
