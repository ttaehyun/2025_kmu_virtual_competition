#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // tf2::doTransform()을 위해 필요

// transformPoint 함수 정의
void transformPoint(tf2_ros::Buffer& tfBuffer) {
  geometry_msgs::PointStamped laser_point;
  laser_point.header.frame_id = "base_laser";
  laser_point.header.stamp = ros::Time::now(); // 가장 최근의 transform 사용
  laser_point.point.x = 1.0;
  laser_point.point.y = 0.2;
  laser_point.point.z = 0.0;

  try {
    // base_laser → base_link 변환 정보 가져오기
    geometry_msgs::TransformStamped transformStamped;
    transformStamped = tfBuffer.lookupTransform("base_link", "base_laser", ros::Time::now(), ros::Duration(1.0));

    // 실제 변환 수행
    geometry_msgs::PointStamped base_point;
    tf2::doTransform(laser_point, base_point, transformStamped);

    ROS_INFO("base_laser: (%.2f, %.2f. %.2f) -----> base_link: (%.2f, %.2f, %.2f) at time %.2f",
        laser_point.point.x, laser_point.point.y, laser_point.point.z,
        base_point.point.x, base_point.point.y, base_point.point.z, base_point.header.stamp.toSec());
  }
  catch (tf2::TransformException &ex) {
    ROS_WARN("Received an exception trying to transform a point from \"base_laser\" to \"base_link\": %s", ex.what());
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "robot_tf2_listener");
  ros::NodeHandle nh;

  // tf2 버퍼 및 리스너 설정
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  // 1초마다 transformPoint 호출
  ros::Timer timer = nh.createTimer(ros::Duration(1.0),
      boost::bind(&transformPoint, boost::ref(tfBuffer)));

  ros::spin();
  return 0;
}
