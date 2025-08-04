#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

class OdomTfBroadcaster
{
public:
  OdomTfBroadcaster()
  {
    // /odom 토픽 구독
    odom_sub_ = nh_.subscribe("/odom", 10, &OdomTfBroadcaster::odomCallback, this);
  }

private:
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;           // odom 메시지 시간 사용
    tf_msg.header.frame_id = "odom";                   // 부모 프레임
    tf_msg.child_frame_id = "base_link";               // 자식 프레임

    // 위치 그대로 복사
    tf_msg.transform.translation.x = msg->pose.pose.position.x;
    tf_msg.transform.translation.y = msg->pose.pose.position.y;
    tf_msg.transform.translation.z = msg->pose.pose.position.z;

    // 회전 그대로 복사
    tf_msg.transform.rotation = msg->pose.pose.orientation;

    // 브로드캐스트
    br_.sendTransform(tf_msg);
  }

  ros::NodeHandle nh_;
  ros::Subscriber odom_sub_;
  tf2_ros::TransformBroadcaster br_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tf_broadcaster");
  OdomTfBroadcaster broadcaster;
  ros::spin();
  return 0;
}
