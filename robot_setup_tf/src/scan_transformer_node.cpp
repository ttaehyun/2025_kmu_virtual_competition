#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <limits>  // for INFINITY

class ScanTransformer
{
public:
  ScanTransformer()
  : tf_buffer_(), tf_listener_(tf_buffer_)
  {
    scan_sub_ = nh_.subscribe("/lidar2D", 10, &ScanTransformer::scanCallback, this);
    cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/lidar2D/pointcloud", 10);
  }

  void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in)
  {
    // 1. LaserScan 복사해서 수정
    sensor_msgs::LaserScan scan = *scan_in;
    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
      if (scan.ranges[i] == 10.0)
        scan.ranges[i] = std::numeric_limits<float>::infinity();
    }

    // 2. LaserScan → PointCloud2
    sensor_msgs::PointCloud2 cloud;
    try {
      projector_.projectLaser(scan, cloud);
    } catch (std::exception& e) {
      ROS_WARN("Laser projection failed: %s", e.what());
      return;
    }

    // 3. tf 변환 받기: base_lidar → base_link
    geometry_msgs::TransformStamped transformStamped;
    try {
      transformStamped = tf_buffer_.lookupTransform(
        "base_link", scan.header.frame_id,
        scan.header.stamp, ros::Duration(0.1));
    } catch (tf2::TransformException& ex) {
      ROS_WARN("Transform error: %s", ex.what());
      return;
    }

    // 4. 좌표계 변환
    sensor_msgs::PointCloud2 transformed_cloud;
    try {
      tf2::doTransform(cloud, transformed_cloud, transformStamped);
    } catch (std::exception& e) {
      ROS_WARN("PointCloud2 transform failed: %s", e.what());
      return;
    }

    // 5. 발행
    transformed_cloud.header.stamp = scan.header.stamp;
    transformed_cloud.header.frame_id = "base_lidar";
    cloud_pub_.publish(transformed_cloud);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Publisher cloud_pub_;
  laser_geometry::LaserProjection projector_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scan_transformer_node");
  ScanTransformer st;
  ros::spin();
  return 0;
}
