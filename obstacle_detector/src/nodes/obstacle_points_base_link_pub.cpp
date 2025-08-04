#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <obstacle_detector/Obstacles.h>
#include <tf/transform_listener.h>

ros::Publisher pc_pub;

void obstaclesCallback(const obstacle_detector::Obstacles::ConstPtr& msg, tf::TransformListener* tf_listener) {
    sensor_msgs::PointCloud pc;
    pc.header.stamp = msg->header.stamp;
    pc.header.frame_id = "base_link"; // 변환 후 프레임으로 명시

    for (const auto& circ : msg->circles) {
        geometry_msgs::PointStamped pt_in, pt_out;
        pt_in.header = msg->header;
        pt_in.header.frame_id = "lidar"; // 클러스터 좌표의 원본 프레임
        pt_in.point.x = circ.center.x;
        pt_in.point.y = circ.center.y;
        pt_in.point.z = 0.0;

        try {
            // "base_link" 기준으로 변환
            tf_listener->transformPoint("base_link", pt_in, pt_out);

            geometry_msgs::Point32 pt;
            pt.x = pt_out.point.x;
            pt.y = pt_out.point.y;
            pt.z = pt_out.point.z;
            pc.points.push_back(pt);
        }
        catch (tf::TransformException &ex) {
            ROS_WARN("%s", ex.what());
            continue;
        }
    }
    pc_pub.publish(pc);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "obstacle_points_base_link_pub");
    ros::NodeHandle nh;
    tf::TransformListener tf_listener;

    pc_pub = nh.advertise<sensor_msgs::PointCloud>("detected_obstacle_points_base_link", 10);
    ros::Subscriber sub = nh.subscribe<obstacle_detector::Obstacles>(
        "/raw_obstacles", 10, boost::bind(obstaclesCallback, _1, &tf_listener)
    );

    ros::spin();
    return 0;
}
