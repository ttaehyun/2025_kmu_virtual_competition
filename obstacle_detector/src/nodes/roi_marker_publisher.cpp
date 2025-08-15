#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "roi_marker_publisher");
    ros::NodeHandle nh("~");

    // ROI 파라미터 - 원하는 값으로 수정하거나, rosparam에서 받거나, launch 통합 가능
    double min_x, max_x, min_y, max_y;
    nh.param("min_x_limit", min_x, -4.0);
    nh.param("max_x_limit", max_x, 2.0);
    nh.param("min_y_limit", min_y, -2.0);
    nh.param("max_y_limit", max_y, 2.0);

    std::string frame_id;
    nh.param<std::string>("frame_id", frame_id, "lidar");

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("roi_marker", 1);

    ros::Rate r(2); // 2Hz(주기적 publish)

    while (ros::ok()) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "lidar"; // 프레임 ID 설정
        marker.header.stamp = ros::Time::now();
        marker.ns = "roi_boundary";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;       // 선 두께(m)
        marker.color.r = 1.0;        // 빨간색
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        marker.pose.orientation.w = 1.0;

        geometry_msgs::Point p1, p2, p3, p4;  

        p1.x = min_x; p1.y = min_y; p1.z = 0;
        p2.x = max_x; p2.y = min_y; p2.z = 0;
        p3.x = max_x; p3.y = max_y; p3.z = 0;
        p4.x = min_x; p4.y = max_y; p4.z = 0;

        marker.points.push_back(p1);
        marker.points.push_back(p2);
        marker.points.push_back(p3);
        marker.points.push_back(p4);
        marker.points.push_back(p1); // 사각형 닫기

        marker.lifetime = ros::Duration(0); // 영구 표시

        marker_pub.publish(marker);
        ros::spinOnce();
        r.sleep();
    }
    return 0;
}
