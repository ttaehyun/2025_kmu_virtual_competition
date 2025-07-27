#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_undistort_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 288.120916, 0, 324.192910,
                                                       0, 288.335460, 251.346265,
                                                       0, 0, 1);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << -0.287588, 0.047197, -0.000458, -0.001711, 0.000000);

    cv::Size image_size(640, 480);
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), camera_matrix, image_size, CV_16SC2, map1, map2);

    image_transport::Publisher pub = it.advertise("/usb_cam/image_raw/calib", 1);

    image_transport::Subscriber sub = it.subscribe("/usb_cam/image_raw", 1, [&map1, &map2, &pub](const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

            cv::Mat undistorted_image;
            cv::remap(frame, undistorted_image, map1, map2, cv::INTER_LINEAR);

            std_msgs::Header header = msg->header;
            header.seq = msg->header.seq; 

            sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(header, "bgr8", undistorted_image).toImageMsg();
            pub.publish(output_msg);

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    });

    ros::spin();
    return 0;
}