#include <ros/ros.h>
#include <opencv4/opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

class RedRoadNode {
public:
    RedRoadNode() : Red_Detect(false) {
        // Publisher 생성
        pub = nh_.advertise<ackermann_msgs::AckermannDriveStamped>("high_level/ackermann_cmd_mux/input/nav_2", 1);
        // Subscriber 생성
        sub_ = nh_.subscribe("/usb_cam/image_raw/calib", 10, &RedRoadNode::imageCallback, this);
        sub_lane = nh_.subscribe("high_level/ackermann_cmd_mux/input/nav_6", 10, &RedRoadNode::Callback, this);
        drive_info.drive.speed = 0.3;
        last_detect_time = ros::Time::now();
        // OpenCV 창 생성
        //cv::namedWindow("red_line_percept");
    }

    ~RedRoadNode() {
        cv::destroyAllWindows();
    }

    void Callback(const ackermann_msgs::AckermannDriveStamped& msg) {
        drive_info.drive.steering_angle = msg.drive.steering_angle;
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // ROS 이미지를 OpenCV 이미지로 변환
            cv::Mat frame_raw = cv_bridge::toCvShare(msg, "bgr8")->image;

            // ROI 설정 (이미지의 특정 영역을 자름)
            //cv::Rect roi(frame_raw.cols * 3 / 8, frame_raw.rows * 3 / 4, frame_raw.cols / 4, frame_raw.rows / 4);
            cv::Rect roi(frame_raw.cols * 1 / 4, frame_raw.rows * 3 / 4, frame_raw.cols / 2, frame_raw.rows / 4);
            cv::Mat frame_roi = frame_raw(roi);

            // 가우시안 필터 적용
            cv::GaussianBlur(frame_roi, frame_roi, cv::Size(5, 5), 0);

            // HSV 설정 및 BGR에서 HSV로 전환
            cv::Mat frame_hsv;
            cv::cvtColor(frame_roi, frame_hsv, cv::COLOR_BGR2HSV);

            // red 색깔 구분 mask
            cv::Mat mask1, mask2, mask;
            cv::inRange(frame_hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask1);
            cv::inRange(frame_hsv, cv::Scalar(170, 100, 0), cv::Scalar(180, 255, 255), mask2);

            cv::bitwise_or(mask1, mask2, mask);

            // 원본 이미지에 ROI 그리기
            cv::rectangle(frame_raw, roi, cv::Scalar(255, 0, 0), 2); // 파란색 박스로 ROI 표시

            // 컨투어 찾기
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            bool detected = false;

            for (size_t i = 0; i < contours.size(); i++) {
                if (cv::contourArea(contours[i]) > 1500) {
                    detected = true;

                    std::vector<cv::Point> adjusted_contour;
                    for (const auto& pt : contours[i]) {
                        adjusted_contour.emplace_back(pt.x + roi.x, pt.y + roi.y);
                    }
                    ROS_INFO("Red Road detected!");
                    cv::putText(frame_raw, "Red Road Detected", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    drive_info.drive.speed = 0.2; // 빨간색 라인이 발견되었을 때 속도
                    cv::drawContours(frame_raw, std::vector<std::vector<cv::Point>>{adjusted_contour}, -1, cv::Scalar(0, 255, 0), 3); // 초록색으로 그림
                    break;
                }
            }

            if (detected) {
                Red_Detect = true;
                last_detect_time = ros::Time::now(); // 감지 시간 업데이트
            } else {
                Red_Detect = false;
                if ((ros::Time::now() - last_detect_time).toSec() > 1.0) {
                    //ROS_INFO("No Red Road detected for 1 seconds!");
                    //drive_info.drive.speed = 0.0; // 2초 동안 감지 안된 경우 정지
                }
                cv::putText(frame_raw, "No Red Road Detected", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            }

            // OpenCV 창에 프레임 표시
            //cv::imshow("red_line_percept", frame_raw);
            // 'q' 키를 누르면 종료
            if (cv::waitKey(1) == 'q') {
                ros::shutdown();
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge 예외: %s", e.what());
        }
    }

    ackermann_msgs::AckermannDriveStamped getAckermannDriveMsg() const {
        return drive_info;
    }

    bool getRedDetect() {
        return Red_Detect;
    }

    ros::Publisher pub;
    ros::Time last_detect_time; // 마지막 감지 시간을 저장

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Subscriber sub_lane;
    ackermann_msgs::AckermannDriveStamped drive_info;

    bool Red_Detect;
};

int main(int argc, char** argv) {
    // ROS 노드 초기화
    ros::init(argc, argv, "red_line_perception");

    RedRoadNode node;

    // 루프 주기 설정 (예: 100Hz)
    ros::Rate loop_rate(30);

    while (ros::ok()) {
        if (node.getRedDetect() || (ros::Time::now() - node.last_detect_time).toSec() <= 0.5) {
            // 모터 속도 발행
            node.pub.publish(node.getAckermannDriveMsg());
        }
        // ROS 스핀 및 주기 유지
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}