#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Int32.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <vector>
#include <algorithm>

class ClusterDirectionVisualizer {
private:
    ros::NodeHandle nh_;
    ros::Subscriber center_sub_;    // 중심점 토픽 구독
    ros::Publisher arrow_pub_;      // 방향 화살표 퍼블리셔
    ros::Publisher flag_pub_;       // 플래그 퍼블리셔
    
    geometry_msgs::Point prev_center_; // 이전 중심점
    ros::Time prev_flag_time;
    ros::Time prev_time_detect;
    ros::Time prev_time_decision;
 
    geometry_msgs::Point origin;
    geometry_msgs::Point left_arr;
    geometry_msgs::Point right_arr;
    std_msgs::Int32 flag;

    bool has_last_center_ = false;    // 이전 중심점이 있는지 확인
    bool has_center_detect = false;
    bool b_flag = false;

    const double detection_threshold_ = 1.5; // 감지 임계 시간
    const double distance_check_interval_ = 0.2; // 거리 계산 간격
    const double detection_time_diff_flag = 1.4; //flag 발동 시간
    const double current_dis_flag = 1.1;   //flag 발동 거리 조건

    const double distance_1_ = 1.0; // 1/3 회전 교차로 지점 거리 
    const double distance_2_ = 1.55; // 2/3 회전 교차로 지점 거리
    const double max_distance_ = 3.0; // 회전교차로에서 총 라이다가 보는 거리
    const double flag_y = -0.1;     //center_maker.y 판단 기준 y 좌표
    const double delta_zero = 0.0;  //delta 0.0 기준

    std::vector<int> flag_history_; // 최근 10개의 플래그 저장
    const int flag_window_size_ = 10; // 플래그 윈도우 크기

public:
    ClusterDirectionVisualizer() {
        center_sub_ = nh_.subscribe("/simple_cluster_centers", 10, &ClusterDirectionVisualizer::centerCallback, this);
        arrow_pub_ = nh_.advertise<visualization_msgs::Marker>("/cluster_direction_arrow", 10);
        flag_pub_ = nh_.advertise<std_msgs::Int32>("/direction_flag", 10);    
        
        origin.x = 0; origin.y = 0; origin.z = 0;
        left_arr.x = -1.0; left_arr.y = -0.5; left_arr.z = 0;
        right_arr.x = -1.0; right_arr.y = 0.5; right_arr.z = 0;
    }
    
    visualization_msgs::Marker createArrowTemplate(const std::string& ns, float r, float g, float b) {
        visualization_msgs::Marker arrow_marker;
        arrow_marker.header.frame_id = "laser";
        arrow_marker.ns = ns;
        arrow_marker.id = 0;
        arrow_marker.type = visualization_msgs::Marker::ARROW;
        arrow_marker.action = visualization_msgs::Marker::ADD;

        // 화살표 크기 설정
        arrow_marker.scale.x = 0.1;  // 화살표 몸체의 길이
        arrow_marker.scale.y = 0.05;  // 화살표 머리의 두께
        arrow_marker.scale.z = 0.1;

        // 화살표 색상 설정
        arrow_marker.color.r = r;
        arrow_marker.color.g = g;
        arrow_marker.color.b = b;
        arrow_marker.color.a = 1.0;
        return arrow_marker;
    }

    void centerCallback(const visualization_msgs::Marker::ConstPtr& msg) {
        if (msg->points.empty()) {
            //ROS_WARN("Received an empty cluster center marker.");
            has_center_detect = false;
            return;
        }
        
        geometry_msgs::Point current_center = msg->points[0];
        ros::Time current_time = msg->header.stamp;

        if (has_center_detect) {
            ros::Duration time_diff_detect = current_time - prev_time_detect;
            if (time_diff_detect.toSec() >= detection_threshold_) {
                double current_dis = std::hypot(current_center.x, current_center.y - flag_y);
                //ROS_INFO("current_dis : %.2f", current_dis);
                if (has_last_center_) {
                    processDetection(current_center, current_time, prev_time_decision, current_dis);
                } else {
                    prev_center_ = current_center;                        
                    has_last_center_ = true;
                }
            } else {
                //ROS_INFO("time_diff_detect %.2f", time_diff_detect.toSec());
                prev_time_decision = current_time;
            }
        } else {
            prev_time_detect = current_time;
            has_center_detect = true;
        }
    }

    void processDetection(const geometry_msgs::Point& current_center, const ros::Time& current_time, const ros::Time& prev_time_decision, double current_dis) {
        ros::Duration time_diff_decision = current_time - prev_time_decision;

        if (time_diff_decision.toSec() >= distance_check_interval_) {
            double delta_dis = current_dis - std::hypot(prev_center_.x, prev_center_.y- flag_y);
            if (current_dis >= distance_2_ && current_dis <= max_distance_) {
                //flag.data = (current_center.y > flag_y) ? ((delta_dis > delta_zero) ? 1 : 0) : ((delta_dis > delta_zero) ? 0 : 1);
                if (current_center.y > flag_y) {
                    if (delta_dis > delta_zero) {
                        flag.data = 1;
                    }
                }
                else {
                    if (delta_dis < delta_zero) {
                        flag.data = 1;
                    }
                }
            } 
            else if (current_dis >= distance_1_) {
                //flag.data = (current_center.y > flag_y) ? ((delta_dis > delta_zero) ? 5 : 0) : ((delta_dis > delta_zero) ? 4 : 1);
                if (current_center.y > flag_y) {
                    if (delta_dis > delta_zero) {
                        flag.data = 5;
                    }
                }
                else {
                    if (delta_dis < delta_zero) {
                        flag.data = 1;
                    }
                }
            }  
            else {
                //flag.data = 3; // Stop
                if (current_center.y>-flag_y) {
                    if (delta_dis > delta_zero) {
                        //flag.data = 0;
                    }
                }
                else {
                    if(delta_dis>delta_zero) {
                        //flag.data = 1;
                    }
                }
            }
            addFlagToHistory(flag.data);
            publishMostFrequentFlag(current_time, current_dis);
            has_last_center_ = false;
        }
    }

    void addFlagToHistory(int flag_value) {
    if (flag_history_.size() >= flag_window_size_) {
        flag_history_.erase(flag_history_.begin());
    }
    flag_history_.push_back(flag_value);
    }

    void publishMostFrequentFlag(const ros::Time& current_time, double current_dis) {
        if (flag_history_.size() < flag_window_size_) {
            return; // 아직 충분한 데이터가 수집되지 않음
        }

        // 플래그 값의 빈도를 계산
        std::map<int, int> frequency;
        for (int val : flag_history_) {
            frequency[val]++;
        }

        // 최빈값 계산
        auto most_frequent = std::max_element(
            frequency.begin(), frequency.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });

        int most_frequent_flag = most_frequent->first;
        int most_frequent_count = most_frequent->second;

        // 7번 이상 발생한 경우에만 퍼블리시
        //좀 가까워졋을 때는 time_diff_flag 증가하지 않게 하기
        if (b_flag) {
            ros::Duration time_diff_flag = current_time - prev_flag_time;
            ROS_INFO("time_diff_flag : %.2f", time_diff_flag.toSec());
            std_msgs::Int32 msg;
            msg.data = most_frequent_flag;
            if (time_diff_flag.toSec()>=detection_time_diff_flag) {
                if (most_frequent_count >= 8) {
                    if (current_dis >=current_dis_flag) {
                        if (msg.data != 0 && msg.data != 4) {
                            flag_pub_.publish(msg);
                            ROS_INFO("Published most frequent flag: %d (Count: %d)", most_frequent_flag, most_frequent_count);

                        } 
                       //ROS_INFO("current_dis %.4f", current_dis);
                        //b_flag = false;

                    }
                    else {
                       // ROS_INFO("TOO CLOSE");
                    }
                } else {
                   // ROS_INFO("Most frequent flag: %d did not meet threshold (Count: %d)", most_frequent_flag, most_frequent_count);
                }
                b_flag = false;
            }
            

        }
        else {
            b_flag = true;
            prev_flag_time = current_time;
        }
           
        
    }

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cluster_direction_visualizer");
    ClusterDirectionVisualizer visualizer;

    ros::spin();
    return 0;
}