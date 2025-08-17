#include "supervisor_control.h"
#include <algorithm>

SupervisorControl::SupervisorControl(ros::NodeHandle &nh) : nh_(nh) {

    ackermann_cmd_pub_ = nh_.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd_mux/input/nav_2", 1);
    w_state_pub_ = nh_.advertise<std_msgs::Float32>("/w_state", 1);

    velocity_sub_ = nh_.subscribe("/target_v", 1, &SupervisorControl::velocityCallback, this);

    lane_angle_sub_ = nh_.subscribe("/lane/angle", 1, &SupervisorControl::laneAngleCallback, this);
    lane_confidence_sub_ = nh_.subscribe("/lane/confidence", 1, &SupervisorControl::laneConfidenceCallback, this);
    lane_assist_sub_ = nh_.subscribe("/lane/assistance", 1, &SupervisorControl::laneAssistCallback, this);
    obstacle_avoidance_sub_ = nh_.subscribe("/obstacle_avoidance", 1, &SupervisorControl::obstacleAvoidanceCallback, this);
    pp_sub_ = nh_.subscribe("/pp/angle", 1, &SupervisorControl::ppCallback, this);

    pp_timeout_ = 0.2;
    lane_timeout_ = 0.2;

    lane_assist_ = false;
    obstacle_avoidance_ = false;
    assist_ts_ = ros::Time::now();
}

void SupervisorControl::obstacleAvoidanceCallback(const std_msgs::Bool::ConstPtr &msg) {
    obstacle_avoidance_ = msg->data;
}

void SupervisorControl::velocityCallback(const std_msgs::Float32::ConstPtr &msg) {
    target_velocity_ = msg->data;
}

void SupervisorControl::ppCallback(const std_msgs::Float32::ConstPtr &msg) {
    pp_angle_ = msg->data;
    pp_ts_ = ros::Time::now();
}

void SupervisorControl::laneAngleCallback(const std_msgs::Float32::ConstPtr &msg) {
    lane_angle_ = msg->data;
    lane_angle_ts_ = ros::Time::now();
}

void SupervisorControl::laneConfidenceCallback(const std_msgs::Float32::ConstPtr &msg) {
    double confidence = msg->data;
    lane_confidence_ = std::max(0.0, std::min(1.0, confidence));
    lane_confidence_ts_ = ros::Time::now();
}

void SupervisorControl::laneAssistCallback(const std_msgs::Bool::ConstPtr &msg) {
    lane_assist_ = msg->data;
}

void SupervisorControl::spin() {
    const ros::Time now = ros::Time::now();
    const bool pp_fresh = (now - pp_ts_).toSec() < pp_timeout_;
    const bool lane_angle_fresh = (now - lane_angle_ts_).toSec() < lane_timeout_;
    const bool lane_confidence_fresh = (now - lane_confidence_ts_).toSec() < lane_timeout_;
    const bool can_switch = (now - assist_ts_).toSec() > min_change_time_;

    if (mode_ == Mode::NORMAL) {
        if (lane_assist_ && lane_angle_fresh && lane_confidence_fresh && pp_fresh && (lane_confidence_ >= conf_on_) && can_switch && !obstacle_avoidance_) {
            mode_ = Mode::LANE_ASSIST;
            assist_ts_ = now;
            ROS_INFO("Switching to LANE_ASSIST mode");
        }
    }
    else {
        if (!lane_assist_ || !lane_angle_fresh || !lane_confidence_fresh || !pp_fresh || (lane_confidence_ <= conf_off_) && can_switch || obstacle_avoidance_) {
            mode_ = Mode::NORMAL;
            assist_ts_ = now;
            ROS_INFO("Switching to NORMAL mode");
        }
    }

    

    // ---- 최종 각도 계산 ----
    double final_angle = 0.0; 
    
    if (pp_fresh) {

        // ---- 가변 가중치 계산 (ASSIST 모드일 때만) ----
        double w_target = 0.0;
        if (mode_ == Mode::LANE_ASSIST) {                 // or Mode::LANE_ASSIST
            double r;                                   // confidence 0..1로 정규화
            if (lane_confidence_ <= conf_off_) {
                r = 0.0;
            } 
            else {
                double denom = 1.0 - conf_off_;
                if (denom < 1e-6) denom = 1e-6;          // 0 나눗셈 방지
                r = (lane_confidence_ - conf_off_) / denom;
                if (r < 0.0) r = 0.0;
                if (r > 1.0) r = 1.0;
            }
            w_target = w_max_ * r;                      // 0..w_max_
        }

        // ---- 가중치 EMA ----
        w_state_ = (1.0 - w_ema_alpha_) * w_state_ + w_ema_alpha_ * w_target;

        // pp, lane, w 수동 클램프
        double pp = std::max(-1.0, std::min(1.0, pp_angle_));

        double lane = k_lane_ * lane_angle_;
        lane = std::max(-1.0, std::min(1.0, lane));

        double w = w_state_;
        w = std::max(w_min_, std::min(w_max_, w));
        
        final_angle = (1.0 - w) * pp + w * lane;

        // 모드에 따른 각도 결정
        // if (mode_ == Mode::LANE_ASSIST) {
        //     final_angle = lane_angle_;
        // }
        // else {
        //     final_angle = pp_angle_;
        // }
        std_msgs::Float32 w_state_msg;
        w_state_msg.data = w_state_;

        ackermann_msgs::AckermannDriveStamped final_cmd;
        final_cmd.header.stamp = now;
        final_cmd.header.frame_id = "base_link";
        final_cmd.drive.speed = target_velocity_;
    
        final_cmd.drive.steering_angle = final_angle;
    
        ackermann_cmd_pub_.publish(final_cmd);
        w_state_pub_.publish(w_state_msg);

        ROS_INFO(
        "[MODE] %s [Obstacle] %s | pp(%.3f,%c) lane(%.3f,%c) conf=(%.2f,%c)  final=%.3f",
        (mode_==Mode::NORMAL?"NORM":"ASSIST"),
        (obstacle_avoidance_ ? "ON" :"OFF"),
        pp_angle_,   pp_fresh?'Y':'N',
        lane_angle_, lane_angle_fresh?'Y':'N',
        lane_confidence_,  lane_confidence_fresh?'Y':'N',
        final_angle);
    }
    else {
        ROS_INFO("Slam mission");
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "supervisor_control_node");
    ros::NodeHandle nh("~");

    SupervisorControl supervisor(nh);
    ros::Rate rate(20);

    while(ros::ok()) {
        ros::spinOnce();
        supervisor.spin();
        rate.sleep();
    }

    return 0;
}