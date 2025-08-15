#ifndef SUPERVISOR_CONTROL_H
#define SUPERVISOR_CONTROL_H

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

class SupervisorControl
{
public:
    SupervisorControl(ros::NodeHandle &nh);
    void spin();

private:
    ros::NodeHandle nh_;

    ros::Publisher ackermann_cmd_pub_;
    ros::Publisher w_state_pub_;

    ros::Subscriber velocity_sub_;

    ros::Subscriber pp_sub_;
    ros::Subscriber lane_angle_sub_;
    ros::Subscriber lane_confidence_sub_;
    ros::Subscriber lane_assist_sub_;
    ros::Subscriber obstacle_avoidance_sub_;

    void velocityCallback(const std_msgs::Float32::ConstPtr &msg);
    void ppCallback(const std_msgs::Float32::ConstPtr &msg);
    void laneAngleCallback(const std_msgs::Float32::ConstPtr &msg);
    void laneConfidenceCallback(const std_msgs::Float32::ConstPtr &msg);
    void laneAssistCallback(const std_msgs::Bool::ConstPtr &msg);
    void obstacleAvoidanceCallback(const std_msgs::Bool::ConstPtr &msg);

    ros::Time pp_ts_;
    ros::Time lane_angle_ts_;
    ros::Time lane_confidence_ts_;
    ros::Time lane_assist_ts_;
    ros::Time assist_ts_;
    // 파라미터
    double pp_timeout_ = 0.2;
    double lane_timeout_ = 0.2;
    double min_change_time_ = 0.5; // Minimum time to switch modes

    double w_max_ = 0.5;
    double k_lane_ = 1.0;

    // 히스테리시스 임계값
    double conf_on_ = 0.8;
    double conf_off_ = 0.4;

    // 가중치 EMA(Exponential Moving Average)
    double w_state_ = 0.0;
    // 파라미터
    double w_ema_alpha_ = 0.15;
    

    // 데이터

    double pp_angle_;
    double lane_angle_;
    double lane_confidence_;
    bool lane_assist_;
    bool obstacle_avoidance_;

    double target_velocity_;

    enum class Mode
    {
        NORMAL,
        LANE_ASSIST
    };

    Mode mode_ = Mode::NORMAL;
};

#endif // SUPERVISOR_CONTROL_H