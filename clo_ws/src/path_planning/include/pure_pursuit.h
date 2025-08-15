#ifndef PURE_PURSUIT_H
#define PURE_PURSUIT_H

#include "butterworth.h"

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/Marker.h>
#include <vesc_msgs/VescStateStamped.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <limits>

struct Carinfo
{
    double x;
    double y;
    
    double speed;
    double yaw;
};
class PurePursuitNode
{
    public:
    PurePursuitNode(ros::NodeHandle &nh);
    
    void spin();
private:
    ros::NodeHandle nh_;

    Carinfo webot_;

    ros::Subscriber path_sub_;
    ros::Subscriber velocity_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber current_v_sub_;

    ros::Publisher cmd_pub_;
    ros::Publisher lfd_pub_;
    ros::Publisher marker_vehicle_based_pub_;
    ros::Publisher angle_pub_;
    
    nav_msgs::Path local_path_;
    geometry_msgs::Pose current_pose_;

    double target_velocity_;

    double lookahead_distance_;
    double wheelbase;

    double max_steering;

    double k;      // Proportional coefficient for lookahead distance
    double offset; // Offset for lookahead distance

    double min_lfd;
    double max_lfd;

    bool is_pose_ready_;
    bool is_status_ready_;
    bool is_local_path_ready_;
    
    // Butterworth filters
    Butter2 butter_speed_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void updatePoseFromTF();
    void pathCallback(const nav_msgs::Path::ConstPtr &msg);
    void velocityCallback(const std_msgs::Float32::ConstPtr &msg);
    void currentV_Callback(const vesc_msgs::VescStateStamped::ConstPtr &msg);

    void publishMarker(const geometry_msgs::Point &pt, ros::Publisher &pub, const std::string &ns, float r, float g, float b);
    void updateLookaheadDistance();
    void computeControl();

    bool findLookaheadPoint_vehicleBased(geometry_msgs::Point &target);
    double computeSteeringAngle(const geometry_msgs::Point &target);
    double calculatePathLength(const nav_msgs::Path &path);
};

#endif