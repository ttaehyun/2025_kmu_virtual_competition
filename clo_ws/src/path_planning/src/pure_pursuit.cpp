#include "pure_pursuit.h"
#include <cmath>


PurePursuitNode::PurePursuitNode(ros::NodeHandle &nh) : nh_(nh), webot_(), tf_buffer_(), tf_listener_(tf_buffer_), butter_speed_(nh_, 10.0, 50.0)
{
    path_sub_ = nh_.subscribe("/local_path", 1, &PurePursuitNode::pathCallback, this);
    velocity_sub_ = nh_.subscribe("/target_v", 1, &PurePursuitNode::velocityCallback, this);
    current_v_sub_ = nh_.subscribe("/sensors/core", 1, &PurePursuitNode::currentV_Callback, this);

    // cmd_pub_ = nh_.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd_mux/input/nav_2", 10);
    angle_pub_ = nh_.advertise<std_msgs::Float32>("/pp/angle", 1);
    lfd_pub_ = nh_.advertise<std_msgs::Float32>("/pp/lfd", 1); // Lookahead distance publisher
    marker_vehicle_based_pub_ = nh_.advertise<visualization_msgs::Marker>("/pp/target_marker_vehicle_based", 1);

    target_velocity_ = 1.0;

    lookahead_distance_ = 5;
    wheelbase = 0.3;
    max_steering = 0.3402;

    // 속도에 따른 lfd 계산 parameter
    // 1차 함수 lfd
    k = 0.3;      // 비례 계수
    offset = 0.3; // 최소 lookahead 거리

    min_lfd = 0.5; // 최소 lookahead 거리
    max_lfd = 3.0;

    is_pose_ready_ = false;
    is_status_ready_ = false;
    is_local_path_ready_ = false;
}

void PurePursuitNode::updatePoseFromTF()
{
    try
    {
        geometry_msgs::TransformStamped transformStamped;
        transformStamped = tf_buffer_.lookupTransform("map", "base_link", ros::Time(0));

        webot_.x = transformStamped.transform.translation.x;
        webot_.y = transformStamped.transform.translation.y;

        tf2::Quaternion q(
            transformStamped.transform.rotation.x,
            transformStamped.transform.rotation.y,
            transformStamped.transform.rotation.z,
            transformStamped.transform.rotation.w
        );

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        webot_.yaw = yaw;

        is_pose_ready_ = true;
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
        is_pose_ready_ = false;
    }
}

void PurePursuitNode::pathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    local_path_ = *msg;
    is_local_path_ready_ = true;
}

void PurePursuitNode::velocityCallback(const std_msgs::Float32::ConstPtr &msg)
{
    target_velocity_ = msg->data;
}

void PurePursuitNode::currentV_Callback(const vesc_msgs::VescStateStamped::ConstPtr &msg)
{
    double filtered_speed = butter_speed_.apply(msg->state.speed);
    double linear_speed = filtered_speed / 282.5;
    if (std::fabs(linear_speed) < 1e-3) linear_speed = 0.0;
    webot_.speed = fabs(linear_speed); // 확인 필요함(m/s단위여야함)
    is_status_ready_ = true;
}


void PurePursuitNode::computeControl()
{

    if (local_path_.poses.size() < 2)
    {
        ROS_INFO("Too short current_path!");
        return;
    }
    double length = calculatePathLength(local_path_);
    updateLookaheadDistance();

    geometry_msgs::Point vehicle_based_target;

    if (findLookaheadPoint_vehicleBased(vehicle_based_target))
    {
        publishMarker(vehicle_based_target, marker_vehicle_based_pub_, "vehicle_based", 1.0, 0.0, 0.0); // 빨간색
    }

    double angle = computeSteeringAngle(vehicle_based_target);
    // 클램프 max_steering
    angle = std::max(std::min(angle, max_steering), -max_steering);

    
    // ackermann_msgs::AckermannDriveStamped cmd_msg;
    // cmd_msg.header.stamp = ros::Time::now();
    // cmd_msg.header.frame_id = "base_link";
    // cmd_msg.drive.speed = target_velocity_;

    // cmd_msg.drive.steering_angle = -angle; // 방향 반대라서 바꿈
    // cmd_pub_.publish(cmd_msg);

    std_msgs::Float32 angle_msg;
    angle_msg.data = -angle; // 방향 반대라서 바꿈
    angle_pub_.publish(angle_msg); // Publish steering angle

    ROS_INFO("Steering: %.3f rad, taerget_v : %.2f Velocity: %.2f km/s (path_len: %.2f m)", -angle,target_velocity_, webot_.speed, length);
}

bool PurePursuitNode::findLookaheadPoint_vehicleBased(geometry_msgs::Point &target)
{
    if (local_path_.poses.empty())
        return false;

    // 1. 차량 위치
    double cx = webot_.x;
    double cy = webot_.y;

    // 2. 경로에서 가장 가까운 점 찾기
    size_t nearest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < local_path_.poses.size(); ++i)
    {
        const auto &pt = local_path_.poses[i].pose.position;
        double dist = std::hypot(pt.x - cx, pt.y - cy);

        if (dist < min_dist)
        {
            min_dist = dist;
            nearest_idx = i;
        }
    }

    // 3. 가장 가까운 점 이후로 누적 거리 계산
    double accumulated_dist = 0.0;

    for (size_t i = nearest_idx + 1; i < local_path_.poses.size(); ++i)
    {
        const auto &p1 = local_path_.poses[i - 1].pose.position;
        const auto &p2 = local_path_.poses[i].pose.position;

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double segment_len = std::hypot(dx, dy);

        accumulated_dist += segment_len;

        if (accumulated_dist >= lookahead_distance_)
        {
            double over = accumulated_dist - lookahead_distance_;
            double ratio = (segment_len - over) / segment_len;

            target.x = p1.x + ratio * dx;
            target.y = p1.y + ratio * dy;
            return true;
        }
    }

    // 경로가 너무 짧으면 마지막 점 사용
    target = local_path_.poses.back().pose.position;
    return true;
}

double PurePursuitNode::computeSteeringAngle(const geometry_msgs::Point &target)
{

    double dx = target.x - webot_.x;
    double dy = target.y - webot_.y;

    double yaw = webot_.yaw;

    double local_x = cos(yaw) * dx + sin(yaw) * dy;
    double local_y = -sin(yaw) * dx + cos(yaw) * dy;

    if (local_x == 0.0)
        return 0.0;

    double curvature = (2.0 * local_y) / (lookahead_distance_ * lookahead_distance_);
    ROS_INFO("curvature: %.3f", curvature);
    return std::atan(curvature * wheelbase);
}

double PurePursuitNode::calculatePathLength(const nav_msgs::Path &path)
{
    double total_length = 0.0;
    for (size_t i = 1; i < path.poses.size(); ++i)
    {
        const auto &p1 = path.poses[i - 1].pose.position;
        const auto &p2 = path.poses[i].pose.position;

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;

        total_length += std::hypot(dx, dy);
    }
    return total_length;
}

void PurePursuitNode::updateLookaheadDistance()
{
    // 1차 함수
    double lfd_raw = k * std::abs(webot_.speed) + offset;


    lookahead_distance_ = std::max(std::min(lfd_raw, max_lfd), min_lfd);

    lookahead_distance_ = 0.8;
    ROS_INFO("lfd_raw : %.2f, lookahead_distance : %.2f", lfd_raw, lookahead_distance_);
    std_msgs::Float32 lfd_msg;
    lfd_msg.data = lookahead_distance_;
    lfd_pub_.publish(lfd_msg); // Publish lookahead distance
}

void PurePursuitNode::publishMarker(const geometry_msgs::Point &pt, ros::Publisher &pub, const std::string &ns, float r, float g, float b)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map"; // 또는 "odom", 실제 프레임에 맞춰서
    marker.header.stamp = ros::Time::now();
    marker.ns = ns;
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position = pt;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.4;
    marker.scale.y = 0.4;
    marker.scale.z = 0.4;

    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration(0.1);
    pub.publish(marker);
}

// -------------------- Main Loop --------------------
void PurePursuitNode::spin()
{
    ros::Rate rate(20);
    while (ros::ok())
    {
        ros::spinOnce();
        updatePoseFromTF();
        if (!is_pose_ready_ && !is_status_ready_ && !is_local_path_ready_)
        {
            // ROS_WARN("Pose or status not ready, skipping control computation.");
            rate.sleep();
            continue;
        }
        
        computeControl();
        
        rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pure_pursuit_node");
    ros::NodeHandle nh;

    PurePursuitNode node(nh);
    node.spin();

    return 0;
}