#ifndef _LOCAL_PATH_H_
#define _LOCAL_PATH_H_

#include "spline.h"
#include "butterworth.h"

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud.h>
#include <morai_msgs/ObjectStatusList.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <vesc_msgs/VescStateStamped.h>

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

using namespace std;

constexpr double PI = 3.14159265358979323846;

struct GlobalPathInfo
{
  bool global_path_ready = false;
  tk::spline cs_x, cs_y;
  vector<double> s_candidates;
  double total_length;
  bool inside_path;
  bool outside_path;
};

struct Carinfo
{
  double x;
  double y;
  double s;
  double q;
  double speed;
  double yaw;
  double ae = 0.21915;      // 상황에 따라 수정
  double be = 0.11;      // 상황에 따라 수정
  double a_min = -4.0;    // 상황에 따라 수정
  double a_max = 4.0;     // 상황에 따라 수정
  double a_lat_max = 2.0; // 상황에 따라 수정
  double v_min = 0.0;     // m/s          // 상황에 따라 수정
  double v_max = 2.0;    // m/s         // 상황에 따라 수정
};

struct Test_Obs
{
  double x;
  double y;
};

struct Pathinfo
{
  bool possible;
  double target_v;
  double length;
};

struct Obs
{
  double s;
  double q;
  double x;
  double y;
  bool same_path;
  double ds_obs;
};

struct Obsinfo
{
  vector<Obs> obs;
  bool inside_path;
  bool outside_path;
};

class LocalPath
{
public:
  LocalPath(ros::NodeHandle &nh);
  void spin();

private:
  // NodeHandle
  ros::NodeHandle nh_;

  // Publishers
  ros::Publisher optimal_path_pub_;
  ros::Publisher target_v_pub_;
  ros::Publisher obstacle_avoidance_pub_;
  ros::Publisher left_lane_pub_;
  ros::Publisher right_lane_pub_;

  // Subscribers
  ros::Subscriber inside_global_path_sub_;
  ros::Subscriber outside_global_path_sub_;
  ros::Subscriber status_sub_;
  ros::Subscriber obs_sub_;

  bool is_pose_ready_, is_obs_ready_, is_status_ready_;
  bool slam_and_navigation_mission_end_;

  Carinfo webot_;
  GlobalPathInfo inside_global_path_, outside_global_path_;
  GlobalPathInfo *global_path_;
  GlobalPathInfo *prev_global_path_;

  double sub_q_;
  double last_pose_sub_q_;
  double sub_q_condition_;
  double last_pose_sub_q_condition_;
  bool car_low_sub_q_;
  bool last_pose_low_sub_q_;
  int num_samples_;
  int num_of_path_;
  nav_msgs::Path optimal_path_;
  double target_v_;
  std_msgs::Float32 vmsg_;
  double final_delta_s_;
  bool must_lane_chage_;
  double s_min_, s_max_;
  geometry_msgs::PoseStamped last_pose_;
  double delta_s_obs_sub_num_;

  bool obstacle_avoidance_;
  std_msgs::Bool obstacle_avoidance_msg_;
  bool left_lane_;
  std_msgs::Bool left_lane_msg_;
  bool right_lane_;
  std_msgs::Bool right_lane_msg_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // 장애물 정보 (Frenet: (s,q) 저장)
  vector<Obs> intergrated_obs_;
  Obsinfo obsinfo_;

  // Butterworth filters
  Butter2 butter_speed_;

  vector<pair<nav_msgs::Path, Pathinfo>> candidate_paths_;

  // Callback functions
  void insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
  void outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
  void updatePoseFromTF();
  void obsCallback(const sensor_msgs::PointCloud::ConstPtr &msg);
  void VescStateCallback(const vesc_msgs::VescStateStamped::ConstPtr& msg);

  void local_to_global(const double local_x, const double local_y, double &global_x, double &global_y, const Carinfo &car);
  void check_slam_and_navigation_mission_end();

  void Find_s_and_q(Carinfo &car, GlobalPathInfo &inside_global_path, GlobalPathInfo &outside_global_path);
  double FindClosestSNewton(const double x0, const double y0, const GlobalPathInfo &global_path);
  double DistSqGrad(const double s, const double x0, const double y0, const tk::spline &cs_x, const tk::spline &cs_y);
  double DistSqHess(const double s, const double x0, const double y0, const tk::spline &cs_x, const tk::spline &cs_y);
  double SignedLateralOffset(const double x0, const double y0, const double s0, const tk::spline &cs_x, const tk::spline &cs_y);

  void compute_last_pose_sup_q(geometry_msgs::PoseStamped last_pose, GlobalPathInfo &inside_global_path, GlobalPathInfo &outside_global_path);
  void compute_obstacle_frenet_all(const double obs_x0, const double obs_y0, double &obs_s0, double &obs_q0, const GlobalPathInfo *const global_path);
  void frenetToCartesian(const double s, const double q, double &X, double &Y, const GlobalPathInfo *const global_path);

  void updateobstacle(Carinfo &car, const GlobalPathInfo *const global_path, Obsinfo &obsinfo);
  void update(double s0, const GlobalPathInfo *const global_path, Obsinfo &obs_info);
  void intergration_obstacle(vector<Obs> &intergrated_obs, Obsinfo &obs1);

  void generateCandidatePaths(Carinfo &car, const GlobalPathInfo *const global_path, vector<Obs> &intergrated_obs, vector<pair<nav_msgs::Path, Pathinfo>> &candidate_paths);
  void generateLocalPath(Carinfo &car, double lane_offset, const GlobalPathInfo *const global_path, pair<nav_msgs::Path, Pathinfo> &path, vector<Obs> &intergrated_obs);
  double compute_delta_s_vel(Carinfo &car);
  double compute_delta_s_with_obstacles(double s0, double delta_s, vector<Obs> &intergrated_obs);
  double normalize_angle(double angle);
  void solve_cubic_spline_coeffs(double q_i, double dq_i, double q_f, double dq_f, double ds, double &a, double &b, double &c, double &d);
  double eval_q_spline_t(double a, double b, double c, double d, double t);

  void computeOptimalPath(Carinfo &car, vector<Obs> &intergrated_obs, const GlobalPathInfo *const global_path, vector<pair<nav_msgs::Path, Pathinfo>> &candidate_paths, nav_msgs::Path &optimal_path, double &target_v);
  double computeCurvatureVelocity(Carinfo &car, const pair<nav_msgs::Path, Pathinfo> &path_info);
  double approxCurvature(double x_prev, double y_prev, double x_curr, double y_curr, double x_next, double y_next);
};

#endif
