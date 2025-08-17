#include <local_path.h>

// -------------------- Constructor --------------------
LocalPath::LocalPath(ros::NodeHandle &nh) : nh_(nh), tf_buffer_(), tf_listener_(tf_buffer_), butter_speed_(nh, 10.0, 50.0)
{
    is_pose_ready_ = is_obs_ready_ = false;
    is_status_ready_ = true;
    slam_and_navigation_mission_end_ = false;

    inside_global_path_.inside_path = outside_global_path_.outside_path = true;
    inside_global_path_.outside_path = outside_global_path_.inside_path = false;

    obsinfo_.inside_path  = false;
    obsinfo_.outside_path = true;

    global_path_ = &outside_global_path_;
    prev_global_path_ = &outside_global_path_;
    sub_q_ = 0.35;
    last_pose_sub_q_ = 0.35;
    sub_q_condition_ = 0.2;
    last_pose_sub_q_condition_ = 0.3;
    car_low_sub_q_ = false;
    last_pose_low_sub_q_ = false;
    num_samples_ = 3000; // 상황에 따라 수정
    num_of_path_ = 2;
    final_delta_s_ = 1.5;
    must_lane_chage_ = false; // 전방 장애물에 의한 차선 변경 변수
    s_min_ = 1.0;             // 상황에 따라 수정
    s_max_ = 2.0;            // 상황에 따라 수정
    delta_s_obs_sub_num_ = 0.3; // 상황에 따라 수정
    obstacle_avoidance_ = false;
    left_lane_ = false;
    right_lane_ = true;

    // Publishers
    optimal_path_pub_ = nh_.advertise<nav_msgs::Path>("/local_path", 1);
    target_v_pub_ = nh_.advertise<std_msgs::Float32>("/target_v", 1);
    obstacle_avoidance_pub_ = nh_.advertise<std_msgs::Bool>("/obstacle_avoidance", 1);
    left_lane_pub_ = nh_.advertise<std_msgs::Bool>("/left_lane", 1);
    right_lane_pub_ = nh_.advertise<std_msgs::Bool>("/right_lane", 1);

    // Subscribers
    inside_global_path_sub_ = nh_.subscribe("/kmu_in_path", 1, &LocalPath::insideGlobalPathCallback, this);
    outside_global_path_sub_ = nh_.subscribe("/kmu_out_path", 1, &LocalPath::outsideGlobalPathCallback, this);
    obs_sub_ = nh_.subscribe("/detected_obstacle_points_base_link", 1, &LocalPath::obsCallback, this);
    status_sub_ = nh_.subscribe("/sensors/core", 1, &LocalPath::VescStateCallback, this);
}

// -------------------- Main Loop --------------------
void LocalPath::spin()
{
    ros::Rate rate(20);
    while (ros::ok())
    {
        updatePoseFromTF();
        if(!slam_and_navigation_mission_end_)
            check_slam_and_navigation_mission_end();
        else
        {
            ros::spinOnce();
            if (inside_global_path_.global_path_ready && outside_global_path_.global_path_ready && is_pose_ready_ && is_status_ready_ && is_obs_ready_)
            {
                // 아크 길이와 횡방향 오프셋 계산
                Find_s_and_q(webot_, inside_global_path_, outside_global_path_);

                GlobalPathInfo *global_path = global_path_;
                Carinfo &webot = webot_;
                Obsinfo &obs1 = obsinfo_;
                vector<Obs> &intergrated_obs = intergrated_obs_;

                cout << "-------------------- Local Path Info --------------------" << endl;
                cout << "Global Path: " << (global_path->outside_path ? "Outside" : "Inside") << endl;
                cout << "x: " << webot.x << ", y: " << webot.y << endl;
                // cout << "yaw: " << webot.yaw * 180 / PI << ", speed(km/h): " << webot.speed * 3.6 << endl;
                cout << "s0: " << webot.s << ", q0: " << webot.q << endl;
                cout << "delta_s: " << final_delta_s_ << ", sub_q: " << sub_q_ << ", last_pose_sub_q: " <<last_pose_sub_q_<< endl;

                // 장애물 정보 업데이트
                updateobstacle(webot, global_path, obs1);

                // 장애물 통합
                intergration_obstacle(intergrated_obs, obs1);

                // 경로 후보군 생성
                generateCandidatePaths(webot, global_path, intergrated_obs, candidate_paths_);

                // 최적 경로 계산
                computeOptimalPath(webot, intergrated_obs, global_path, candidate_paths_, optimal_path_, target_v_);
                optimal_path_pub_.publish(optimal_path_);

                vmsg_.data = target_v_;
                target_v_pub_.publish(vmsg_);

                obstacle_avoidance_msg_.data = obstacle_avoidance_;
                obstacle_avoidance_pub_.publish(obstacle_avoidance_msg_);
                left_lane_msg_.data = left_lane_;
                left_lane_pub_.publish(left_lane_msg_);
                right_lane_msg_.data = right_lane_;
                right_lane_pub_.publish(right_lane_msg_);

                last_pose_ = optimal_path_.poses.back();
                compute_last_pose_sup_q(last_pose_, inside_global_path_, outside_global_path_);

                prev_global_path_ = global_path;
            }
            else
            {
                if (!inside_global_path_.global_path_ready)
                    ROS_WARN("Inside global path not ready");
                if (!outside_global_path_.global_path_ready)
                    ROS_WARN("Outside global path not ready");
                if (!is_pose_ready_)
                    ROS_WARN("Pose not ready");
                if (!is_obs_ready_)
                    ROS_WARN("Obs not ready");
                if (!is_status_ready_)
                    ROS_WARN("Status not ready");
            }
        }
        rate.sleep();
    }
}

// -------------------- Callback Implementations --------------------
void LocalPath::insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{

    if (msg->poses.empty())
    {
        ROS_WARN("Received empty inside_global_path");
        return;
    }
    if (msg->poses.size() < 2) {
        ROS_ERROR("inside path must contain at least 2 poses! (received %zu)", msg->poses.size());
        return;
    }
    if (!inside_global_path_.global_path_ready)
    {
        vector<double> x_points, y_points;
        for (const auto &pose : msg->poses)
        {
            x_points.push_back(pose.pose.position.x);
            y_points.push_back(pose.pose.position.y);
        }
        vector<double> s_vals;
        s_vals.clear();
        s_vals.push_back(0.0);
        for (size_t i = 1; i < x_points.size(); i++)
        {
            double dist = hypot(x_points[i] - x_points[i - 1], y_points[i] - y_points[i - 1]);
            s_vals.push_back(s_vals.back() + dist);
        }
        inside_global_path_.total_length = s_vals.back();

        if (inside_global_path_.s_candidates.empty())
        {
            double step;
            inside_global_path_.s_candidates.resize(num_samples_);
            step = inside_global_path_.total_length / (num_samples_ - 1);
            for (int i = 0; i < num_samples_; i++)
                inside_global_path_.s_candidates[i] = i * step;
        }
        // Create splines using tk::spline
        inside_global_path_.cs_x.set_points(s_vals, x_points);
        inside_global_path_.cs_y.set_points(s_vals, y_points);
        inside_global_path_.global_path_ready = true;
    }
}

void LocalPath::outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty outside_global_path");
        return;
    }
    if (msg->poses.size() < 2) {
        ROS_ERROR("inside path must contain at least 2 poses! (received %zu)", msg->poses.size());
        return;
    }
    if (!outside_global_path_.global_path_ready)
    {
        vector<double> x_points, y_points;
        for (const auto &pose : msg->poses)
        {
            x_points.push_back(pose.pose.position.x);
            y_points.push_back(pose.pose.position.y);
        }
        vector<double> s_vals;
        s_vals.clear();
        s_vals.push_back(0.0);
        for (size_t i = 1; i < x_points.size(); i++)
        {
            double dist = hypot(x_points[i] - x_points[i - 1], y_points[i] - y_points[i - 1]);
            s_vals.push_back(s_vals.back() + dist);
        }
        outside_global_path_.total_length = s_vals.back();

        if (outside_global_path_.s_candidates.empty())
        {
            double step;
            outside_global_path_.s_candidates.resize(num_samples_);
            step = outside_global_path_.total_length / (num_samples_ - 1);
            for (int i = 0; i < num_samples_; i++)
                outside_global_path_.s_candidates[i] = i * step;
        }
        // Create splines using tk::spline
        outside_global_path_.cs_x.set_points(s_vals, x_points);
        outside_global_path_.cs_y.set_points(s_vals, y_points);
        outside_global_path_.global_path_ready = true;
    }
}

void LocalPath::updatePoseFromTF()
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

void LocalPath::obsCallback(const sensor_msgs::PointCloud::ConstPtr &msg)
{
    if (!inside_global_path_.global_path_ready || !outside_global_path_.global_path_ready || inside_global_path_.s_candidates.empty() || outside_global_path_.s_candidates.empty() || !is_pose_ready_)
        return;

    GlobalPathInfo *global_path = global_path_;
    Carinfo &webot = webot_;

    // 이전 콜백에서 쌓인 장애물 정보 초기화
    obsinfo_.obs.clear();
    // 현재 어떤 전역 경로를 기준으로 계산했는지
    obsinfo_.inside_path = global_path->inside_path;
    obsinfo_.outside_path = global_path->outside_path;

    for (const auto &obs : msg->points)
    {
        double global_obs_x, global_obs_y;
        // global_obs_x = obs.position.x;
        // global_obs_y = obs.position.y;
        local_to_global(obs.x, obs.y, global_obs_x, global_obs_y, webot);

        double s_obs, q_obs;
        compute_obstacle_frenet_all(global_obs_x, global_obs_y, s_obs, q_obs, global_path);

        double ds_obs = s_obs - webot.s;
        if (ds_obs > global_path->total_length / 2.0)
            ds_obs -= global_path->total_length;
        else if (ds_obs < -global_path->total_length / 2.0)
            ds_obs += global_path->total_length;

        if (fabs(webot.q - q_obs) <= 2 * webot.be && 0.0 > ds_obs)
            continue; // 현재 차량 기준 후방 장애물 무시

        if(!last_pose_low_sub_q_ && car_low_sub_q_)
        {
            if (global_path->outside_path) // out
            {
                if (0.7 + last_pose_sub_q_ >= q_obs && q_obs >= -0.7) // 도로 규격 내에 존재해야함
                    if (q_obs <= last_pose_sub_q_ / 2)
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, true, ds_obs}); // 같은 차선에 존재
                    else
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, false, ds_obs}); // 다른 차선에 존재
            }
            else // in
            {
                if (0.7 >= q_obs && q_obs >= -last_pose_sub_q_ - 0.7) // 도로 규격 내에 존재해야함
                    if (q_obs >= -last_pose_sub_q_ / 2)
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, true, ds_obs}); // 같은 차선에 존재
                    else
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, false, ds_obs}); // 다른 차선에 존재
            }
        }
        else
        {
            if (global_path->outside_path) // out
            {
                if (0.7 + sub_q_ >= q_obs && q_obs >= -0.7) // 도로 규격 내에 존재해야함
                    if (q_obs <= sub_q_ / 2)
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, true, ds_obs}); // 같은 차선에 존재
                    else
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, false, ds_obs}); // 다른 차선에 존재
            }
            else // in
            {
                if (0.7 >= q_obs && q_obs >= -sub_q_ - 0.7) // 도로 규격 내에 존재해야함
                    if (q_obs >= -sub_q_ / 2)
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, true, ds_obs}); // 같은 차선에 존재
                    else
                        obsinfo_.obs.push_back({s_obs, q_obs, global_obs_x, global_obs_y, false, ds_obs}); // 다른 차선에 존재
            }
        }
    }
    is_obs_ready_ = true;
}

void LocalPath::VescStateCallback(const vesc_msgs::VescStateStamped::ConstPtr& msg)
{
    double filtered_speed = butter_speed_.apply(msg->state.speed);
    double linear_speed = filtered_speed / 1017.7;
    if (std::fabs(linear_speed) < 1e-3) linear_speed = 0.0;
    webot_.speed = fabs(linear_speed); // 확인 필요함(m/s단위여야함)
    is_status_ready_ = true;
}

void LocalPath::local_to_global(const double local_x, const double local_y, double &global_x, double &global_y, const Carinfo &car)
{
    double rotated_x = local_x * cos(car.yaw) - local_y * sin(car.yaw);
    double rotated_y = local_x * sin(car.yaw) + local_y * cos(car.yaw);

    global_x = rotated_x + car.x;
    global_y = rotated_y + car.y;
}

void LocalPath::check_slam_and_navigation_mission_end()
{
    double x = webot_.x;
    double y = webot_.y;

    // 사각형 영역: (-2.0, -5.0), (-2.0, -6.0), (0.0, -5.0), (0.0, -6.0)
    if (x >= -2.0 && x <=  0.0 && y >= -6.0 && y <= -5.0)
        slam_and_navigation_mission_end_ = true;
}

void LocalPath::Find_s_and_q(Carinfo &car, GlobalPathInfo &inside_global_path, GlobalPathInfo &outside_global_path)
{
    double inside_s0 = FindClosestSNewton(car.x, car.y, inside_global_path);
    double inside_q0 = SignedLateralOffset(car.x, car.y, inside_s0, inside_global_path.cs_x, inside_global_path.cs_y);

    double outside_s0 = FindClosestSNewton(car.x, car.y, outside_global_path);
    double outside_q0 = SignedLateralOffset(car.x, car.y, outside_s0, outside_global_path.cs_x, outside_global_path.cs_y);

    sub_q_ = fabs(inside_q0 - outside_q0);
    if (sub_q_ <= sub_q_condition_)
        car_low_sub_q_ = true;
    else
        car_low_sub_q_ = false;


    if (!car_low_sub_q_) // 차량이 직선 구간에 있음
    {
        if (global_path_->inside_path)
        {
            if (car.s < 37)
            {
                if (fabs(inside_q0) < fabs(outside_q0))
                {
                    car.s = inside_s0;
                    car.q = inside_q0;
                    global_path_ = &inside_global_path;
                    left_lane_ = true;
                    right_lane_ = false;
                }
                else
                {
                    car.s = outside_s0;
                    car.q = outside_q0;
                    global_path_ = &outside_global_path;
                    left_lane_ = false;
                    right_lane_ = true;
                }
            }
            else
            {
                car.s = outside_s0;
                car.q = outside_q0;
                global_path_ = &outside_global_path;
                left_lane_ = false;
                right_lane_ = true;
            }
        }
        else
        {
            if (car.s < 39)
            {
                if (fabs(inside_q0) < fabs(outside_q0))
                {
                    car.s = inside_s0;
                    car.q = inside_q0;
                    global_path_ = &inside_global_path;
                    left_lane_ = true;
                    right_lane_ = false;
                }
                else
                {
                    car.s = outside_s0;
                    car.q = outside_q0;
                    global_path_ = &outside_global_path;
                    left_lane_ = false;
                    right_lane_ = true;
                }
            }
            else
            {
                car.s = outside_s0;
                car.q = outside_q0;
                global_path_ = &outside_global_path;
                left_lane_ = false;
                right_lane_ = true;
            }
        }
    }
    else // 차량이 곡선 구간에 진입
    {
        if (global_path_->outside_path)
        {
            car.s = outside_s0;
            car.q = outside_q0;
        }
        else
        {
            car.s = inside_s0;
            car.q = inside_q0;
        }
    }
}

double LocalPath::FindClosestSNewton(const double x0, const double y0, const GlobalPathInfo &global_path)
{
    if (global_path.s_candidates.empty()) {
        ROS_ERROR("s_candidates is empty!");
        return 0.0;  // 혹은 합리적인 기본값
    }
    double s_current = global_path.s_candidates[0];
    // Vectorized initial guess using candidate points:
    double min_dist = 1e12;
    for (size_t i = 0; i < global_path.s_candidates.size(); i++)
    {
        double s_val = global_path.s_candidates[i];
        double dx = x0 - global_path.cs_x(s_val);
        double dy = y0 - global_path.cs_y(s_val);
        double d = dx * dx + dy * dy; // 실제 거리를 최소화 하는 것과, 제곱된 거리를 최소화 하는 것은 같은 최솟값의 위치를 준다.
        if (d < min_dist)
        {
            min_dist = d;
            s_current = s_val;
        }
    }
    int max_iter = 30;
    double tol = 1e-6;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // f(x)는 거리 제곱 함수
        double fprime = DistSqGrad(s_current, x0, y0, global_path.cs_x, global_path.cs_y);
        double fsecond = DistSqHess(s_current, x0, y0, global_path.cs_x, global_path.cs_y);
        if (fabs(fsecond) < 1e-12)
            break;
        double step = -fprime / fsecond;
        s_current += step;
        s_current = fmod(s_current, global_path.total_length);
        if (s_current < 0)
            s_current += global_path.total_length;
        if (fabs(step) < tol)
            break;
    }
    return s_current;
}

double LocalPath::DistSqGrad(const double s, const double x0, const double y0, const tk::spline &cs_x, const tk::spline &cs_y)
{
    double dx = x0 - cs_x(s);
    double dy = y0 - cs_y(s);
    double dxds = cs_x.deriv(1, s);
    double dyds = cs_y.deriv(1, s);
    return -2.0 * (dx * dxds + dy * dyds);
}

double LocalPath::DistSqHess(const double s, const double x0, const double y0, const tk::spline &cs_x, const tk::spline &cs_y)
{
    double dx = x0 - cs_x(s);
    double dy = y0 - cs_y(s);
    double dxds = cs_x.deriv(1, s);
    double dyds = cs_y.deriv(1, s);
    double d2xds2 = cs_x.deriv(2, s);
    double d2yds2 = cs_y.deriv(2, s);
    double val = (-dxds * dxds + dx * d2xds2) + (-dyds * dyds + dy * d2yds2);
    return -2.0 * val;
}

double LocalPath::SignedLateralOffset(const double x0, const double y0, const double s0, const tk::spline &cs_x, const tk::spline &cs_y)
{
    double x_s0 = cs_x(s0);
    double y_s0 = cs_y(s0);
    double dxds = cs_x.deriv(1, s0);
    double dyds = cs_y.deriv(1, s0);
    double dx_veh = x0 - x_s0;
    double dy_veh = y0 - y_s0;
    double cross_val = dxds * dy_veh - dyds * dx_veh;
    double q0 = sqrt(dx_veh * dx_veh + dy_veh * dy_veh);
    return (cross_val > 0) ? q0 : -q0;
}

void LocalPath::compute_last_pose_sup_q(geometry_msgs::PoseStamped last_pose, GlobalPathInfo &inside_global_path, GlobalPathInfo &outside_global_path)
{

    double x = last_pose.pose.position.x;
    double y = last_pose.pose.position.y;
    double inside_s0 = FindClosestSNewton(x, y, inside_global_path);
    double inside_q0 = SignedLateralOffset(x, y, inside_s0, inside_global_path.cs_x, inside_global_path.cs_y);

    double outside_s0 = FindClosestSNewton(x, y, outside_global_path);
    double outside_q0 = SignedLateralOffset(x, y, outside_s0, outside_global_path.cs_x, outside_global_path.cs_y);

    last_pose_sub_q_ = fabs(inside_q0 - outside_q0);

    if (last_pose_sub_q_ <= last_pose_sub_q_condition_)
        last_pose_low_sub_q_ = true;
    else
        last_pose_low_sub_q_ = false;
}

void LocalPath::compute_obstacle_frenet_all(const double obs_x0, const double obs_y0, double &obs_s0, double &obs_q0, const GlobalPathInfo *const global_path)
{
    obs_s0 = FindClosestSNewton(obs_x0, obs_y0, *global_path);
    obs_q0 = SignedLateralOffset(obs_x0, obs_y0, obs_s0, global_path->cs_x, global_path->cs_y);
}

void LocalPath::updateobstacle(Carinfo &car, const GlobalPathInfo *const global_path, Obsinfo &obsinfo)
{
    update(car.s, global_path, obsinfo);
}

void LocalPath::update(double s0, const GlobalPathInfo *const global_path, Obsinfo &obs_info)
{
    if (global_path->outside_path != obs_info.outside_path)
    {
        obs_info.inside_path = global_path->inside_path;
        obs_info.outside_path = global_path->outside_path;
        for (auto &obs : obs_info.obs)
        {
            double s_obs, q_obs;
            compute_obstacle_frenet_all(obs.x, obs.y, s_obs, q_obs, global_path);

            double ds_obs = s_obs - s0;
            if (ds_obs > global_path->total_length / 2.0)
                ds_obs -= global_path->total_length;
            else if (ds_obs < -global_path->total_length / 2.0)
                ds_obs += global_path->total_length;

            obs.s = s_obs;
            obs.q = q_obs;
            obs.ds_obs = ds_obs;

            if(!last_pose_low_sub_q_ && car_low_sub_q_)
            {
                if (global_path->outside_path) // out
                {
                    if (q_obs <= last_pose_sub_q_ / 2)
                        obs.same_path = true;
                    else
                        obs.same_path = false;
                }
                else // in
                {
                    if (q_obs >= -last_pose_sub_q_ / 2)
                        obs.same_path = true;
                    else
                        obs.same_path = false;
                }
            }
            else
            {
                if (global_path->outside_path) // out
                {
                    if (q_obs <= sub_q_ / 2)
                        obs.same_path = true;
                    else
                        obs.same_path = false;
                }
                else // in
                {
                    if (q_obs >= -sub_q_ / 2)
                        obs.same_path = true;
                    else
                        obs.same_path = false;
                }
            }
        }
    }
}

void LocalPath::intergration_obstacle(vector<Obs> &intergrated_obs, Obsinfo &obs1)
{
    intergrated_obs.clear();
    intergrated_obs.reserve(obs1.obs.size());
    intergrated_obs.insert(intergrated_obs.end(), obs1.obs.begin(), obs1.obs.end());
}

void LocalPath::generateCandidatePaths(Carinfo &car, const GlobalPathInfo *const global_path, vector<Obs> &intergrated_obs, vector<pair<nav_msgs::Path, Pathinfo>> &candidate_paths)
{
    candidate_paths.clear();
    vector<double> lane_offsets;
    ROS_INFO("→ generateCandidatePaths: car_low_sub_q_=%d, last_pose_low_sub_q_=%d, sub_q_=%f, last_pose_sub_q_=%f",
        car_low_sub_q_, last_pose_low_sub_q_, sub_q_, last_pose_sub_q_);

    if (!last_pose_low_sub_q_ && !car_low_sub_q_) // 차량이 직선 구간에 있음
    {
        if (global_path->outside_path)
        {
            for (double off = 0.0; off <= sub_q_ + 1e-3; off += sub_q_) // in path 방향으로 증가
                lane_offsets.push_back(off);
        }
        else
        {
            for (double off = 0.0; off >= -sub_q_ - 1e-3; off -= sub_q_) // out path 방향으로 감소
                lane_offsets.push_back(off);
        }
    }
    else if(last_pose_low_sub_q_ && !car_low_sub_q_) // 차량이 곡선 구간에 진입
    {
        lane_offsets.push_back(0.0);
    }
    else if(last_pose_low_sub_q_ && car_low_sub_q_) // 차량이 곡선 구간에서 주행 중
    {
        lane_offsets.push_back(0.0);
    }
    else if(!last_pose_low_sub_q_ && car_low_sub_q_) // 차량이 곡선 구간에서 빠져나옴
    {
        // if (global_path->outside_path)
        // {
        //     for (double off = 0.0; off <= last_pose_sub_q_ + 1e-3; off += last_pose_sub_q_) // in path 방향으로 증가
        //         lane_offsets.push_back(off);
        // }
        // else
        // {
        //     for (double off = 0.0; off >= -last_pose_sub_q_ - 1e-3; off -= last_pose_sub_q_) // out path 방향으로 감소
        //         lane_offsets.push_back(off);
        // }
        lane_offsets.push_back(0.0);
    }
    if (lane_offsets.empty()) {
        ROS_WARN("lane_offsets is empty — skipping candidate generation");
        return;
    }
    num_of_path_ = lane_offsets.size();
    candidate_paths.resize(num_of_path_);
    for (int i = 0; i < num_of_path_; ++i)
        generateLocalPath(car, lane_offsets[i], global_path, candidate_paths[i], intergrated_obs);
}

void LocalPath::generateLocalPath(Carinfo &car, double lane_offset, const GlobalPathInfo *const global_path, pair<nav_msgs::Path, Pathinfo> &path, vector<Obs> &intergrated_obs)
{
    path.first.header.frame_id = "map";

    double delta_s = compute_delta_s_vel(car);
    // double delta_s = s_max_;
    final_delta_s_ = compute_delta_s_with_obstacles(car.s, delta_s, intergrated_obs);

    double dxds = global_path->cs_x.deriv(1, car.s);
    double dyds = global_path->cs_y.deriv(1, car.s);
    double path_yaw = atan2(dyds, dxds);

    double dtheta = normalize_angle(car.yaw - path_yaw);
    double q_i = car.q;
    double dq_i = tan(dtheta);
    double q_f = lane_offset;
    double dq_f = 0.0;

    // 3차 스플라인 계수 계산: solve_cubic_spline_coeffs()를 호출하여 a, b, c, d 결정
    double a, b, c, d;
    solve_cubic_spline_coeffs(q_i, dq_i, q_f, dq_f, final_delta_s_, a, b, c, d);

    // t 구간 샘플링 (예: 10개의 샘플)
    int num_samples = 10;
    vector<double> t_samples(num_samples);
    double dt = final_delta_s_ / (num_samples - 1);
    for (int i = 0; i < num_samples; i++)
        t_samples[i] = i * dt;

    double prev_X = car.x;
    double prev_Y = car.y;
    double local_path_total_length = 0.0;

    for (double t : t_samples)
    {
        double q_val = eval_q_spline_t(a, b, c, d, t);
        double s_val = fmod(car.s + t, global_path->total_length);

        // Cartesian 변환
        double X, Y;
        frenetToCartesian(s_val, q_val, X, Y, global_path);
        local_path_total_length += hypot(X - prev_X, Y - prev_Y);
        prev_X = X;
        prev_Y = Y;

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = X;
        pose.pose.position.y = Y;
        pose.pose.orientation.w = 1.0;
        path.first.poses.push_back(pose);
    }
    path.second.length = local_path_total_length;
}

double LocalPath::normalize_angle(double angle)
{
    double two_pi = 2 * PI;
    double a = fmod(angle + PI, two_pi);
    if (a < 0)
        a += two_pi;
    return a - PI;
}

double LocalPath::compute_delta_s_vel(Carinfo &car)
{
    // double s_candidate = s_min_ + current_speed_ * current_speed_ / fabs(a_min_);
    double s_candidate = s_min_ + (s_max_ - s_min_) / car.v_max * car.speed;
    return std::min(s_candidate, s_max_);
}

double LocalPath::compute_delta_s_with_obstacles(double s0, double delta_s, vector<Obs> &intergrated_obs)
{
    vector<double> dist_candidates;
    for (const auto &obs : intergrated_obs)
    {
        if (obs.same_path)
        {
            if (obs.ds_obs >= 0 && obs.ds_obs <= s_max_)
                dist_candidates.push_back(obs.ds_obs);
        }
    }
    if (dist_candidates.empty())
        return delta_s;
    else
    {
        double obs_s0 = *min_element(dist_candidates.begin(), dist_candidates.end());
        return max(obs_s0 - delta_s_obs_sub_num_, s_min_);
    }
}

void LocalPath::solve_cubic_spline_coeffs(double q_i, double dq_i, double q_f, double dq_f, double ds,
                                          double &a, double &b, double &c, double &d)
{
    d = q_i;  // q(0) = q_i
    c = dq_i; // q'(0) = dq_i
    double X_f = ds;
    // 2x2 선형 시스템:
    // a_*X_f^3 + b_*X_f^2 = q_f - (c_*X_f + d_)
    // 3a_*X_f^2 + 2b_*X_f = dq_f - c_
    double A11 = X_f * X_f * X_f;
    double A12 = X_f * X_f;
    double A21 = 3 * X_f * X_f;
    double A22 = 2 * X_f;
    double B1 = q_f - (c * X_f + d);
    double B2 = dq_f - c;
    double det = A11 * A22 - A12 * A21;
    a = (B1 * A22 - A12 * B2) / det;
    b = (A11 * B2 - B1 * A21) / det;
}

double LocalPath::eval_q_spline_t(double a, double b, double c, double d, double t)
{
    return a * pow(t, 3) + b * pow(t, 2) + c * t + d;
}

void LocalPath::frenetToCartesian(const double s, const double q, double &X, double &Y, const GlobalPathInfo *const global_path)
{
    double x_s = global_path->cs_x(s);
    double y_s = global_path->cs_y(s);
    double dxds = global_path->cs_x.deriv(1, s);
    double dyds = global_path->cs_y.deriv(1, s);
    double normT = hypot(dxds, dyds);
    if (normT < 1e-9)
    {
        X = x_s;
        Y = y_s;
        return;
    }
    // 접선 벡터를 정규화하여 단위벡터로 만듬
    dxds /= normT;
    dyds /= normT;
    // 법선 벡터: (-dyds, dxds)
    double nx = -dyds;
    double ny = dxds;
    X = x_s + q * nx;
    Y = y_s + q * ny;
}

void LocalPath::computeOptimalPath(Carinfo &car, vector<Obs> &intergrated_obs, const GlobalPathInfo *const global_path, vector<pair<nav_msgs::Path, Pathinfo>> &candidate_paths, nav_msgs::Path &optimal_path, double &target_v)
{
    if (candidate_paths.empty()) {
        ROS_ERROR("No candidate paths available!");
        return;
    }

    // 장애물 검사용 플래그
    bool front_obs = false;
    bool side_obs = false;
    bool obstacle_flag = false;

    // 전체 목표 속도(장애물 고려 전)
    double path1_target_v = car.v_max;
    double path2_target_v = car.v_max;

    // 충돌 거리 한계
    double obs_front_limit = 0.8;
    double obs_sidefront_limit = obs_front_limit;
    double obs_sideback_limit = 0.5;
    double threshold = obs_front_limit + car.v_max;

    if (candidate_paths.size() == 1) // 경로 1개
    {
        must_lane_chage_ = false;
        auto &path1 = candidate_paths[0]; // 현재 경로
        path1.second.possible = true;
        path1.second.target_v = car.v_max;
        for (const auto &obs : intergrated_obs)
        {
            cout << "장애물 위치 s: " << obs.s << ", q: " << obs.q << ", 장애물과의 거리: " << obs.ds_obs << ", 장애물 차선 위치: " << obs.same_path << endl;

            // 전방 너무 가까우면 급정거
            if (0.0 <= obs.ds_obs && obs.ds_obs <= obs_front_limit)
            {
                path1.second.target_v = car.v_min;
                front_obs = true;
            }
            // 2) 기타 전방 범위 내 장애물이면 점진적 감속
            else if (0.0 <= obs.ds_obs && obs.ds_obs <= threshold)
            {
                double ego_to_obs_dist = hypot(obs.x - car.x, obs.y - car.y);
                double v_candidate = std::max(car.v_min, std::min(car.v_max, car.v_max + (ego_to_obs_dist - threshold)/2));
                path1_target_v = std::min(path1_target_v, v_candidate);
                if (!front_obs)
                    path1.second.target_v = std::min(path1_target_v, path1.second.target_v);
            }
        }
        optimal_path = path1.first;
        double v_limit = path1.second.target_v;
        double v_curv = computeCurvatureVelocity(car, path1);
        target_v = std::min(v_limit, v_curv);
        obstacle_avoidance_ = true;
        return;
    }
    else // 경로 2개
    {
        if (candidate_paths.size() < 2) {
            ROS_ERROR("Expected 2 candidate paths, got %zu", candidate_paths.size());
            return;
        }
        // 두 후보 경로 가져오기
        auto &path1 = candidate_paths[0]; // 현재 경로
        auto &path2 = candidate_paths[1]; // 다른 경로

        // possible, target_v 초기화
        path1.second.possible = true;
        path2.second.possible = true;
        path1.second.target_v = car.v_max;
        path2.second.target_v = car.v_max;

        // 장애물 목록 순회
        for (const auto &obs : intergrated_obs)
        {
            cout << "장애물 위치 s: " << obs.s << ", q: " << obs.q << ", 장애물과의 거리: " << obs.ds_obs << ", 장애물 차선 위치: " << obs.same_path << endl;

            // 1) 같은 차선 전방 너무 가까우면 급정거
            if (0.0 <= obs.ds_obs && obs.ds_obs <= obs_front_limit && obs.same_path)
            {
                path1.second.possible = false;
                path1.second.target_v = car.v_min;
                front_obs = true;
            }
            // 2) 다른 차선에 측방 장애물이면 차선 변경 금지
            else if (obs_sideback_limit <= obs.ds_obs && obs.ds_obs <= obs_sidefront_limit && !obs.same_path)
            {
                path2.second.possible = false;
                side_obs = true;
            }
            // 3) 기타 전방/측방 범위 내 장애물이면 점진적 감속
            else if (0.0 <= obs.ds_obs && obs.ds_obs <= threshold)
            {
                double ego_to_obs_dist = hypot(obs.x - car.x, obs.y - car.y);
                double v_candidate = max(car.v_min, min(car.v_max, car.v_max + (ego_to_obs_dist - threshold)/2));
                if (obs.same_path)
                    path1_target_v = min(path1_target_v, v_candidate);
                else
                    path2_target_v = min(path2_target_v, v_candidate);

                // 같은 차선/다른 차선별로 각 경로 속도 갱신
                if (obs.same_path && !front_obs)
                {
                    path1.second.target_v = min(path1_target_v, path1.second.target_v);
                    obstacle_flag = true;
                }
                else if (!obs.same_path && !side_obs)
                    path2.second.target_v = min(path2_target_v, path2.second.target_v);
            }
        }

        auto select_best = [&](auto &p)
        {
            optimal_path = p.first;
            double v_limit = p.second.target_v;
            double v_curv = computeCurvatureVelocity(car, p);
            target_v = std::min(v_limit, v_curv);
            return;
        };

        bool ok1 = path1.second.possible;
        bool ok2 = path2.second.possible;
        must_lane_chage_ = false;
        if (must_lane_chage_ && ok2)
        {
            if (prev_global_path_->outside_path == global_path->outside_path) // 아직 차선 변경 중
                return select_best(path2);
            else // 차선 변경 완료
            {
                must_lane_chage_ = false;
                return select_best(path1);
            }
        }
        else if (must_lane_chage_ && !ok2) // 차선 변경 하려는데 옆차선에 차량 등장
        {
            must_lane_chage_ = false;
            return select_best(path1);
        }

        if (ok1 && ok2)
        {
            if (obstacle_flag) // 둘 다 가능하지만 현재 차선 멀리에 장애물 존재
            {
                must_lane_chage_ = true;
                obstacle_avoidance_ = true;
                return select_best(path2);
            }
            else
            {
                obstacle_avoidance_ = false;
                return select_best(path1);
            }
        }
        else if (ok1)
        {
            obstacle_avoidance_ = false;
            return select_best(path1);
        }
        else if (ok2)
        {
            must_lane_chage_ = true;
            obstacle_avoidance_ = true;
            return select_best(path2);
        }
        else
        {
            obstacle_avoidance_ = false;
            return select_best(path1);
        }
    }
}

double LocalPath::computeCurvatureVelocity(Carinfo &car, const pair<nav_msgs::Path, Pathinfo> &path_info)
{
    const auto &path = path_info.first;

    vector<pair<double, double>> xy;
    xy.reserve(path.poses.size());
    for (const auto &pose : path.poses)
    {
        double X = pose.pose.position.x;
        double Y = pose.pose.position.y;
        xy.emplace_back(X, Y);
    }

    // 곡률 κ(s) 계산 & 최대값 찾기
    double max_kappa = 0.0;
    for (size_t i = 1; i + 1 < xy.size(); ++i)
    {
        double k = approxCurvature(
            xy[i - 1].first, xy[i - 1].second,
            xy[i].first, xy[i].second,
            xy[i + 1].first, xy[i + 1].second);
        max_kappa = max(max_kappa, fabs(k));
    }

    // cout << "최대 곡률: " << max_kappa << endl;

    // 최대 곡률에 따른 제한 속도 계산
    if (max_kappa < 1e-9)
    {
        // 곡률이 거의 0이면 직선으로 간주, 최고속도 반환
        return car.v_max;
    }
    else
    {
        return std::min(car.v_max, std::sqrt(car.a_lat_max / max_kappa));
    }
}

double LocalPath::approxCurvature(double x_prev, double y_prev, double x_curr, double y_curr, double x_next, double y_next)
{
    double ax = x_curr - x_prev, ay = y_curr - y_prev;
    double bx = x_next - x_curr, by = y_next - y_curr;
    double dot_ab = ax * bx + ay * by;
    double cross_ab = ax * by - ay * bx;
    double mag_a = hypot(ax, ay);
    double mag_b = hypot(bx, by);
    if (mag_a < 1e-9 || mag_b < 1e-9)
        return 0.0;
    double sin_theta = fabs(cross_ab) / (mag_a * mag_b);
    double chord = hypot(x_next - x_prev, y_next - y_prev);
    if (chord < 1e-9)
        return 0.0;
    double kappa = 2.0 * sin_theta / chord;
    return kappa;
}

// -------------------- Main --------------------
int main(int argc, char **argv)
{
    ros::init(argc, argv, "local_path_pub");
    ros::NodeHandle nh;
    LocalPath local_path_pub(nh);
    local_path_pub.spin();
    return 0;
}
