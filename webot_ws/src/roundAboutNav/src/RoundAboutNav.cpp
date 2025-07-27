#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <cmath>
#include <vector>

class SimpleLidarProcessor {
private:
    ros::NodeHandle nh_;
    ros::Subscriber lidar_sub_;
    ros::Publisher cluster_marker_pub_;
    ros::Publisher center_marker_pub_;
    ros::Publisher other_marker_pub_;
    double cluster_tolerance_ = 0.2;  // 유클리디언 거리 임계값
    double distanceMaxFromMe_ = 2.5;
    double distanceMinFromMe_ = 0.0;
    int min_cluster_size_ = 5;  // 최소 클러스터 크기

    const double max_y_ = 1.0;
    const double min_y_ = -1.0;
    
    const double max_x_ = 0.0;
    const double min_x_ = -2.2;

public:
    SimpleLidarProcessor() {
        lidar_sub_ = nh_.subscribe("/scan", 10, &SimpleLidarProcessor::lidarCallback, this);
        cluster_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/simple_clusters", 10);
        center_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/simple_cluster_centers", 10);
        other_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/other_center", 10);
    }

    visualization_msgs::Marker createMarkerTemplate(const std::string& ns, float r, float g, float b, int type) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "laser";
        marker.ns = ns;
        marker.type = type;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        return marker;
    }
    bool isWall(const std::vector<geometry_msgs::Point>& cluster) {
        double min_x = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::lowest();

        for (const auto& point : cluster) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }

        double length = std::sqrt(std::pow(max_x - min_x, 2) + std::pow(max_y - min_y, 2));
        return length <= 0.25; // 특정 길이 이상이면 벽으로 간주
    }



    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        // Step 1: 2D 포인트 생성
        std::vector<geometry_msgs::Point> points;
        float angle_min = msg->angle_min;
        for (size_t i =0; i<= 300; i++) {
            float range = msg->ranges[i];
            if (std::isfinite(range) && range < 3.0 && range > distanceMinFromMe_) {
                    geometry_msgs::Point point;
                    point.x = range * cos(angle_min);
                    point.y = range * sin(angle_min);
                    point.z = 0.0;
                    points.push_back(point);
            }
            angle_min+= msg->angle_increment;
        }
        float angle_max = msg->angle_max;
        for (size_t i =1284; i>=1084; i--) {
            float range = msg->ranges[i];
            if (std::isfinite(range) && range < 3.0 && range > distanceMinFromMe_) {
                geometry_msgs::Point point;
                point.x = range * cos(angle_max);
                point.y = range * sin(angle_max);
                point.z = 0.0;
                points.push_back(point);
            }
            angle_max += -1 * msg->angle_increment;
        }

        // Step 2: 클러스터링
        std::vector<std::vector<geometry_msgs::Point>> clusters;
        std::vector<bool> processed(points.size(), false);

        for (size_t i = 0; i < points.size(); ++i) {
            if (processed[i]) continue;

            std::vector<geometry_msgs::Point> cluster;
            findCluster(points, i, processed, cluster);
            if (cluster.size() >= min_cluster_size_) {
                if (isWall(cluster)) 
                    clusters.push_back(cluster);
                
            }
        }

        // Step 3: 시각화 메시지 생성

        visualization_msgs::Marker cluster_marker = createMarkerTemplate("simple_clusters", 0.0, 0.0, 1.0, visualization_msgs::Marker::POINTS);
        visualization_msgs::Marker center_marker = createMarkerTemplate("simple_centers", 0.0, 1.0, 0.0, visualization_msgs::Marker::SPHERE_LIST);
        visualization_msgs::Marker other_center_marker = createMarkerTemplate("simple_centers", 0.5, 1.0, 0.0, visualization_msgs::Marker::SPHERE_LIST);
        cluster_marker.header.stamp = msg->header.stamp;
        center_marker.header.stamp = msg->header.stamp;
        other_center_marker.header.stamp = msg->header.stamp;
        int count =0;
        for (const auto& cluster : clusters) {
            geometry_msgs::Point center;
            center.x = 0.0;
            center.y = 0.0;

            for (const auto& point : cluster) {
                cluster_marker.points.push_back(point);
                center.x += point.x;
                center.y += point.y;
            }

            center.x /= cluster.size();
            center.y /= cluster.size();

            // x, y좌표로 ROI 영역 정해주기 if문으로
            if (center.x <= max_x_ && center.x >= min_x_ && center.y <=max_y_ && center.y >=min_y_) {
                double dis = std::sqrt(std::pow(center.x, 2) + std::pow(center.y,2)); 
                if ( dis>= distanceMinFromMe_  && dis <=distanceMaxFromMe_) {
                    if (count == 0) {
                        center_marker.points.push_back(center);
                        
                    }
                    else {
                        //center_marker.points.clear();
                        
                    }
                    other_center_marker.points.push_back(center);
                    count++;
                }

            }
        }

        // Step 4: 퍼블리시
        //센터 포인트가 여러개 생기면 퍼블리쉬 안하게 설정 
        cluster_marker_pub_.publish(cluster_marker);
        if (count == 1) {
            if (!center_marker.points.empty()) {
                //ROS_INFO("center_marker count : %d | x: %.2f y: %.2f", center_marker.points.size(), center_marker.points[0].x, center_marker.points[0].y);
                //ROS_INFO("current_dis : %.4f", std::hypot(center_marker.points[0].x, center_marker.points[0].y));
            }
           
            other_center_marker.points.clear();
        }
        else {
            //ROS_INFO("Zero or Many_center_marker count : %d", other_center_marker.points.size());
            center_marker.points.clear();
        }
        center_marker_pub_.publish(center_marker);
        other_marker_pub_.publish(other_center_marker);
        
    }

    void findCluster(const std::vector<geometry_msgs::Point>& points, size_t idx, std::vector<bool>& processed, std::vector<geometry_msgs::Point>& cluster) {
        std::vector<size_t> queue = {idx};

        while (!queue.empty()) {
            size_t current_idx = queue.back();
            queue.pop_back();

            if (processed[current_idx]) continue;
            processed[current_idx] = true;

            
            cluster.push_back(points[current_idx]);
            
            for (size_t i = 0; i < points.size(); ++i) {
                if (!processed[i] && distance(points[current_idx], points[i]) < cluster_tolerance_) {
                    queue.push_back(i);
                }
            }
        }
    }

    double distance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simple_lidar_clustering");
    SimpleLidarProcessor processor;
    ros::spin();
    return 0;
}