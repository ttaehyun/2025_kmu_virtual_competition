#include <ros/ros.h>
#include <morai_msgs/GPSMessage.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/NavSatStatus.h>

class GPSToNavSatFix
{
public:
  GPSToNavSatFix()
  {
    gps_sub_ = nh_.subscribe("/gps", 10, &GPSToNavSatFix::gpsCallback, this);
    fix_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/fix", 10);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber gps_sub_;
  ros::Publisher  fix_pub_;

  void gpsCallback(const morai_msgs::GPSMessage::ConstPtr& msg)
  {
    sensor_msgs::NavSatFix fix;

    // Header
    fix.header.stamp    = ros::Time::now();
    fix.header.frame_id = msg->header.frame_id;

    // Status: morai_msgs/GPSMessage 의 status → NavSatStatus
    fix.status.status  = static_cast<int8_t>(msg->status);
    fix.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;

    // 위치 정보
    fix.latitude  = msg->latitude;
    fix.longitude = msg->longitude;
    fix.altitude  = msg->altitude;

    // 시뮬레이션이므로 매우 낮은 공분산 (1e-6) 로 대각선 채우기
    for (size_t i = 0; i < 9; ++i) {
      fix.position_covariance[i] = 1e-6;
    }
    fix.position_covariance_type =
      sensor_msgs::NavSatFix::COVARIANCE_TYPE_KNOWN;

    fix_pub_.publish(fix);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gps_to_fix");
  GPSToNavSatFix node;
  ros::spin();
  return 0;
}
