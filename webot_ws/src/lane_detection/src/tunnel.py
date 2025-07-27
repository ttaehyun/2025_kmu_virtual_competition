#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from math import pi,radians
from ackermann_msgs.msg import AckermannDriveStamped

class PIDController:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class Tunnel:
    def __init__(self):
        rospy.init_node('tunnel')
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.is_lidar=False
        self.points=[]

        self.target_control_pub = rospy.Publisher('high_level/ackermann_cmd_mux/input/nav_5', AckermannDriveStamped, queue_size=1)
        self.target_control = AckermannDriveStamped()
        ####### 변경 가능한 파라미터 #######
        self.target_control.drive.speed = 0.4
        self.distance_from_lidar_to_wall = 0.4
        self.pid = PIDController(0.6, 0.004, 0.8)
        self.max_steering_angle = 25
        ################################
        self.target_control.drive.steering_angle=0 # 차량 바퀴 초기화
        self.normalization = 0 # 정규화 초기화


    def lidar_callback(self,msg):
        self.is_lidar=True
        self.points=msg.ranges
        
        # LiDAR 데이터에서 정확한 90도와 -90도의 인덱스 계산
        index_270 = int((msg.angle_max + radians(90)) / msg.angle_increment) # 왼쪽 
        index_90 = int((msg.angle_max - radians(90)) / msg.angle_increment) # 오른쪽

        # 정확한 90도와 -90도에서의 거리 값 추출
        distance_270 = msg.ranges[index_90]
        distance_90 = msg.ranges[index_270]
        #print(distance_270,distance_90)

        if 0 < distance_270 < self.distance_from_lidar_to_wall and 0 < distance_90 < self.distance_from_lidar_to_wall:
            diff = distance_270-distance_90
            add = distance_270+distance_90
            self.normalization = self.max_steering_angle/add
            pid_diff = abs(self.pid.compute(diff))
            steering = pid_diff*self.normalization
            # 거리 비교하여 차량 조향 각도 조정
            if distance_270 > distance_90: 
                self.target_control.drive.steering_angle = steering*pi/180 # 왼쪽으로 조향 (90도 쪽이 더 멀면)
            elif distance_270 < distance_90:
                self.target_control.drive.steering_angle = -steering*pi/180 # 오른쪽으로 조향 (-90도 쪽이 더 멀면)
            else:
                self.target_control.drive.steering_angle = 0

            self.target_control_pub.publish(self.target_control)
            rospy.loginfo(f"tunnel_mission_pub")


if __name__ == '__main__':
    try:
        tunnel = Tunnel()
        rospy.spin()

    except rospy.ROSInternalException:
        pass 