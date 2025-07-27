#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from math import cos,sin,pi,sqrt,pow,atan2,tan,radians, degrees
from std_msgs.msg import Header
from obstacle_detector.msg import Obstacles
import matplotlib.pyplot as plt
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
        self.points_size=897

        rospy.Subscriber("/raw_obstacles", Obstacles, self.segments_callback)
        self.is_segments = False
        self.segments = []

        self.target_control_pub = rospy.Publisher('high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=1)
        self.target_control = AckermannDriveStamped()
        self.target_control.drive.speed=1.0
        self.target_control.drive.steering_angle=0

        self.normalization = 0

        self.median_angle=0
        self.angles=None
        self.distances=None

        #plt.ion()  # 인터랙티브 모드 활성화
        #self.fig, self.ax = plt.subplots(figsize=(10, 4))  # 초기 플롯 생성
        self.k_u=0
        self.t_u=0

        self.pid = PIDController(0.5, 0.004, 0.8)

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # 데이터 시각화
            """
            plt.figure(figsize=(10, 4))
            plt.bar(self.angles, self.distances, width=1.0, color='blue')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Distance (meters)')
            plt.title('Lidar Data Histogram from -90 to 90 Degrees')
            plt.grid(True)
            plt.show()
            """
            """
            self.ax.clear()  # 이전 그래프를 지웁니다
            self.ax.bar(np.flip(self.angles), self.distances, width=1.0, color='blue')  # 막대 그래프 그리기
            self.ax.set_xlabel('Angle (degrees)')
            self.ax.set_ylabel('Distance (meters)')
            self.ax.set_title('Lidar Data Histogram from -90 to 90 Degrees')
            self.ax.grid(True)

            plt.draw()  # 그래프 업데이트
            plt.pause(0.001)  # 잠시 대기
            """
            # 평면 추출
            # 왼쪽 평면 path
            # 오른족 평면 path
            # 중앙 평면 path
            rate.sleep()

    def segments_callback(self, msg):
        self.is_segments = True
        self.segments = msg.segments
        #print(self.segments[0].first_point.x)

    def lidar_callback(self,msg):
        self.is_lidar=True
        self.points=msg.ranges
        # LiDAR 데이터에서 정확한 90도와 -90도의 인덱스 계산
        index_270 = int((msg.angle_max + radians(90)) / msg.angle_increment) # 왼쪽 
        index_90 = int((msg.angle_max - radians(90)) / msg.angle_increment) # 오른쪽
        print(index_270,index_90)


        # -90도 ~ 90도에서 검색
        # x축의 변화량 < y축 변화량이 더 큰 시작 point좌표 기록
        # 이후 계속해서 찾다가 10번 반복했는데 만족하지 않는 다면 끝 point 기록
        # 시작 point와 끝 point 길이 구함
        # 시작 point의 y좌표가 음수라면 distance_minus_90 + 길이
        # 시작 point의 y좌표가 양수라면 distance_90 - 길이


        """
        angle_range = 45  # 각도 범위 설정 (45도 ~ -45도)
        distance_threshold = 5.0  # 거리 제한 설정 (5m)

        # LiDAR 데이터에서 -45도에서 45도 범위의 데이터 인덱스 계산
        start_index = int((3*radians(angle_range) + msg.angle_min) / msg.angle_increment) # 135 - 180 = -45
        end_index = int((-3*radians(angle_range) + msg.angle_max) / msg.angle_increment)  # -135 + 180 = 45

        # 인덱스 범위가 실제 범위를 벗어나지 않도록 보정
        start_index = max(0, start_index)
        end_index = min(len(self.points) - 1, end_index)

        # 각도 배열 생성
        angles = np.linspace(-angle_range, angle_range, end_index - start_index + 1)

        # 거리 제한 적용
        filtered_ranges = np.array(self.points[start_index:end_index + 1])
        filtered_ranges[filtered_ranges > distance_threshold] = distance_threshold

         # distance_threshold 값에 해당하는 인덱스 추출
        threshold_indices = np.where(filtered_ranges == distance_threshold)[0]
        
        # 중앙값에 해당하는 각도 추출
        if len(threshold_indices) > 0:
            median_index = threshold_indices[len(threshold_indices) // 2]  # 중앙값 인덱스
            self.median_angle = -angles[median_index]*pi/180  # 중앙값 각도
            print("Median angle at max distance:", self.median_angle*180/pi)
        else:
            self.median_angle=0
        """

        # 정확한 90도와 -90도에서의 거리 값 추출
        distance_270 = msg.ranges[index_90]
        distance_90 = msg.ranges[index_270]



        # 각도 범위 설정: -90도에서 90도
        """
        start_index = int((radians(90) + msg.angle_min) / msg.angle_increment)  # -90도에 해당하는 인덱스
        end_index = int((radians(-90) + msg.angle_max) / msg.angle_increment)  # 90도에 해당하는 인덱스
        distances=[]
        #print(start_index,end_index)
        # -90도에서 90도 범위의 거리 데이터 추출
        for i in range(start_index,0,-1):
            distances.append(msg.ranges[i])
        for i in range(0,end_index+1):
            distances.append(msg.ranges[i])
        self.distances = distances
        self.distances = np.clip(self.distances, 0, 10)  # 10m 이상의 데이터는 10m로 제한
        self.angles = np.linspace(-90, 90, num=len(self.distances))
        #print(self.distances)
        """



        """
         # 각도 범위 설정: -90도에서 90도
        start_index = int((radians(-90) - msg.angle_min) / msg.angle_increment)  # 90도에 해당하는 인덱스
        end_index = int((radians(90) - msg.angle_min) / msg.angle_increment)  # 270도에 해당하는 인덱스

        # 인덱스 범위가 실제 범위를 벗어나지 않도록 보정
        #start_index = max(0, start_index)
        #end_index = min(len(self.points) - 1, end_index)

        distances = []
        # -90도에서 90도 범위의 거리 데이터 추출
        if start_index <= end_index:
            distances = self.points[start_index:end_index + 1]
        else:
            # 스캔 범위가 360도가 아닐 경우 start_index가 end_index보다 클 수 있습니다.
            distances = self.points[start_index:] + self.points[:end_index + 1]

        self.distances = distances
        self.distances = np.clip(self.distances, 0, 10)  # 10m 이상의 데이터는 10m로 제한
        self.angles = np.linspace(-90, 90, num=len(distances))
        #print(self.distances)



        # 인덱스 범위가 실제 범위를 벗어나지 않도록 보정
        #start_index = min(0, start_index)
        #end_index = min(len(self.points) - 1, end_index + 1)

        #print(msg.angle_min,start_index*msg.angle_increment,end_index*msg.angle_increment)

        start_point = None
        end_point = None
        counter = 0
        segment_length=0

        for i in range(start_index, end_index):
            current_point=(self.points[i]*cos(i*msg.angle_increment),self.points[i]*sin(i*msg.angle_increment))
            next_point=(self.points[i+1]*cos((i+1)*msg.angle_increment),self.points[i+1]*sin((i+1)*msg.angle_increment))
            if abs(current_point[0]-next_point[0]) < abs(current_point[1]-next_point[1]) and start_point == None: # y의 변화량이 크다면 전방에 장애물이 있다고 판단
                start_point=current_point
            elif start_point != None and abs(current_point[0]-next_point[0]) >= abs(current_point[1]-next_point[1]):
                if counter >= 3:
                    break
                elif counter == 0:
                    end_point = current_point
                    counter += 1
                else:
                    counter += 1
            elif start_point != None and abs(current_point[0]-next_point[0]) < abs(current_point[1]-next_point[1]):
                end_point = None
                counter = 0

        #print(start_point,end_point)

        if start_point and end_point:
            segment_length = sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
            if abs(start_point[1]) > abs(end_point[1]):
                adjusted_distance = distance_minus_90 + segment_length
            else:
                adjusted_distance = distance_90 - segment_length
            """


        if 0 < distance_270 < 1 and 0 < distance_90 < 1:
            print(distance_270,distance_90)
            diff=distance_270-distance_90
            add=distance_270+distance_90
            self.normalization=25/add
            pid_diff=abs(self.pid.compute(diff))
            steering=pid_diff*self.normalization
            #print(steering)
            # 거리 비교하여 차량 조향 각도 조정
            if distance_270 > distance_90: 
                self.target_control.drive.steering_angle = steering*pi/180 # +self.median_angle  # 왼쪽으로 조향 (90도 쪽이 더 멀면)
            elif distance_270 < distance_90:
                self.target_control.drive.steering_angle = -steering*pi/180 # +self.median_angle # 오른쪽으로 조향 (-90도 쪽이 더 멀면)
            else:
                self.target_control.drive.steering_angle=0

        else:
            self.target_control.drive.steering_angle=0

        # 조향 메시지 발행
        #print(segment_length)
        self.target_control_pub.publish(self.target_control)

if __name__ == '__main__':
    try:
        xycar = Tunnel()

    except rospy.ROSInternalException:
        pass 