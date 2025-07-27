#!/usr/bin/env python3

import rospy
import numpy as np
from obstacle_detector.msg import Obstacles
from math import sin,sqrt,atan2
from ackermann_msgs.msg import AckermannDriveStamped

class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []
        self.point_list = [] 

        self.target_control_pub = rospy.Publisher('high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=1)
        
        # speed : lfd = 1 : 1.0 ~ 1.1

        self.target_control = AckermannDriveStamped()
        self.target_control.drive.steering_angle=0

        ########### 변경 가능 ##############
        self.target_control.drive.speed=0.4 # 최대 1.4 같음
        self.vehicle_length = 0.26
        self.lfd = 0.6
        self.distance_from_left_first_rubber_cone = 1.0
        self.distance_from_right_first_rubber_cone = 1.0
        self.distance_between_rubber_cone = 0.3
        ##################################

        self.is_look_forward_point = False

        self.mission_start = False

        rate = rospy.Rate(10)
        ################### main #######################
        while not rospy.is_shutdown():
            if self.mission_start == False:
                self.mission_start = self.rubber_cone_start()
            else:
                if self.rubber_cone_ing():
                    self.rubber_cone()
                else:
                    self.mission_start = False
            rate.sleep()

    def obstacle_callback(self, msg):
        self.is_obstacles = True
        self.obstacles = msg.circles

        self.point_list = [] 

        for obstacle in self.obstacles:
            point=(obstacle.center.x,obstacle.center.y)
            self.point_list.append(point)

    def add_line_points(self, points, current_point, line):
        next_points = points[(points[:, 0] < current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= self.distance_between_rubber_cone)]
        while next_points.size > 0:
            next_point = next_points[np.argmin(np.linalg.norm(next_points, axis=1))]
            if np.linalg.norm(next_point - current_point) <= self.distance_between_rubber_cone:
                line.append(next_point)
                current_point = next_point
                next_points = points[(points[:, 0] < current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= self.distance_between_rubber_cone)]
            else:
                break

    def rubber_cone_start(self):  # 러버콘 미션 시작 판단
        count = 0  # 조건을 만족하는 점의 수를 세는 변수
        for point in self.point_list:
            x, y = point
            if -1.3 < x < 0 and -0.6 < y < 0.6: # 환경에 맞게 수정 필요
                count += 1
                if count >= 6: # 환경에 맞게 수정 필요
                    return True  # 조건을 만족하는 포인트가 6개 이상일 경우 True 반환
                
        return False  # 6개 미만일 경우 False 반환
    
    def rubber_cone_ing(self):  # 러버콘 미션 진행 판단
        count = 0  # 조건을 만족하는 점의 수를 세는 변수

        for point in self.point_list:
            x, y = point
            if -1.0 < x < 0 and -1.0 < y < 1.0: # 환경에 맞게 수정 필요
                count += 1
                if count >= 4: # 환경에 맞게 수정 필요
                    return True  # 조건을 만족하는 포인트가 4개 이상일 경우 True 반환
                
        return False  # 4개 미만일 경우 False 반환

    def rubber_cone(self):
        if self.point_list:  # self.point_list가 비어있지 않은지 추가로 확인
            points = np.array(self.point_list)
            if points.size == 0:
                return
            
            # 왼쪽과 오른쪽 포인트 초기화 및 라인 구성
            left_line, right_line = [], []

            # 왼쪽 포인트 초기화
            left_points = points[(points[:, 1] < 0)]  # y < 0
            if left_points.size > 0:
                # 각도 필터링 (라디안을 도로 변환하여 -135도보다 큰 포인트 선택)
                angles = np.degrees(np.arctan2(left_points[:, 1], left_points[:, 0]))
                filtered_points = left_points[angles > -135]
                if filtered_points.size > 0:
                    current_point = filtered_points[np.argmin(np.linalg.norm(filtered_points, axis=1))]  # 조건을 만족하는 가장 가까운 포인트
                else:
                    current_point = left_points[np.argmin(np.linalg.norm(left_points, axis=1))]  # 조건을 만족하는 포인트가 없을 때 가장 가까운 포인트
                #print(f"left_line_first_point_dis: {np.linalg.norm(current_point)}")
                if np.linalg.norm(current_point) < self.distance_from_left_first_rubber_cone:  # 거리 0.8m 내
                    left_line.append(current_point)
                    self.add_line_points(points, current_point, left_line)

            # 오른쪽 포인트 초기화
            right_points = points[(points[:, 1] > 0)]  # y > 0
            if right_points.size > 0:
                # 각도 필터링 (라디안을 도로 변환하여 135도보다 작은 포인트 선택)
                angles = np.degrees(np.arctan2(right_points[:, 1], right_points[:, 0]))
                filtered_points = right_points[angles < 135]
                if filtered_points.size > 0:
                    current_point = filtered_points[np.argmin(np.linalg.norm(filtered_points, axis=1))]  # 조건을 만족하는 가장 가까운 포인트
                else:
                    current_point = right_points[np.argmin(np.linalg.norm(right_points, axis=1))]  # 조건을 만족하는 포인트가 없을 때 가장 가까운 포인트
                #print(f"right_line_first_point_dis: {np.linalg.norm(current_point)}")
                if np.linalg.norm(current_point) < self.distance_from_right_first_rubber_cone:
                    right_line.append(current_point)
                    self.add_line_points(points, current_point, right_line)

            # 경로 데이터 초기화
            center_line=[]
            if left_line != [] and right_line != []:
                left_size=len(left_line)
                right_size=len(right_line)

                if left_size > right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    for i in range(right_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        center_line.append((center_x, center_y))

                    diff_x=center_line[right_size-1][0]-left_line[right_size-1][0]
                    diff_y=center_line[right_size-1][1]-left_line[right_size-1][1]

                    for i in range(right_size,left_size):
                        center_x = left_line[i][0] + diff_x
                        center_y = left_line[i][1] + diff_y
                        center_line.append((center_x, center_y))

                elif left_size < right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    for i in range(left_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        center_line.append((center_x, center_y))

                    diff_x=center_line[left_size-1][0]-right_line[left_size-1][0]
                    diff_y=center_line[left_size-1][1]-right_line[left_size-1][1]

                    for i in range(left_size,right_size):
                        center_x = right_line[i][0] + diff_x
                        center_y = right_line[i][1] + diff_y
                        center_line.append((center_x, center_y))
                else:
                    for left_point,right_point in zip(left_line,right_line):
                        center_x=(left_point[0]+right_point[0]) / 2
                        center_y=(left_point[1]+right_point[1]) / 2
                        center_line.append((center_x, center_y))

            elif left_line != [] and right_line == []:
                diff_x = -left_line[0][0]
                diff_y = -left_line[0][1]
                for i in range(len(left_line)):
                    center_x = left_line[i][0] + diff_x
                    center_y = left_line[i][1] + diff_y
                    center_line.append((center_x, center_y))

            elif left_line == [] and right_line != []:
                diff_x = -right_line[0][0]
                diff_y = -right_line[0][1]
                for i in range(len(right_line)):
                    center_x = right_line[i][0] + diff_x
                    center_y = right_line[i][1] + diff_y
                    center_line.append((center_x, center_y))

            if center_line == []:
                self.target_control.drive.steering_angle=0
            else:
                # center_line 리스트를 x 좌표에 따라 정렬
                sorted_center_line = sorted(center_line, key=lambda point: point[0], reverse=True)
                theta = 0

                for point in sorted_center_line:
                    dis = sqrt(point[0]**2 + point[1]**2)
                    if dis >= self.lfd:
                        self.is_look_forward_point = True
                        theta = atan2(point[1]+sorted_center_line[0][1], point[0]+sorted_center_line[0][0])  # Correct order of arguments
                        break

                if self.is_look_forward_point:
                    self.target_control.drive.steering_angle = -atan2(2 * self.vehicle_length * sin(theta), self.lfd)
                    self.is_look_forward_point = False
                else:
                    theta = atan2(sorted_center_line[-1][1]+sorted_center_line[0][1], sorted_center_line[-1][0]+sorted_center_line[0][0])
                    self.target_control.drive.steering_angle = -atan2(2 * self.vehicle_length * sin(theta), self.lfd)
                
            self.target_control_pub.publish(self.target_control)
            rospy.loginfo("rubber cone 미션 수행 중")

if __name__ == '__main__':
    try:
        xycar = Rubber_cone()

    except rospy.ROSInternalException:
        pass 