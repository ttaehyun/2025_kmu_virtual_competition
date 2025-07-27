#!/usr/bin/env python3

import rospy
import numpy as np
from obstacle_detector.msg import Obstacles
from math import cos,sin,pi,sqrt,pow,atan2,tan
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Point
from ackermann_msgs.msg import AckermannDriveStamped

class Rubber_cone:
    def __init__(self):
        rospy.init_node('rubber_cone')
        rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacle_callback)
        self.is_obstacles = False
        self.obstacles = []
        self.point_list = [] 

        self.target_control_pub = rospy.Publisher('high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=1)

        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=1)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=1)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=1)
        
        # speed : lfd = 1 : 1.0 ~ 1.1

        self.target_control = AckermannDriveStamped()
        self.target_control.drive.speed=0.5
        self.target_control.drive.steering_angle=0

        self.vehicle_length = 0.32
        self.lfd = 1.0
        self.forward_point = Point()
        self.is_look_forward_point = False

        self.mission_start = False

        rate = rospy.Rate(10)

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

    def add_line_points(self, points, current_point, line, side):
        dis=0.6 # 범위 거리
        next_points = points[(points[:, 0] < current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= dis)]
        while next_points.size > 0:
            next_point = next_points[np.argmin(np.linalg.norm(next_points, axis=1))]
            #print(np.linalg.norm(next_point - current_point))
            if np.linalg.norm(next_point - current_point) <= dis:
                line.append(next_point)
                current_point = next_point
                next_points = points[(points[:, 0] < current_point[0]) & (np.linalg.norm(points - current_point, axis=1) <= dis)]
            else:
                break

    def rubber_cone_start(self):  # 러버콘 미션 시작 판단
        count = 0  # 조건을 만족하는 점의 수를 세는 변수
        for point in self.point_list:
            x, y = point
            if -1.2 < x < 0 and -0.4 < y < 0.4:
                count += 1
                if count >= 6:
                    return True  # 조건을 만족하는 포인트가 6개 이상일 경우 True 반환
                
        return False  # 6개 미만일 경우 False 반환
    
    def rubber_cone_ing(self):  # 러버콘 미션 진행 판단
        count = 0  # 조건을 만족하는 점의 수를 세는 변수

        for point in self.point_list:
            x, y = point
            if -1.2 < x < 0 and -0.6 < y < 0.6:
                count += 1
                if count >= 4:
                    return True  # 조건을 만족하는 포인트가 4개 이상일 경우 True 반환
                
        return False  # 4개 미만일 경우 False 반환

    def rubber_cone(self):
        if self.point_list:  # self.point_list가 비어있지 않은지 추가로 확인
            points = np.array(self.point_list)
            if points.size == 0:
                rospy.loginfo("No points received yet.")
                return
            
            # 왼쪽과 오른쪽 포인트 초기화 및 라인 구성
            left_line, right_line = [], []

            # 왼쪽 포인트 초기화
            left_points = points[(points[:, 1] < 0)] # y < 0
            if left_points.size > 0:
                current_point = left_points[np.argmin(np.linalg.norm(left_points, axis=1))] # 왼쪽에 있는 포인트 중에서 제일 가까운 포인트
                if np.linalg.norm(current_point) < 0.8: # 거리 3.0m 내
                    left_line.append(current_point)
                    self.add_line_points(points, current_point, left_line, 'left')

            # 오른쪽 포인트 초기화
            right_points = points[(points[:, 1] > 0)] # y > 0
            if right_points.size > 0:
                current_point = right_points[np.argmin(np.linalg.norm(right_points, axis=1))] # 오른쪽에 있는 포인트 중에서 제일 가까운 포인트
                #print(np.linalg.norm(current_point))
                if np.linalg.norm(current_point) < 0.8:
                    right_line.append(current_point)
                    self.add_line_points(points, current_point, right_line, 'right')

            # 경로 데이터 초기화
            right_path = Path()
            right_path.header = Header(stamp=rospy.Time.now(), frame_id='laser')
            if right_line != []:
                for point in right_line:
                    pose = PoseStamped()
                    pose.header = Header(stamp=rospy.Time.now(), frame_id='laser')
                    pose.pose.position.x = point[0]
                    pose.pose.position.y = point[1]
                    pose.pose.position.z = 0
                    right_path.poses.append(pose)

            self.right_path_pub.publish(right_path)

            # 경로 데이터 초기화
            left_path = Path()
            left_path.header = Header(stamp=rospy.Time.now(), frame_id='laser')
            if left_line != []:
                for point in left_line:
                    pose = PoseStamped()
                    pose.header = Header(stamp=rospy.Time.now(), frame_id='laser')
                    pose.pose.position.x = point[0]
                    pose.pose.position.y = point[1]
                    pose.pose.position.z = 0
                    left_path.poses.append(pose)

            self.left_path_pub.publish(left_path)

            # 경로 데이터 초기화
            center_line=[]
            if left_line != [] and right_line != []:
                left_size=len(left_line)
                right_size=len(right_line)

                if left_size > right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    prev_x=0
                    prev_y=0
                    for i in range(right_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            #center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                    diff_x=center_line[right_size-1][0]-left_line[right_size-1][0]
                    diff_y=center_line[right_size-1][1]-left_line[right_size-1][1]

                    prev_x=0
                    prev_y=0
                    for i in range(right_size,left_size):
                        center_x = left_line[i][0] + diff_x
                        center_y = left_line[i][1] + diff_y
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            #center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                elif left_size < right_size:
                    # 더 작은 배열의 크기에 맞춰서 center_line을 계산
                    prev_x=0
                    prev_y=0
                    for i in range(left_size):
                        # 왼쪽과 오른쪽 라인의 평균 위치 계산
                        center_x = (left_line[i][0] + right_line[i][0]) / 2
                        center_y = (left_line[i][1] + right_line[i][1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            #center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

                    diff_x=center_line[left_size-1][0]-right_line[left_size-1][0]
                    diff_y=center_line[left_size-1][1]-right_line[left_size-1][1]

                    prev_x=0
                    prev_y=0
                    for i in range(left_size,right_size):
                        center_x = right_line[i][0] + diff_x
                        center_y = right_line[i][1] + diff_y
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            #center_line.append((between_center_x, between_center_y))
                        # 계산된 중앙점을 center_line에 추가
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y
                else:
                    prev_x=0
                    prev_y=0
                    for left_point,right_point in zip(left_line,right_line):
                        center_x=(left_point[0]+right_point[0]) / 2
                        center_y=(left_point[1]+right_point[1]) / 2
                        if prev_x != 0 and prev_y != 0:
                            between_center_x=(center_x+prev_x)/2
                            between_center_y=(center_y+prev_y)/2
                            #center_line.append((between_center_x, between_center_y))
                        center_line.append((center_x, center_y))
                        prev_x=center_x
                        prev_y=center_y

            elif left_line != [] and right_line == []:
                diff_x = -left_line[0][0]
                diff_y = -left_line[0][1]
                prev_x=0
                prev_y=0
                for i in range(len(left_line)):
                    center_x = left_line[i][0] + diff_x
                    center_y = left_line[i][1] + diff_y
                    if prev_x != 0 and prev_y != 0:
                        between_center_x=(center_x+prev_x)/2
                        between_center_y=(center_y+prev_y)/2
                        #center_line.append((between_center_x, between_center_y))
                    # 계산된 중앙점을 center_line에 추가
                    center_line.append((center_x, center_y))
                    prev_x=center_x
                    prev_y=center_y

            elif left_line == [] and right_line != []:
                diff_x = -right_line[0][0]
                diff_y = -right_line[0][1]
                prev_x=0
                prev_y=0
                for i in range(len(right_line)):
                    center_x = right_line[i][0] + diff_x
                    center_y = right_line[i][1] + diff_y
                    if prev_x != 0 and prev_y != 0:
                        between_center_x=(center_x+prev_x)/2
                        between_center_y=(center_y+prev_y)/2
                        #center_line.append((between_center_x, between_center_y))
                    # 계산된 중앙점을 center_line에 추가
                    center_line.append((center_x, center_y))
                    prev_x=center_x
                    prev_y=center_y

            if center_line == []:
                self.target_control.drive.steering_angle=0
            else:
                # 경로 데이터 초기화
                center_path = Path()
                center_path.header = Header(stamp=rospy.Time.now(), frame_id='laser')
                # center_line 리스트를 x 좌표에 따라 정렬
                sorted_center_line = sorted(center_line, key=lambda point: point[0], reverse=True)
                
                # 두 점 사이 중간값을 추가한 새로운 리스트 생성
                enhanced_center_line = []
                for i in range(len(sorted_center_line) - 1):
                    # 현재 점과 다음 점
                    point1 = sorted_center_line[i]
                    point2 = sorted_center_line[i + 1]
                    
                    # 두 점 사이의 중간값 계산
                    midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
                    
                    # 현재 점과 중간값 추가
                    enhanced_center_line.append(point1)
                    enhanced_center_line.append(midpoint)

                # 마지막 점 추가
                enhanced_center_line.append(sorted_center_line[-1])

                #distance_to_center_line_based_on_car = sorted_center_line[0][1]
                #self.pid.compute(distance_to_center_line_based_on_car)
                #print(distance_to_center_line_based_on_car*180/pi)

                if sorted_center_line:
                    for point in sorted_center_line:
                        pose = PoseStamped()
                        pose.header = Header(stamp=rospy.Time.now(), frame_id='laser')
                        pose.pose.position.x = point[0]
                        pose.pose.position.y = point[1]
                        pose.pose.position.z = 0
                        center_path.poses.append(pose)

                self.center_path_pub.publish(center_path)



                theta = 0

                for i, pose in enumerate(center_path.poses):
                    path_point = pose.pose.position
                    dis = sqrt(path_point.x**2 + path_point.y**2)
                    if dis >= self.lfd:
                        self.forward_point = path_point
                        self.is_look_forward_point = True
                        theta = atan2(path_point.y+sorted_center_line[0][1], path_point.x+sorted_center_line[0][0])  # Correct order of arguments
                        break

                if self.is_look_forward_point:
                    steering = -atan2(2 * self.vehicle_length * sin(theta), self.lfd)
                    #self.ctrl_cmd_msg.steering = self.pid.update(steering*180/pi)
                    self.target_control.drive.steering_angle=steering# + self.pid.compute(distance_to_center_line_based_on_car)
                    print(steering)
                    #print(self.ctrl_cmd_msg.steering)
                    #print(self.ctrl_cmd_msg.steering * 180 / pi)  # Conversion to degrees for readability
                    self.is_look_forward_point = False
                else:
                    self.target_control.drive.steering_angle=0
                
            self.target_control_pub.publish(self.target_control)

if __name__ == '__main__':
    try:
        xycar = Rubber_cone()

    except rospy.ROSInternalException:
        pass 