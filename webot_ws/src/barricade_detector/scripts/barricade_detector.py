#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

# LiDAR 기반 Wall 검출 파라미터
ANGLE_OFFSET = np.pi
LIDAR_LEFT_BOUND = -30
LIDAR_RIGHT_BOUND = 30
DISTANCE_THRESHOLD_MIN = 0.1
DISTANCE_THRESHOLD_MAX = 0.4
RANSAC_DISTANCE_THRESHOLD = 0.15
RANSAC_NUM_ITERATIONS = 200
RANSAC_MIN_INLIERS = 30
MIN_LINE_WIDTH = 0.1
MAX_LINE_WIDTH = 0.6

# 터널 인식 기준
TUNNEL_LEFT_INDEX_RANGE = range(224, 425)  # 좌측 라이다 인덱스 범위
TUNNEL_RIGHT_INDEX_RANGE = range(867, 1068)  # 우측 라이다 인덱스 범위
TUNNEL_DISTANCE_MIN = 0.05  # 터널 최소 거리 (m)
TUNNEL_DISTANCE_MAX = 0.5  # 터널 최대 거리 (m)

# LaserScan 데이터를 처리하여 Wall 검출
def process_scan_data(scan_data):
    lidar_points = []
    angle_min = scan_data.angle_min + ANGLE_OFFSET
    angle_increment = scan_data.angle_increment
    ranges_length = len(scan_data.ranges)

    start_index = max(0, int((np.radians(LIDAR_LEFT_BOUND) - angle_min) / angle_increment))
    end_index = min(ranges_length, int((np.radians(LIDAR_RIGHT_BOUND) - angle_min) / angle_increment))

    for index in range(start_index, end_index):
        r = scan_data.ranges[index]
        angle = angle_min + index * angle_increment
        if DISTANCE_THRESHOLD_MIN <= r <= DISTANCE_THRESHOLD_MAX:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            lidar_points.append([x, y])

    return np.array(lidar_points)

# RANSAC을 사용하여 직선형 장애물 감지 함수
def detect_line(lidar_points):
    if len(lidar_points) < 2:
        return False, None

    best_inliers = []
    best_model = None

    for _ in range(RANSAC_NUM_ITERATIONS):
        sample_indices = np.random.choice(len(lidar_points), 2, replace=False)
        p1, p2 = lidar_points[sample_indices]

        if p2[0] - p1[0] == 0:
            continue
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]

        distances = np.abs(slope * lidar_points[:, 0] - lidar_points[:, 1] + intercept) / np.sqrt(slope**2 + 1)
        inliers = lidar_points[distances < RANSAC_DISTANCE_THRESHOLD]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (slope, intercept)

        if len(best_inliers) > RANSAC_MIN_INLIERS:
            break

    if best_model is not None and len(best_inliers) >= RANSAC_MIN_INLIERS:
        inlier_points = np.array(best_inliers)
        width = np.linalg.norm(inlier_points.max(axis=0)[:2] - inlier_points.min(axis=0)[:2])

        if MIN_LINE_WIDTH <= width <= MAX_LINE_WIDTH:
            return True, inlier_points
    return False, None

# 터널 인식 함수
def detect_tunnel(scan_data):
    left_distances = [
        scan_data.ranges[i]
        for i in TUNNEL_LEFT_INDEX_RANGE
        if TUNNEL_DISTANCE_MIN <= scan_data.ranges[i] <= TUNNEL_DISTANCE_MAX
    ]
    right_distances = [
        scan_data.ranges[i]
        for i in TUNNEL_RIGHT_INDEX_RANGE
        if TUNNEL_DISTANCE_MIN <= scan_data.ranges[i] <= TUNNEL_DISTANCE_MAX
    ]

    if len(left_distances) > 5 and len(right_distances) > 5:  # 양쪽 충분한 포인트가 있으면 터널로 인식
        rospy.loginfo("Tunnel detected.")
        return True
    return False

def lidar_callback(scan_data, control_pub):
    if detect_tunnel(scan_data):
        return  # 터널로 인식되면 정지 신호를 발행하지 않음

    lidar_points = process_scan_data(scan_data)
    detected, inlier_points = detect_line(lidar_points)

    ackermann_msg = AckermannDriveStamped()
    if detected:
        rospy.loginfo("barricade detected! Stopping.")
        ackermann_msg.drive.speed = 0.0
        control_pub.publish(ackermann_msg)

def main():
    rospy.init_node("barricade_detector", anonymous=True)
    control_pub = rospy.Publisher("/high_level/ackermann_cmd_mux/input/nav_3", AckermannDriveStamped, queue_size=10)

    rospy.Subscriber("/scan", LaserScan, lambda scan_data: lidar_callback(scan_data, control_pub))

    rospy.loginfo("barricade_detector node started.")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass