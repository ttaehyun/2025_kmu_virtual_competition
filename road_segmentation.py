#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import signal

class RoadSegmentationNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image_jpeg/compressed',
                                          CompressedImage,
                                          self.image_callback,
                                          queue_size=1,
                                          buff_size=2**24)
        self.segmented_pub = rospy.Publisher('/segmented_image', Image, queue_size=1)

        self.running = True
        signal.signal(signal.SIGINT, self.shutdown_handler)
        rospy.loginfo("Road Segmentation Node Initialized.")

    def image_callback(self, msg):
        if not self.running:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            height, width = image.shape[:2]
            roi_y_start = height // 2

            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 흰색 마스크
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            # 노란색 마스크
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 마스크 결합
            mask = cv2.bitwise_or(mask_white, mask_yellow)

            # ROI 하단 영역만 유지
            roi_mask = np.zeros_like(mask)
            roi_mask[roi_y_start:, :] = mask[roi_y_start:, :]

            # 모폴로지 연산
            kernel = np.ones((5, 5), np.uint8)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

            # Segmented 이미지 시각화용
            vis = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

            # HoughLinesP 적용
            edges = cv2.Canny(roi_mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=20)

            # 선 시각화
            detected_lines_img = image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # 각도 계산
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                    # 수평선은 무시 (±15도 범위)
                    if angle < 15 or angle > 165:
                        continue

                    cv2.line(detected_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ROS 퍼블리시
            self.segmented_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))

            # 디스플레이
            cv2.imshow("Original", image)
            cv2.imshow("Segmented", vis)
            cv2.imshow("Detected_Lines", detected_lines_img)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"[RoadSegmentationNode] {e}")

    def shutdown_handler(self, sig, frame):
        rospy.loginfo("Shutting down...")
        self.running = False
        cv2.destroyAllWindows()
        rospy.signal_shutdown("User interrupted")

    def run(self):
        while not rospy.is_shutdown() and self.running:
            rospy.sleep(0.05)

if __name__ == '__main__':
    rospy.init_node('road_segmentation_node', anonymous=True)
    RoadSegmentationNode().run()
