#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Int16
class StoplineDetector:
    def __init__(self):
        rospy.init_node('StoplineNode')

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/BEV_image",CompressedImage,self.callback)
        self.stopline_pub = rospy.Publisher('/stop_line', Int16, queue_size=10)
        self.frame = None
        self.roi = None
        self.mask = None
        self.masked = None
        self.gray = None
        self.blur = None
        self.edges = None
        self.count = 0

        self.detected = False
       # cv2.namedWindow("Raw Image", cv2.WINDOW_NORMAL)

    def callback(self, msg):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        
        height, width = self.frame.shape[:2]
        roi_top = int(height / 2) + 45
        roi_bottom = height
        self.roi = self.frame[roi_top:roi_bottom, :]
        
        # HSV 변환 및 마스킹
        hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 131])
        upper_white = np.array([40, 171, 255])
        self.mask = cv2.inRange(hsv, lower_white, upper_white)

        # 마스킹된 이미지 처리
        self.masked = cv2.bitwise_and(self.roi, self.roi, mask=self.mask)
        self.gray = cv2.cvtColor(self.masked, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edges = cv2.Canny(self.blur, 50, 150)

        # contour 기반 정지선 검출
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stopline_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print(area)
            if area < 5000:  # 넓이 기준 필터링 (필요 시 조정)
                continue

            rect = cv2.minAreaRect(cnt)  # 중심, 크기(w, h), 회전각
            (x, y), (w, h), angle = rect
            # if w == 0 or h == 0:
            #     continue

            aspect_ratio = max(w, h) / min(w, h)
            # print(aspect_ratio)
            if aspect_ratio > 5:  # 너무 길쭉한 직사각형은 제외 (차선)
                continue
            # print(angle)
            if abs(angle) > 77 or abs(angle) < 20:
                cv2.drawContours(self.roi, [cnt], -1, (0, 0, 255), -1)
                
                stopline_detected = True

        # 결과 출력
        msg = Int16()
        if stopline_detected:
            msg.data = 1  # 정지선이 감지되었음을 나타내는 값
            self.stopline_pub.publish(msg)
            cv2.putText(self.roi, "Stop Line Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            msg.data = 0
            self.stopline_pub.publish(msg)
            cv2.putText(self.roi, "No Stop Line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if (not self.detected):
            
            if (stopline_detected):
                self.count += 1  
        
        self.detected = stopline_detected
        # ROI 사각형 시각화
        cv2.rectangle(self.frame, (0, roi_top), (width, roi_bottom), (0, 255, 0), 2)

        #print(self.count)
    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Raw Image", self.frame)
                #cv2.imshow("ROI", self.roi)
                #cv2.imshow("Mask", self.mask)
                #cv2.imshow("Masked", self.masked)
                #cv2.imshow("gray", self.gray)
                #cv2.imshow("blur", self.blur)
                #cv2.imshow("edges", self.edges)

                cv2.waitKey(1)
            rate.sleep()

if __name__ == "__main__":
    viewer = StoplineDetector()
    viewer.spin()
    cv2.destroyAllWindows()