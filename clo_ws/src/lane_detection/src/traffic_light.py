#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import GetTrafficLightStatus
from std_msgs.msg import Int16
class TrafficLightDetector:
    def __init__(self):
        rospy.init_node('traffic_light_node')
        self.bridge = CvBridge()
        # 트랙바 UI
        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
        self.create_trackbars()

        self.traffic_pub = rospy.Publisher('/trafficLight', Int16, queue_size=10)

        self.image_sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        self.traffic_sub = rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.traffic_callback)
        self.frame = None
        self.red_mask = None
        self.yellow_mask = None
        self.green_mask = None
        self.roi = None
        self.edges = None
        self.combined_mask = None
        self.trafficLightStatus = None
        self.is_traffic_light = False
    def create_trackbars(self):
        def nothing(x): pass
        # 빨강 범위
        cv2.createTrackbar('R Low H', 'Trackbars', 0, 180, nothing)
        cv2.createTrackbar('R High H', 'Trackbars', 10, 180, nothing)
        # 초록 범위
        cv2.createTrackbar('G Low H', 'Trackbars', 40, 180, nothing)
        cv2.createTrackbar('G High H', 'Trackbars', 80, 180, nothing)
        # 노랑 범위
        cv2.createTrackbar('Y Low H', 'Trackbars', 20, 180, nothing)
        cv2.createTrackbar('Y High H', 'Trackbars', 30, 180, nothing)

        # Canny
        cv2.createTrackbar('Lower Thresh', 'Trackbars', 50, 255, nothing) # 50 # 0 
        cv2.createTrackbar('Upper Thresh', 'Trackbars', 100, 255, nothing) # 100 # 150

        for name in ['R', 'G', 'Y']:
            cv2.createTrackbar(f'{name} Low S', 'Trackbars', 100, 255, nothing)
            cv2.createTrackbar(f'{name} High S', 'Trackbars', 255, 255, nothing)
            cv2.createTrackbar(f'{name} Low V', 'Trackbars', 100, 255, nothing)
            cv2.createTrackbar(f'{name} High V', 'Trackbars', 255, 255, nothing)

    def get_hsv_range(self, color):
        lh = cv2.getTrackbarPos(f'{color} Low H', 'Trackbars')
        hh = cv2.getTrackbarPos(f'{color} High H', 'Trackbars')
        ls = cv2.getTrackbarPos(f'{color} Low S', 'Trackbars')
        hs = cv2.getTrackbarPos(f'{color} High S', 'Trackbars')
        lv = cv2.getTrackbarPos(f'{color} Low V', 'Trackbars')
        hv = cv2.getTrackbarPos(f'{color} High V', 'Trackbars')
        return (np.array([lh, ls, lv]), np.array([hh, hs, hv]))

    def detect_traffic_light(self, hsv_img):
        is_traffic_light = False
        # 원본 복사본 생성
        #output_frame = self.frame.copy()

        R_lower, R_upper = self.get_hsv_range('R')
        Y_lower, Y_upper = self.get_hsv_range('Y')
        G_lower, G_upper = self.get_hsv_range('G')

        self.red_mask = cv2.inRange(hsv_img, R_lower, R_upper)
        
        # contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > 100:  # 최소 크기 필터
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0,0,255), 2)
        

        self.yellow_mask = cv2.inRange(hsv_img, Y_lower, Y_upper)
        # contours, _ = cv2.findContours(self.yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > 100:  # 최소 크기 필터
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0,255,255), 2)

        self.green_mask = cv2.inRange(hsv_img, G_lower, G_upper)
        # contours, _ = cv2.findContours(self.green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > 100:  # 최소 크기 필터
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0,255,0), 2)

        self.combined_mask = cv2.bitwise_or(self.red_mask, self.yellow_mask)
        self.combined_mask = cv2.bitwise_or(self.combined_mask, self.green_mask)

         # HSV 빛 중심점 찾기
        light_centers = []
        contours, _ = cv2.findContours(self.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    light_centers.append((cx, cy))
                    cv2.circle(self.roi, (cx, cy), 3, (255, 255, 255), -1)
        #print(light_centers)
        # Canny ########################
        # low_canny_thresh = cv2.getTrackbarPos('Lower Thresh', 'Trackbars')
        # high_canny_thresh = cv2.getTrackbarPos('Upper Thresh', 'Trackbars')
        low_canny_thresh = 255
        high_canny_thresh = 255
        # 전체 프레임에 대해 Canny 엣지 적용
        gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self.edges = cv2.Canny(blur, low_canny_thresh, high_canny_thresh)

        self.combined_mask = cv2.bitwise_or(self.combined_mask, self.edges)
        # 엣지 기반 contour에서 사각형 찾기
        edge_contours, _ = cv2.findContours(self.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in edge_contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            # for pt in approx:
            #    cx, cy = pt[0]
            #    cv2.circle(self.roi, (cx, cy), 3, (255, 0, 0), -1)
            area = cv2.contourArea(cnt)
            if (4<= len(approx) <=6) and area > 2000 and cv2.isContourConvex(approx):
                #print(area)
                x, y, w, h = cv2.boundingRect(cnt)
                for cx, cy in light_centers:
                    if x < cx < x + w and y < cy < y + h:
                        cv2.rectangle(self.roi, (x, y), (x + w, y + h), (255, 0, 255), 2)
                        cv2.putText(self.roi, "Traffic Light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        is_traffic_light = True
            
        
        return is_traffic_light

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        
        height, width = self.frame.shape[:2]
        roi_top = int(height / 2)
        roi_bottom = height - 60
        self.roi = self.frame[:roi_top, :]

        hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        # print(self.detect_traffic_light(hsv))
        if (self.detect_traffic_light(hsv)):
            traffic_msg = Int16()
            traffic_msg.data = self.trafficLightStatus
            if (self.trafficLightStatus == 16):
                print("직진")
            elif (self.trafficLightStatus == 33):
                print("좌회전")
            elif (self.trafficLightStatus == 1):
                print("정지")
            elif (self.trafficLightStatus == 4):
                print("노란불")
            elif (self.trafficLightStatus == 5):
                print("좌회전 노란불")
        
            self.traffic_pub.publish(traffic_msg)

    def traffic_callback(self, msg):
        self.trafficLightStatus = msg.trafficLightStatus
        #print(self.trafficLightStatus)
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Raw Image", self.roi)
                cv2.imshow("R", self.red_mask)
                cv2.imshow("Y", self.yellow_mask)
                cv2.imshow("G", self.green_mask)
                cv2.imshow("combined", self.combined_mask)
                #cv2.imshow("blur", self.blur)
                #cv2.imshow("edges", self.edges)

                cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    node = TrafficLightDetector()
    node.run()
    cv2.destroyAllWindows()