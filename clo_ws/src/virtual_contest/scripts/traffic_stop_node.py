#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16
from ackermann_msgs.msg import AckermannDriveStamped
from morai_msgs.msg import GetTrafficLightStatus
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
class TrafficStopNode:
    def __init__(self):
        rospy.init_node('traffic_stop_node')
        
        self.stopline_sub = rospy.Subscriber('/stop_line', Int16, self.stopline_callback)
        self.image_sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_callback)
        self.traffic_light_sub = rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.traffic_light_callback)
        self.ack_nav2_sub = rospy.Subscriber('/ackermann_cmd_mux/input/nav_2', AckermannDriveStamped, self.ack_nav2_callback)
        self.ack_pub = rospy.Publisher('/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=1)

        self.bridge = CvBridge()
        self.is_ack_nav2 = False
        self.stopline_detected = False
        self.trafficlight_detected = 0
        self.hsv = None
        self.frame = None
        self.roi = None
        self.red_mask = None
        self.yellow_mask = None
        self.green_mask = None
        self.combined_mask = None
        self.ack_nav2_msg = AckermannDriveStamped()
    def stopline_callback(self, msg):
        self.stopline_detected = msg.data
        #print(f"Stop line detected: {self.stopline_detected}")
    def traffic_light_callback(self, msg):
        self.trafficlight_detected = msg.trafficLightStatus
        #print(f"Traffic light status: {self.trafficlight_detected}")

        

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        
        height, width = self.frame.shape[:2]
        roi_top = int(height / 2) - 20
        roi_bottom = height - 60
        
        self.roi = self.frame[:roi_top, 200:441]

        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        # print(self.detect_traffic_light(hsv))
        if (self.detect_traffic_light(self.hsv)):
            print("Traffic light detected")
            if self.is_ack_nav2:
                ack_msg = AckermannDriveStamped()
                ack_msg.drive.steering_angle = self.ack_nav2_msg.drive.steering_angle
                if self.trafficlight_detected <= 5:
                    ack_msg.drive.speed = 0.3
                    if self.stopline_detected:
                        ack_msg.drive.speed = 0.0
                        #print("Stop line detected, stopping vehicle.")
                    self.ack_pub.publish(ack_msg)
                else:
                    ack_msg.drive.speed = self.ack_nav2_msg.drive.speed
                    #print("Traffic light is green, proceeding.")    
                    self.ack_pub.publish(ack_msg)
        else:
            print("No traffic light detected")
            
    def ack_nav2_callback(self, msg):
        #if self.trafficlight_detected <99:
        if self.is_ack_nav2 == False:
            self.is_ack_nav2 = True
        self.ack_nav2_msg = msg
    
    def detect_traffic_light(self, hsv_img):
        is_traffic_light = False
        # 원본 복사본 생성
        #output_frame = self.frame.copy()

        R_lower = np.array([0,180,40])
        R_upper = np.array([90,255,255])
        Y_lower = np.array([20,89,100])
        Y_upper = np.array([30,255,255])
        G_lower = np.array([40,100,100])
        G_upper = np.array([80,255,255])
        self.red_mask = cv2.inRange(hsv_img, R_lower, R_upper)
        
        

        self.yellow_mask = cv2.inRange(hsv_img, Y_lower, Y_upper)


        self.green_mask = cv2.inRange(hsv_img, G_lower, G_upper)


        self.combined_mask = cv2.bitwise_or(self.red_mask, self.yellow_mask)
        self.combined_mask = cv2.bitwise_or(self.combined_mask, self.green_mask)

         # HSV 빛 중심점 찾기
        
        contours, _ = cv2.findContours(self.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            if area > 5:
                is_traffic_light = True
            
        return is_traffic_light
    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            if self.frame is not None:
                # cv2.imshow("Combined", self.combined_mask)
                #cv2.imshow("ROI", self.roi)
                
                cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    node = TrafficStopNode()
    node.run()