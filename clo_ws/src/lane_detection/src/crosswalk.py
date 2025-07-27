#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Bool
import time

# image shape
Width = 640
Height = 480

temp1 = 40
temp2 = 80


import threading 

class CrossWalk:
    def __init__(self):
        rospy.init_node('Crosswalk', anonymous=True)
        rospy.loginfo("Crosswalk Receiver Object is Created")
        self.bridge = CvBridge()
        rospy.Subscriber("/BEV_image", CompressedImage, self.callback)
        
        self.frame = None
        self.mask = None
        self.roi = None

    def callback(self, msg):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        
        height, width = self.frame.shape[:2]
        roi_top = int(height / 2) - 10
        roi_bottom = height - 60
        self.roi = self.frame[roi_top:, :]
        
        # HSV 변환 및 마스킹
        hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 131])
        upper_white = np.array([40, 171, 255])
        self.mask = cv2.inRange(hsv, lower_white, upper_white)
        
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  
            
            if (h > w and w > 5) or (w > h and w > 80):
                counter += 1
                cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 255, 0), 4) 

                total_pixels = self.mask.size
                white_pixels = np.count_nonzero(self.mask == 255)
                black_pixels = np.count_nonzero(self.mask == 0)
                white_ratio = white_pixels / total_pixels * 100
              
                if counter >= 6  and white_ratio > 10:
                    rospy.loginfo("Crosswalk Detected")
                    cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 0, 255), 2) 


    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Raw Image", self.frame)
                cv2.imshow("ROI", self.roi)
                cv2.imshow("Mask", self.mask)

                cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    viewer = CrossWalk()
    viewer.spin()
    cv2.destroyAllWindows()