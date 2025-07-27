#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32

def nothing(x):
    pass

cv2.namedWindow('thresh')
cv2.createTrackbar('Lower Thresh', 'thresh', 170, 255, nothing) 
cv2.createTrackbar('Upper Thresh', 'thresh', 255, 255, nothing) 

class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Curv Receiver Object is Created")
        self.bridge = CvBridge()
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        #self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_2", AckermannDriveStamped, queue_size=1)
        #self.drive_info = AckermannDriveStamped()
        self.flag_pub = rospy.Publisher("/direction_flag", Int32, queue_size=1)
        self.flag = Int32()

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        roi_img = image[280:, :]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
        low_thresh = cv2.getTrackbarPos('Lower Thresh', 'thresh')
        high_thresh = cv2.getTrackbarPos('Upper Thresh', 'thresh')
        _, threshold = cv2.threshold(blurred, low_thresh, high_thresh, cv2.THRESH_BINARY)  
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 5:  
                ellipse = cv2.fitEllipse(contour)
                (x, y), (major_axis, minor_axis), angle = ellipse

                if 90 < minor_axis < 250 and major_axis > 30 and 240 < x < 340 and 10 < y < 60:
                    cv2.ellipse(roi_img, ellipse, (0, 255, 0), 2)  
                    #self.drive_info.drive.speed = 0.0
                    #self.drive_info.drive.steering_angle = 0.0
                    #self.drive_pub.publish(self.drive_info)
                    self.flag.data = 3
                    self.flag_pub(self.flag)

        cv2.imshow('Ellipses', roi_img)  
        cv2.imshow("thresh", threshold)
        cv2.waitKey(1)


def run():
    rospy.init_node("curv_pub")
    cam = CameraReceiver()
    rospy.spin()


if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass