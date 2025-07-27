#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32, Bool
import time

# image shape
Width = 640
Height = 480

# warp shape
warp_img_w = 320
warp_img_h = 240

# warp parameter
x_h = 70
x_l = 550
y_h = 70
y_l = 40

temp1 = 40
temp2 = 80

def nothing(x):
    pass

cv2.namedWindow('thresh')
cv2.createTrackbar('Lower Thresh', 'thresh', 65, 255, nothing) 
cv2.createTrackbar('Upper Thresh', 'thresh', 255, 255, nothing) 

cv2.namedWindow('HSV')
cv2.createTrackbar('Lower H', 'HSV', 0, 180, nothing) 
cv2.createTrackbar('Lower S', 'HSV', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'HSV', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing) 
cv2.createTrackbar('Upper S', 'HSV', 140, 255, nothing) 
cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing) 

warp_src = np.array([
    [x_h, Height//2 + y_h], # 좌상단
    [-x_l, Height - y_l], # 좌하단
    [Width - x_h, Height//2 + y_h], # 우상단
    [Width + x_l, Height - y_l] # 우하단
], dtype=np.float32)

warp_dst = np.array([
    [0, 0],
    [0, warp_img_h],
    [warp_img_w, 0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)

import threading 

class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Crosswalk Receiver Object is Created")
        self.bridge = CvBridge()
        rospy.Subscriber("/usb_cam/image_raw/calib", Image, self.callback)
        rospy.Subscriber("/direction_flag", Int32, self.flag_callback)
        self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_4", AckermannDriveStamped, queue_size=1)
        self.parking_pub = rospy.Publisher("/parking_flag", Bool, queue_size=1)
        self.drive_info = AckermannDriveStamped()
        self.parking_msg = Bool()
        self.stop_publishing = False 
        self.lane_flag = -1
        
    def flag_callback(self, data):
        self.lane_flag = data.data

    def reset_stop_publishing(self):
        self.stop_publishing = False

    def curv_timer(self):
        rate = rospy.Rate(30)  
        
        while True:
            rospy.loginfo("CURV STOP")
            self.drive_info.drive.speed = 0.0
            self.drive_info.drive.steering_angle = 0.0
            self.drive_pub.publish(self.drive_info)

            if self.lane_flag != -1 and self.lane_flag != 3:
                rospy.loginfo("CURV START")
                break

            rate.sleep()  

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        warp_img, M, Minv = warp_image(image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        hsv_img, threshold, result = image_processing(warp_img)
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  
            
            if (h > w and w > 5) or (w > h and w > 80):
                counter += 1
                cv2.rectangle(warp_img, (x, y), (x + w, y + h), (0, 255, 0), 4) 

                total_pixels = result.size
                white_pixels = np.count_nonzero(result == 255)
                black_pixels = np.count_nonzero(result == 0)
                white_ratio = white_pixels / total_pixels * 100
              
                if counter >= 4 and not self.stop_publishing and white_ratio > 10:
                    rospy.loginfo("Crosswalk Detected")
                    cv2.rectangle(warp_img, (x, y), (x + w, y + h), (0, 0, 255), 2) 
                   
                    start_time = time.time()
                    while time.time() - start_time < 5:
                        self.drive_info.drive.speed = 0.0
                        self.drive_info.drive.steering_angle = 0.0
                        self.drive_pub.publish(self.drive_info)
                        time.sleep(0.05) 
                    
                    self.stop_publishing = True
                    rospy.loginfo("Crosswalk END")
                    
                    #self.parking_msg.data = True
                    #self.parking_pub.publish(self.parking_msg)
                   
                    threading.Timer(6.5, self.curv_timer).start() # 3초 후 교차로에서 정지
                    #threading.Timer(15, self.reset_stop_publishing).start() # 15초 후 코드 재실행 가능
                    
        #cv2.imshow("warp", warp_img)
        #cv2.imshow("HSV", hsv_img)
        #cv2.imshow("thresh", threshold)
        #cv2.imshow("result_CROSS", result)
        #cv2.waitKey(1)


def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    warp_img = warp_img[temp2:, temp1:warp_img_w - temp1]

    return warp_img, M, Minv


def image_processing(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos('Lower H', 'HSV')
    l_s = cv2.getTrackbarPos('Lower S', 'HSV')
    l_v = cv2.getTrackbarPos('Lower V', 'HSV')
    u_h = cv2.getTrackbarPos('Upper H', 'HSV')
    u_s = cv2.getTrackbarPos('Upper S', 'HSV')
    u_v = cv2.getTrackbarPos('Upper V', 'HSV')

    lower_white = np.array([l_h, l_s, l_v])
    upper_white = np.array([u_h, u_s, u_v])
    hsv_img = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    low_thresh = cv2.getTrackbarPos('Lower Thresh', 'thresh')
    high_thresh = cv2.getTrackbarPos('Upper Thresh', 'thresh')
    _, threshold = cv2.threshold(blurred, low_thresh, high_thresh, cv2.THRESH_BINARY) 

    result = cv2.bitwise_and(hsv_img, threshold)

    return hsv_img, threshold, result


def run():
    rospy.init_node("crosswalk_pub")
    cam = CameraReceiver()
    rospy.spin()


if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass