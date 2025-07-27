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
cv2.createTrackbar('Lower Thresh', 'thresh', 80, 255, nothing) 
cv2.createTrackbar('Upper Thresh', 'thresh', 255, 255, nothing) 

cv2.namedWindow('HSV')
cv2.createTrackbar('Lower H', 'HSV', 0, 180, nothing) 
cv2.createTrackbar('Lower S', 'HSV', 50, 255, nothing) 
cv2.createTrackbar('Lower V', 'HSV', 60, 255, nothing) 
cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
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


class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Parking Receiver Object is Created")
        self.bridge = CvBridge()
        rospy.Subscriber("/usb_cam/image_raw/calib", Image, self.callback)
        rospy.Subscriber("/parking_flag", Bool, self.flag_callback)
        self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=1)
        self.drive_info = AckermannDriveStamped()
        self.flag = False
        self.crosswalk_flag = False
    
    def flag_callback(self, data):
        self.crosswalk_flag = data.data

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        warp_img, M, Minv = warp_image(image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        hsv_img, threshold, result = image_processing(warp_img)
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  
            cv2.rectangle(warp_img, (x + 100, y), (x + w + 100, y + h), (0, 255, 0), 4) 
            
            #if w > h and 60 < w < 160 and not self.flag and self.crosswalk_flag:
            if w > h and 30 < w and h < 50 and not self.flag and self.crosswalk_flag: # 50 < w < 160
                cv2.rectangle(warp_img, (x + 100, y), (x + w + 100, y + h), (0, 0, 255), 2) 
                rospy.loginfo("START Detected")
                self.flag = True

                #전진 1.8 
                start_time = time.time()
                while time.time() - start_time < 0.6: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                start_time = time.time()
                while time.time() - start_time < 0.2: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = -0.25
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                start_time = time.time()
                while time.time() - start_time < 0.8: 
                    self.drive_info.drive.speed = 0.3
                    self.drive_info.drive.steering_angle = -0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                self.drive_info.drive.speed = 0.0
                self.drive_info.drive.steering_angle = 0.0
                self.drive_pub.publish(self.drive_info)
                time.sleep(0.05)
                
                #후진 
                start_time = time.time()
                while time.time() - start_time < 0.3: 
                    self.drive_info.drive.speed = -0.2
                    self.drive_info.drive.steering_angle = 0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                start_time = time.time()
                while time.time() - start_time < 1.2: 
                    self.drive_info.drive.speed = -0.2
                    self.drive_info.drive.steering_angle = 0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                start_time = time.time()
                while time.time() - start_time < 2.2: 
                    self.drive_info.drive.speed = -0.2
                    self.drive_info.drive.steering_angle = -0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)

                start_time = time.time()
                while time.time() - start_time < 0.2: 
                    self.drive_info.drive.speed = -0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

                self.drive_info.drive.speed = 0.0
                self.drive_info.drive.steering_angle = 0.0
                self.drive_pub.publish(self.drive_info)
                time.sleep(0.05)

                #전진
                start_time = time.time()
                while time.time() - start_time < 1.0: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.25
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)

                start_time = time.time()
                while time.time() - start_time < 0.3: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)
                    
                # 주차완료 5초
                start_time = time.time()
                while time.time() - start_time < 5.5: 
                    self.drive_info.drive.speed = 0.0
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 
                
                start_time = time.time()
                while time.time() - start_time < 1.2:
                    self.drive_info.drive.speed = -0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)  

                start_time = time.time()
                while time.time() - start_time < 2.2: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = -0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)  

                start_time = time.time()
                while time.time() - start_time < 0.8: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)

                start_time = time.time()
                while time.time() - start_time < 1.8: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.45
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)  

                start_time = time.time()
                while time.time() - start_time < 1.5: 
                    self.drive_info.drive.speed = 0.2
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05)  

                while True:
                    self.drive_info.drive.speed = 0.0 # 정지
                    self.drive_info.drive.steering_angle = 0.0
                    self.drive_pub.publish(self.drive_info)
                    time.sleep(0.05) 

        #cv2.imshow("warp", warp_img)
        #cv2.imshow("HSV", hsv_img)
        #cv2.imshow("thresh", threshold)
        #cv2.imshow("result_PARK", result)
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
    rospy.init_node("parking_rect_pub")
    cam = CameraReceiver()
    rospy.spin()


if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass