#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
# image shape
Width = 640
Height = 480

# warp shape
warp_img_w = 320
warp_img_h = 240

# # warp parameter
# x_h = 120
# x_l = 550
# y_h = 50
# y_l = 40

x_h = 80
x_l = 600
y_h = 30
y_l = 40
def nothing(x):
    pass

# # HSV
# cv2.createTrackbar('Lower H', 'HSV', 20, 180, nothing)
# cv2.createTrackbar('Lower S', 'HSV', 215, 255, nothing) 
# cv2.createTrackbar('Lower V', 'HSV', 0, 255, nothing) 
# cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing)
# cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
# cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing)

# 원근법을 수직 차선 투영법으로 변환
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

def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    

    return warp_img, M, Minv

class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Camera Receiver Object is Created")
        self.bridge = CvBridge()

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.drive_pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)
        self.bevImage_pub = rospy.Publisher("/BEV_image", CompressedImage, queue_size=10)

        self.latest_image = None
        self.hsv_image = None

        self.sum_image = None
        self.lane_image = None
        self.tracker_image = None
        self.warp_image = None
        self.hsv = None

        self.drive_msg = AckermannDriveStamped()
    def callback(self, data):
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            #rospy.loginfo("Image received: shape = %s", self.latest_image.shape)
        except Exception as e:
            rospy.logerr(f"[Image Conversion Error] {e}")
        # bev image 전달
        self.warp_image, M, Minv = warp_image(self.latest_image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        # OpenCV → ROS Image 메시지
        bevImage_msg = self.bridge.cv2_to_compressed_imgmsg(self.warp_image, dst_format='jpeg')  # 또는 "mono8" 등

        # 시간 정보 (optional)
        bevImage_msg.header.stamp = rospy.Time.now()
        bevImage_msg.header.frame_id = "bev"

        # Publish
        self.bevImage_pub.publish(bevImage_msg)
 
def run():
    rospy.init_node("BEV_Image")
    cam = CameraReceiver()

    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        if cam.latest_image is not None:
            #cv2.imshow("HSV", cam.hsv_image)
            #cv2.imshow("Image", cam.latest_image)

            cv2.imshow("warp", cam.warp_image)
            cv2.waitKey(1)
        rate.sleep()

if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass