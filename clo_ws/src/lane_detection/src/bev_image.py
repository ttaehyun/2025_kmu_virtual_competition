#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

# image shape
Width = 640
Height = 480

# warp shape
warp_img_w = 320
warp_img_h = 240

# warp parameter
x_h = 120
x_l = 550
y_h = 50
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

# # Canny
# # cv2.namedWindow('Canny')

# cv2.createTrackbar('Lower Thresh', 'Canny', 50, 255, nothing) # 50 # 0 
# cv2.createTrackbar('Upper Thresh', 'Canny', 100, 255, nothing) # 100 # 150

# cv2.namedWindow('lane')
# cv2.namedWindow('sum')
# cv2.namedWindow('tracker')
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

def image_processing_canny(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    processed_img = np.dstack((img, img, img)) * 255

    return leftx_base, rightx_base, processed_img


def draw_lane(image, warp_img, Minv, left_fit, right_fit, avex, avey):
    global Width, Height

    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros((warp_img.shape[0], warp_img.shape[1], 3), dtype=np.uint8)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.circle(color_warp, (int(avex), int(avey)), 10, (0, 0, 255), -1)

    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))
    lane_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return lane_img


class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Camera Receiver Object is Created")
        self.bridge = CvBridge()

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.bevImage_pub = rospy.Publisher("/BEV_image", CompressedImage, queue_size=10)

        self.parking_flag = False
        self.latest_image = None
        self.hsv_image = None
        self.canny_image = None
        self.sum_image = None
        self.lane_image = None
        self.tracker_image = None
        self.warp_image = None
        self.hsv = None
    def callback(self, data):
        global speed, lane_flag, prev_lane
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            #rospy.loginfo("Image received: shape = %s", self.latest_image.shape)
        except Exception as e:
            rospy.logerr(f"[Image Conversion Error] {e}")
        #blur = cv2.GaussianBlur(self.latest_image, (5, 5), 0)
        

        # HSV ########################
        self.hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
    
        # l_h = cv2.getTrackbarPos('Lower H', 'HSV')
        # l_s = cv2.getTrackbarPos('Lower S', 'HSV')
        # l_v = cv2.getTrackbarPos('Lower V', 'HSV')
        # u_h = cv2.getTrackbarPos('Upper H', 'HSV')
        # u_s = cv2.getTrackbarPos('Upper S', 'HSV')
        # u_v = cv2.getTrackbarPos('Upper V', 'HSV')

        l_h = 0
        l_s = 0
        l_v = 131
        u_h = 40
        u_s = 171
        u_v = 255

        lower_white = np.array([l_h, l_s, l_v])
        upper_white = np.array([u_h, u_s, u_v])
        self.hsv_image = cv2.inRange(self.hsv, lower_white, upper_white)
        ###############################
        
        # No Canny
        #warp_img, M, Minv = warp_image(self.image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        #leftx_base, rightx_base, hsv_img, processed_img = image_processing(warp_img)

        # Canny
        self.warp_image, M, Minv = warp_image(self.latest_image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        # OpenCV → ROS Image 메시지
        bevImage_msg = self.bridge.cv2_to_compressed_imgmsg(self.warp_image, dst_format='jpeg')  # 또는 "mono8" 등

        # 시간 정보 (optional)
        bevImage_msg.header.stamp = rospy.Time.now()
        bevImage_msg.header.frame_id = "bev"

        # Publish
        self.bevImage_pub.publish(bevImage_msg)

        #left_fit, right_fit, avex, avey, self.tracker_image = sliding_window(leftx_base, rightx_base, processed_img, lane_flag)
        #self.lane_image = draw_lane(self.latest_image, self.warp_image, Minv, left_fit, right_fit, avex, avey)

        # cv2.circle(self.lane_image,(x_h, Height//2 + y_h), 5, (0, 255, 255), -1)
        # cv2.circle(self.lane_image,(x_l, Height - y_l), 5, (0, 255, 255), -1)
        # cv2.circle(self.lane_image,(Width - x_h, Height//2 + y_h), 5, (0, 255, 255), -1)
        # cv2.circle(self.lane_image,(Width - x_l, Height - y_l), 5, (0, 255, 255), -1)

    

def run():
    rospy.init_node("BEV_Image")
    cam = CameraReceiver()

    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        if cam.latest_image is not None:
            #cv2.imshow("HSV", cam.hsv_image)
            #cv2.imshow("Image", cam.latest_image)
            #cv2.imshow("lane", cam.lane_image)
            #cv2.imshow("sum", cam.sum_image)
            #cv2.imshow("tracker", cam.tracker_image)
            cv2.imshow("warp", cam.warp_image)
            cv2.waitKey(1)
        rate.sleep()

if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass