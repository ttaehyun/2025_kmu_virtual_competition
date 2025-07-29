#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32, Bool, Float64

import threading
import time

# image shape
Width = 640
Height = 480

# warp shape
warp_img_w = 320
warp_img_h = 240

# # warp parameter
# x_h = 70
# x_l = 550
# y_h = 70
# y_l = 40

# warp parameter
# x_h = 120
# x_l = 550
# y_h = 50
# y_l = 40

x_h = 80
x_l = 600
y_h = 30
y_l = 40

# sliding window parameter
nwindows = 20
margin = 20
minpix = 15 # 수정
lane_width = 90

# 0:left, 1:right, 2:both
lane_flag = 1

angle = 0.0
max_angle = 1.0

speed = 1.5
max_speed = 0.4
min_speed = 0.4

is_lane = True
prev_lane = 1

def nothing(x):
    pass

cv2.namedWindow('HSV')

cv2.namedWindow('Canny')


# HSV
cv2.createTrackbar('Lower H', 'HSV', 0, 180, nothing)
cv2.createTrackbar('Lower S', 'HSV', 29, 255, nothing) 
cv2.createTrackbar('Lower V', 'HSV', 215, 255, nothing) 
cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing)

# Canny
cv2.createTrackbar('Lower Thresh', 'Canny', 50, 255, nothing) # 50 # 0 
cv2.createTrackbar('Upper Thresh', 'Canny', 100, 255, nothing) # 100 # 150

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



class CameraReceiver:
    def __init__(self):
        rospy.loginfo("Camera Receiver Object is Created")
        self.bridge = CvBridge()

        self.hsv_img = None
        self.tracker_img = None
        self.canny_img = None
        self.sum_img = None
        self.warp_img = None
        self.warp_color = None
        self.image_with_warp = None
        self.image = None
        # ✅ MORAI의 compressed 이미지 토픽 구독
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)


        # 차량이 ackermann 퍼블리셔를 지원하지 않음 
        # self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_6", AckermannDriveStamped, queue_size=1)
        # self.drive_info = AckermannDriveStamped()

        # ✅ Ackermann 메시지 퍼블리셔 추가
        self.ackermann_pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)

    def callback(self, data):
        global speed, lane_flag, prev_lane

        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image received: shape = %s", self.image.shape)
        except Exception as e:
            rospy.logerr(f"[Image Conversion Error] {e}")
        blur = cv2.GaussianBlur(self.image, (5, 5), 0)

        # HSV ########################
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

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
        self.hsv_img = cv2.inRange(hsv, lower_white, upper_white)
        ###############################

        # Canny ########################
        low_canny_thresh = cv2.getTrackbarPos('Lower Thresh', 'Canny')
        high_canny_thresh = cv2.getTrackbarPos('Upper Thresh', 'Canny')
        self.canny_img = cv2.Canny(blur, low_canny_thresh, high_canny_thresh)
        ###############################

        sum_img = cv2.bitwise_or(self.canny_img, self.hsv_img)
        
        # Canny
        self.warp_img, M, Minv = warp_image(sum_img, warp_src, warp_dst, (warp_img_w, warp_img_h))
        leftx_base, rightx_base, processed_img = image_processing_canny(self.warp_img)

        left_fit, right_fit, avex, avey, self.tracker_img, rx, ry = sliding_window(leftx_base, rightx_base, processed_img, lane_flag, self.warp_img)
        #lane_img = draw_lane(self.image, warp_img, Minv, left_fit, right_fit, avex, avey)

        x = avex - 150
        y = 270 - avey
        angle = math.atan2(x, y)


        if lane_flag == 4 or lane_flag == 5:
            speed = min_speed

            rospy.loginfo("CURV")
            threading.Timer(4.0, self.restore_speed, args=(lane_flag,)).start()

        if is_lane == True:
            # ✅ Ackermann 메시지 구성 및 퍼블리시
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = angle
            self.ackermann_pub.publish(drive_msg)

        #else:
            #rospy.loginfo("No Lane Detected")

        self.image_with_warp = self.image.copy()

        # 점 찍기
        for pt in warp_src:
            cv2.circle(self.image_with_warp, tuple(pt.astype(int)), 5, (0, 255, 255), -1)

        # 선 그리기
        for i in range(4):
            pt1 = tuple(warp_src[i].astype(int))
            pt2 = tuple(warp_src[(i + 1) % 4].astype(int))
            cv2.line(self.image_with_warp, pt1, pt2, (255, 255, 0), 2)

        # ✅ Bird's Eye View (컬러 원본 이미지로부터)
        self.warp_color, _, _ = warp_image(self.image, warp_src, warp_dst, (warp_img_w, warp_img_h))

        
        
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

def sliding_window(leftx_base, rightx_base, processed_img, lane_flag, warp_img):
    global nwindows, margin, minpix, lane_width, warp_img_w, warp_img_h, is_lane

    window_height = int(processed_img.shape[0] // nwindows)

    nonzero = processed_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = [] 
    right_lane_inds = []
    
    lx, ly, rx, ry = [], [], [], [] 

    lefty = 0
    righty = 0

    right_direction = 0  # ✅ 초기 방향 0으로 설정
    left_direction = 0
    
    for window in range(nwindows):

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin
        win_yl = processed_img.shape[0] - (window + 1) * window_height 
        win_yh = processed_img.shape[0] - window * window_height 
        y_center = (win_yl + win_yh) / 2

        # 윈도우 좌표 계산은 생략 (그대로 유지)

        good_left_inds = ((nonzeroy >= win_yl) & (nonzeroy < win_yh) & 
                        (nonzerox >= win_xll) & (nonzerox < win_xlh)).nonzero()[0] 
        good_right_inds = ((nonzeroy >= win_yl) & (nonzeroy < win_yh) & 
                        (nonzerox >= win_xrl) & (nonzerox < win_xrh)).nonzero()[0] 

        left_lane_inds.append(good_left_inds) 
        right_lane_inds.append(good_right_inds)

        # 📌 LEFT
        if len(good_left_inds) > minpix:
            new_x = int(np.mean(nonzerox[good_left_inds]))
            new_y = int(np.mean(nonzeroy[good_left_inds]))

            dx = new_x - leftx_current
            leftx_current = new_x
            left_direction = dx

        else:
            if len(ly) > 5:
                left_fit = np.polyfit(ly, lx, 2)
                leftx_current = int(left_fit[0]*y_center**2 + left_fit[1]*y_center + left_fit[2])
            else:
                leftx_current += int(left_direction * 0.8)

        # 📌 RIGHT
        if len(good_right_inds) > minpix:
            new_x = int(np.mean(nonzerox[good_right_inds]))
            new_y = int(np.mean(nonzeroy[good_right_inds]))

            dx = new_x - rightx_current
            rightx_current = new_x
            right_direction = dx

        else:
            if len(ry) > 5:
                right_fit = np.polyfit(ry, rx, 2)
                rightx_current = int(right_fit[0]*y_center**2 + right_fit[1]*y_center + right_fit[2])
            else:
                rightx_current += int(right_direction * 0.8)

        lx.append(leftx_current) 
        ly.append(y_center)
        rx.append(rightx_current)
        ry.append(y_center)


    left_lane_inds = np.concatenate(left_lane_inds) 
    right_lane_inds = np.concatenate(right_lane_inds)

    if lane_flag == 0 or lane_flag == 4: # 좌측 차선만 
        lfit = np.polyfit(np.array(ly), np.array(lx), 2) 
        rfit = np.polyfit(np.array(ly), np.array(lx) + lane_width, 2)

        processed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] 
        
        avex = int(np.array(lx).mean() + lane_width//2)
        avey = int(np.array(ly).mean())
        rospy.loginfo("Left Lane Detected")
        
    elif lane_flag == 1 or lane_flag == 5: # 우측 차선만 
        lfit = np.polyfit(np.array(ry), np.array(rx) - lane_width, 2) 
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)

        processed_img[nonzeroy[right_lane_inds] , nonzerox[right_lane_inds]] = [0, 0, 255] 
        
        avex = int(np.array(rx).mean() - lane_width//2)
        avey = int(np.array(ry).mean())

        # rospy.loginfo("Right Lane Detected")

    else: # 양쪽 차선
        lfit = np.polyfit(np.array(ly), np.array(lx), 2) 
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)

        processed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] 
        processed_img[nonzeroy[right_lane_inds] , nonzerox[right_lane_inds]] = [0, 0, 255] 
        
        avex = int(np.array(lx).mean() + np.array(rx).mean())//2
        avey = int(np.array(ly).mean() + np.array(ry).mean())//2
    
    if len(lx) < 5 and len(rx) < 5:
        is_lane = False
        avex = warp_img_w // 2
        avey = warp_img_h // 2
    else:
        is_lane = True
    
    cv2.circle(processed_img,(avex, avey), 5, (0, 255, 255), -1)

    return lfit, rfit, avex, avey, processed_img, rx, ry

def calculate_curvature(poly_fit, y_eval):
    """
    poly_fit: np.polyfit()의 반환값 (차선 곡선 계수 A, B, C)
    y_eval: 곡률을 계산할 y 위치 (보통 이미지 하단)
    return: 곡률 값 (픽셀 단위)
    """
    A = poly_fit[0]
    B = poly_fit[1]
    curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A + 1e-6)
    return curvature


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

def run():
    rospy.init_node("ld_pub")
    cam = CameraReceiver()

    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        if cam.image is not None:
            
            cv2.imshow("HSV", cam.hsv_img)     
            cv2.imshow("tracker", cam.tracker_img)
            cv2.imshow("Canny", cam.canny_img)
            #cv2.imshow("sum_img", cam.sum_img)
            cv2.imshow("Bird's Eye View Color", cam.warp_color)

            cv2.imshow("Original + warp area", cam.image_with_warp)
            cv2.imshow("Bird's Eye View", cam.warp_img)
            cv2.waitKey(1)
        rate.sleep()
if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass