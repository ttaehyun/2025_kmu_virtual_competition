#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32, Bool, Float64
from geometry_msgs.msg import Twist
import threading
import time

# image shape
Width = 640
Height = 480

# warp shape
warp_img_w = 320
warp_img_h = 240

###################################### 수정한거 #####################################
x_h = 70
x_l = 550
y_h = 70
y_l = 0

# sliding window parameter
nwindows = 20
margin = 20
minpix = 15 # 수정
lane_width = 110

###################################### 수정한거 #####################################
# 0:left, 1:right, 2:both
lane_flag = 1
steering_gain = 1.0
angle = 0.0
max_angle = 1.0
###################################### 수정한거 #####################################
speed = 1.3
max_speed = 0.4
min_speed = 0.4

is_lane = True
prev_lane = 1

def nothing(x):
    pass

# ✅ GUI 창 생성은 메인 스레드에서 미리 수행
cv2.namedWindow('HSV')
cv2.namedWindow('Canny')
cv2.namedWindow('tracker')
cv2.namedWindow('sum_img')
cv2.namedWindow('Original + warp area')
cv2.namedWindow("Bird's Eye View Color")
cv2.namedWindow("Bird's Eye View")

# HSV
cv2.createTrackbar('Lower H', 'HSV', 0, 180, nothing)
cv2.createTrackbar('Lower S', 'HSV', 29, 255, nothing) 
cv2.createTrackbar('Lower V', 'HSV', 215, 255, nothing) 
cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing)

# Canny
cv2.createTrackbar('Lower Thresh', 'Canny', 50, 255, nothing)
cv2.createTrackbar('Upper Thresh', 'Canny', 100, 255, nothing)

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

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)
        rospy.Subscriber("/direction_flag", Int32, self.flag_callback)
        rospy.Subscriber("/parking_flag", Bool, self.parking_callback)
        self.ackermann_pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)

        self.parking_flag = False
        self.start_time = time.time()
        
        # ✅ 메인 루프에서 사용할 이미지들을 저장할 변수
        self.image_to_show = {
            "hsv": None,
            "canny": None,
            "tracker": None,
            "sum": None,
            "warp_area": None,
            "bev_color": None,
            "bev": None
        }
        self.image_ready = False

    def restore_speed(self, flag):
        global speed, max_speed, lane_flag
        speed = max_speed 
        if flag == 4:
            lane_flag = 0
        else:
            lane_flag = 1

    def change_lane(self):
        global lane_flag, lane_width
        rospy.loginfo("AR Change lane")
        lane_flag = 1
        lane_width = 135 

    def parking_callback(self, data):
        self.parking_flag = data.data
        if self.parking_flag:
            threading.Timer(9.0, self.change_lane).start()

    def flag_callback(self, data):
        global lane_flag
        if not self.parking_flag:
            lane_flag = data.data

    def callback(self, data):
        global speed, lane_flag, is_lane
        
        try:
            # 이미지 디코딩 및 전처리 (계산만 수행)
            np_arr = np.frombuffer(data.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if original_image is None:
                rospy.logerr("Failed to decode image")
                return

            blur = cv2.GaussianBlur(original_image, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            # 트랙바 값 읽기
            l_h = cv2.getTrackbarPos('Lower H', 'HSV')
            l_s = cv2.getTrackbarPos('Lower S', 'HSV')
            l_v = cv2.getTrackbarPos('Lower V', 'HSV')
            u_h = cv2.getTrackbarPos('Upper H', 'HSV')
            u_s = cv2.getTrackbarPos('Upper S', 'HSV')
            u_v = cv2.getTrackbarPos('Upper V', 'HSV')
            lower_white = np.array([l_h, l_s, l_v])
            upper_white = np.array([u_h, u_s, u_v])
            hsv_img = cv2.inRange(hsv, lower_white, upper_white)

            low_canny_thresh = cv2.getTrackbarPos('Lower Thresh', 'Canny')
            high_canny_thresh = cv2.getTrackbarPos('Upper Thresh', 'Canny')
            canny_img = cv2.Canny(blur, low_canny_thresh, high_canny_thresh)
            
            sum_img = cv2.bitwise_or(canny_img, hsv_img)
            
            # 차선 인식 로직
            warp_img, M, Minv = warp_image(sum_img, warp_src, warp_dst, (warp_img_w, warp_img_h))
            leftx_base, rightx_base, processed_img = image_processing_canny(warp_img)
            left_fit, right_fit, avex, avey, tracker_img = sliding_window(leftx_base, rightx_base, processed_img, lane_flag)
            
            # 조향값 계산 및 퍼블리시
            x = avex - 160
            y = 240 - avey
            angle = -steering_gain*math.atan2(x, y)

            if lane_flag == 4 or lane_flag == 5:
                speed = min_speed
                rospy.loginfo("CURV")
                threading.Timer(4.0, self.restore_speed, args=(lane_flag,)).start()

            if is_lane:
                drive_msg = AckermannDriveStamped()
                drive_msg.header.stamp = rospy.Time.now()
                drive_msg.drive.speed = speed
                drive_msg.drive.steering_angle = angle
                self.ackermann_pub.publish(drive_msg)

            # ✅ 시각화를 위한 이미지들 생성
            image_with_warp = original_image.copy()
            for pt in warp_src:
                cv2.circle(image_with_warp, tuple(pt.astype(int)), 5, (0, 255, 255), -1)
            for i in range(4):
                pt1 = tuple(warp_src[i].astype(int))
                pt2 = tuple(warp_src[(i + 1) % 4].astype(int))
                cv2.line(image_with_warp, pt1, pt2, (255, 255, 0), 2)
            warp_color, _, _ = warp_image(original_image, warp_src, warp_dst, (warp_img_w, warp_img_h))

            # ✅ 계산된 최종 이미지들을 클래스 변수에 저장
            self.image_to_show["hsv"] = hsv_img
            self.image_to_show["canny"] = canny_img
            self.image_to_show["tracker"] = tracker_img
            self.image_to_show["sum"] = sum_img
            self.image_to_show["warp_area"] = image_with_warp
            self.image_to_show["bev_color"] = warp_color
            self.image_to_show["bev"] = warp_img
            
            self.image_ready = True

        except Exception as e:
            rospy.logerr(f"An error occurred in the callback function: {e}")

# 아래 함수들은 로직 수정 없이 그대로 사용
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

def sliding_window(leftx_base, rightx_base, processed_img, lane_flag):
    global nwindows, margin, minpix, lane_width, warp_img_w, warp_img_h, is_lane
    window_height = int(processed_img.shape[0] // nwindows)
    nonzero = processed_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    lx, ly, rx, ry = [], [], [], []
    lefty, righty = 0, 0
    
    for window in range(nwindows): 
        win_yl = processed_img.shape[0] - (window + 1) * window_height
        win_yh = processed_img.shape[0] - window * window_height
        win_xll, win_xlh = leftx_current - margin, leftx_current + margin
        win_xrl, win_xrh = rightx_current - margin, rightx_current + margin

        if lane_flag == 0 or lane_flag == 4:
            cv2.rectangle(processed_img, (win_xll, win_yl), (win_xlh,win_yh), (0,255,0), 2)
        elif lane_flag == 1 or lane_flag == 5:
            cv2.rectangle(processed_img, (win_xrl, win_yl), (win_xrh,win_yh), (0,255,0), 2)
        else:
            cv2.rectangle(processed_img, (win_xll, win_yl), (win_xlh,win_yh), (0,255,0), 2)
            cv2.rectangle(processed_img, (win_xrl, win_yl), (win_xrh,win_yh), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_yl) & (nonzeroy < win_yh) & (nonzerox >= win_xll) & (nonzerox < win_xlh)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_yl) & (nonzeroy < win_yh) & (nonzerox >= win_xrl) & (nonzerox < win_xrh)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            lefty = int(np.mean(nonzeroy[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            righty = int(np.mean(nonzeroy[good_right_inds]))
        
        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    if lane_flag == 0 or lane_flag == 4:
        lfit = np.polyfit(np.array(ly), np.array(lx), 2)
        rfit = np.polyfit(np.array(ly), np.array(lx) + lane_width, 2)
        if len(left_lane_inds) > 0: processed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        avex = int(np.array(lx).mean() + lane_width//2)
        avey = int(np.array(ly).mean())
    elif lane_flag == 1 or lane_flag == 5:
        lfit = np.polyfit(np.array(ry), np.array(rx) - lane_width, 2)
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)
        if len(right_lane_inds) > 0: processed_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        avex = int(np.array(rx).mean() - lane_width//2)
        avey = int(np.array(ry).mean())
    else:
        lfit = np.polyfit(np.array(ly), np.array(lx), 2)
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)
        if len(left_lane_inds) > 0: processed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        if len(right_lane_inds) > 0: processed_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        avex = int((np.array(lx).mean() + np.array(rx).mean()) / 2)
        avey = int((np.array(ly).mean() + np.array(ry).mean()) / 2)

    if lefty == 0 and righty == 0:
        is_lane = False
        avex, avey = warp_img_w//2, warp_img_h//2
    else:
        is_lane = True
    
    cv2.circle(processed_img,(avex, avey), 5, (0, 255, 255), -1)
    return lfit, rfit, avex, avey, processed_img

def run():
    rospy.init_node("ld_pub")
    cam = CameraReceiver()
    
    rate = rospy.Rate(30) # 30Hz, GUI 업데이트 주기
    
    # ✅ rospy.spin() 대신 메인 루프 사용
    while not rospy.is_shutdown():
        # ✅ 콜백 함수가 이미지를 준비했는지 확인
        if cam.image_ready:
            # ✅ 메인 스레드에서만 imshow 호출
            cv2.imshow('HSV', cam.image_to_show["hsv"])
            cv2.imshow('Canny', cam.image_to_show["canny"])
            cv2.imshow('tracker', cam.image_to_show["tracker"])
            cv2.imshow('sum_img', cam.image_to_show["sum"])
            cv2.imshow('Original + warp area', cam.image_to_show["warp_area"])
            cv2.imshow("Bird's Eye View Color", cam.image_to_show["bev_color"])
            cv2.imshow("Bird's Eye View", cam.image_to_show["bev"])
        
        # ✅ GUI 이벤트를 처리하는 waitKey는 여기서 단 한 번만 호출
        key = cv2.waitKey(1)
        if key == 27: # ESC 키를 누르면 종료
            break
            
        rate.sleep()
    
    # ✅ 모든 창 닫기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass