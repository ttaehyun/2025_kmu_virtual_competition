#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Bool
import threading
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
y_l = 0

# sliding window parameter
nwindows = 20
margin = 20
minpix = 15 # 수정
lane_width = 110

# 0:left, 1:right, 2:both
lane_flag = 1

angle = 0.0
max_angle = 1.0

speed = 0.4
max_speed = 0.4
min_speed = 0.4

is_lane = True
prev_lane = 1

def nothing(x):
    pass

cv2.namedWindow('HSV')
# cv2.namedWindow('Thresh')
# cv2.namedWindow('Canny')
#cv2.namedWindow('Yellow')

# HSV
cv2.createTrackbar('Lower H', 'HSV', 20, 180, nothing)
cv2.createTrackbar('Lower S', 'HSV', 215, 255, nothing) 
cv2.createTrackbar('Lower V', 'HSV', 0, 255, nothing) 
cv2.createTrackbar('Upper H', 'HSV', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing)


# Threshlod
# cv2.createTrackbar('Lower Thresh', 'Thresh', 0, 255, nothing) 
# cv2.createTrackbar('Upper Thresh', 'Thresh', 0, 255, nothing) 


# Canny
# cv2.createTrackbar('Lower Thresh', 'Canny', 50, 255, nothing) # 50 # 0 
# cv2.createTrackbar('Upper Thresh', 'Canny', 100, 255, nothing) # 100 # 150

'''
# Yellow
cv2.createTrackbar('Lower B', 'Yellow', 0, 255, nothing) 
cv2.createTrackbar('Lower G', 'Yellow', 200, 255, nothing)
cv2.createTrackbar('Lower R', 'Yellow', 200, 255, nothing)
cv2.createTrackbar('Upper B', 'Yellow', 50, 255, nothing)
cv2.createTrackbar('Upper G', 'Yellow', 255, 255, nothing)
cv2.createTrackbar('Upper R', 'Yellow', 255, 255, nothing)
'''

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
        # self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_6", AckermannDriveStamped, queue_size=1)
        # self.drive_info = AckermannDriveStamped()
        self.parking_flag = False
        self.start_time = time.time()
        self.latest_image = None
    '''
    def restore_speed_stop(self):
        global speed, max_speed, lane_flag, prev_lane
        speed = max_speed
        
        if prev_lane == 1 or prev_lane == 5:
            lane_flag = 1
        
        if prev_lane == 0 or prev_lane == 4:
            lane_flag = 0
        #print("Run")
    ''' 

    def restore_speed(self, flag):
        global speed, max_speed, lane_flag
        speed = max_speed 

        if flag == 4:
            lane_flag = 0
        else:
            lane_flag = 1

    def callback(self, data):
        global speed, lane_flag, prev_lane
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image received: shape = %s", self.image.shape)
        except Exception as e:
            rospy.logerr(f"[Image Conversion Error] {e}")
        blur = cv2.GaussianBlur(self.image, (5, 5), 0)

        
        # Thresh ########################
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # low_thresh = cv2.getTrackbarPos('Lower Thresh', 'Thresh')
        # high_thresh = cv2.getTrackbarPos('Upper Thresh', 'Thresh')
        # _, thresh_img = cv2.threshold(gray, low_thresh, high_thresh, cv2.THRESH_BINARY)
        ###############################
        

        # HSV ########################
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
        ###############################

        # Canny ########################
        # low_canny_thresh = cv2.getTrackbarPos('Lower Thresh', 'Canny')
        # high_canny_thresh = cv2.getTrackbarPos('Upper Thresh', 'Canny')
        # canny_img = cv2.Canny(blur, low_canny_thresh, high_canny_thresh)
        ###############################

        '''
        # Yellow ########################
        l_b = cv2.getTrackbarPos('Lower B', 'Yellow')
        l_g = cv2.getTrackbarPos('Lower G', 'Yellow')
        l_r = cv2.getTrackbarPos('Lower R', 'Yellow')
        u_b = cv2.getTrackbarPos('Upper B', 'Yellow')
        u_g = cv2.getTrackbarPos('Upper G', 'Yellow')
        u_r = cv2.getTrackbarPos('Upper R', 'Yellow')

        lower_yellow = np.array([l_b, l_g, l_r])  # 노란색의 하한값 (B, G, R)
        upper_yellow = np.array([u_b, u_g, u_r])  # 노란색의 상한값 (B, G, R)
    
        mask = cv2.inRange(blur, lower_yellow, upper_yellow)
        yellow_img = cv2.bitwise_and(blur, blur, mask=mask)
        ###############################
        '''

        # sum_img = cv2.bitwise_or(canny_img, hsv_img)
        
        # No Canny
        #warp_img, M, Minv = warp_image(self.image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        #leftx_base, rightx_base, hsv_img, processed_img = image_processing(warp_img)

        # Canny
        # warp_img, M, Minv = warp_image(sum_img, warp_src, warp_dst, (warp_img_w, warp_img_h))
        # leftx_base, rightx_base, processed_img = image_processing_canny(warp_img)

        # left_fit, right_fit, avex, avey, tracker_img = sliding_window(leftx_base, rightx_base, processed_img, lane_flag)
        # lane_img = draw_lane(self.image, warp_img, Minv, left_fit, right_fit, avex, avey)

        # x = avex - 160
        # y = 240 - avey
        # angle = (-1 * math.atan(x / y) / (math.pi / 2)) * max_angle

        if lane_flag == 4 or lane_flag == 5:
            speed = min_speed

            rospy.loginfo("CURV")
            threading.Timer(4.0, self.restore_speed, args=(lane_flag,)).start()

        #if is_lane == True:
           #self.drive_info.drive.speed = speed
           #self.drive_info.drive.steering_angle = angle
           #self.drive_pub.publish(self.drive_info)

        #else:
            #rospy.loginfo("No Lane Detected")

        #if lane_flag == 3:
            #speed = 0.0
            #threading.Timer(4.0, self.restore_speed_stop).start()
        
        #else:
            #prev_lane = lane_flag
        #result_img = cv2.bitwise_and(self.image, self.image, mask=hsv_img)
        if self.image is None:
            rospy.logwarn("Image is None!")
        elif self.image.shape[0] == 0 or self.image.shape[1] == 0:
            rospy.logwarn("Image shape is invalid: %s", self.image.shape)
        else:
            cv2.imshow("HSV", self.image)
            cv2.waitKey(30)
       # cv2.imshow("HSV", self.image)
        # cv2.imshow("tracker", tracker_img)
        # cv2.imshow("Canny", canny_img)
        # cv2.imshow("sum_img", sum_img)

        # cv2.imshow("warp", warp_img)
        # cv2.imshow("result", lane_img)
        # cv2.imshow("Thresh", thresh_img)
        #cv2.imshow("Yellow", yellow_img)

        #cv2.waitKey(1)
        
        
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


def image_processing(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Thresh ########################
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    low_thresh = cv2.getTrackbarPos('Lower Thresh', 'Thresh')
    high_thresh = cv2.getTrackbarPos('Upper Thresh', 'Thresh')
    _, thresh_img = cv2.threshold(gray, low_thresh, high_thresh, cv2.THRESH_BINARY)
    ###############################

    # HSV ########################
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
    ###############################

    histogram = np.sum(hsv_img[hsv_img.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    processed_img = np.dstack((hsv_img, hsv_img, hsv_img)) * 255

    return leftx_base, rightx_base, hsv_img, processed_img


def sliding_window(leftx_base, rightx_base, processed_img, lane_flag):
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
    
    for window in range(nwindows): 

        win_yl = processed_img.shape[0] - (window + 1) * window_height 
        win_yh = processed_img.shape[0] - window * window_height 
        
        win_xll = leftx_current - margin 
        win_xlh = leftx_current + margin 
        win_xrl = rightx_current - margin 
        win_xrh = rightx_current + margin

        if lane_flag == 0 or lane_flag == 4: # 좌측 차선만
            cv2.rectangle(processed_img, (win_xll, win_yl), (win_xlh,win_yh), (0,255, 0), 2) 
        
        elif lane_flag == 1 or lane_flag == 5: # 우측 차선만
            cv2.rectangle(processed_img, (win_xrl, win_yl), (win_xrh,win_yh), (0,255, 0), 2) 

        else: # 양쪽 차선
            cv2.rectangle(processed_img, (win_xll, win_yl), (win_xlh,win_yh), (0,255, 0), 2) 
            cv2.rectangle(processed_img, (win_xrl, win_yl), (win_xrh,win_yh), (0,255, 0), 2) 

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
        rospy.loginfo("Right Lane Detected")

    else: # 양쪽 차선
        lfit = np.polyfit(np.array(ly), np.array(lx), 2) 
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)

        processed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] 
        processed_img[nonzeroy[right_lane_inds] , nonzerox[right_lane_inds]] = [0, 0, 255] 
        
        avex = int(np.array(lx).mean() + np.array(rx).mean())//2
        avey = int(np.array(ly).mean() + np.array(ry).mean())//2
    
    if lefty == 0 and righty == 0: 
        is_lane = False
        avex = warp_img_w//2
        avey = warp_img_h//2

    #elif lefty == 0 or righty == 0:
    #    is_lane = False
    #    avex = warp_img_w//2
    #    avey = warp_img_h//2

    else:
        is_lane = True
    
    cv2.circle(processed_img,(avex, avey), 5, (0, 255, 255), -1)

    return lfit, rfit, avex, avey, processed_img


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
    rospy.init_node("lane_detection")
    cam = CameraReceiver()

    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        if cam.latest_image is not None:
            cv2.imshow("HSV", cam.latest_image)
            cv2.waitKey(1)
        rate.sleep()


if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass