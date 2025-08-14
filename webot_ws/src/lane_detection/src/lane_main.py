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

# 최소 세로 연속 길이 비율( warp_img_h 에 대한 비율 ) — 점선 무시 기준
MIN_VERTICAL_RUN_RATIO = 0.30

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

x_h = 160 # 80
x_l = 600
y_h = 30
y_l = 40

DEBUG_VIS = True  # 필요 없을 땐 False


# ===== 색상 마스크 튜닝값(HSV) =====
# 노란색(왼쪽 중앙선): 조명 따라 H 15~35, S 80~255, V 100~255 권장
Y_H_L, Y_H_U = 15, 35
Y_S_L, Y_S_U = 80, 255
Y_V_L, Y_V_U = 100, 255

# 흰색(우측 실선): 채도 낮고 밝기 높음 — S 0~60, V 200~255 권장
W_S_L, W_S_U = 0, 90    # 60 → 90
W_V_L, W_V_U = 170, 255

EDGE_BAND_RATIO = 0.22      # 0.18~0.28
MIN_RUN_RATIO_BOOT = 0.30   # 0.28~0.35 (세로 bbox 높이 비율)



# sliding window parameter
nwindows = 20
margin = 20
minpix = 15 # 수정
lane_width = 90

# 0:left, 1:right, 2:both
# lane_flag = 1

angle = 0.0
max_angle = 1.0

speed = 0.9
max_speed = 0.4
min_speed = 0.4

is_lane = True
prev_lane = 1

def nothing(x):
    pass

cv2.namedWindow('HSV')
#cv2.namedWindow('Thresh')
cv2.namedWindow('Canny')
#cv2.namedWindow('Yellow')

# HSV
cv2.createTrackbar('Lower H', 'HSV', 0, 180, nothing)
cv2.createTrackbar('Lower S', 'HSV', 29, 255, nothing) 
cv2.createTrackbar('Lower V', 'HSV', 93, 255, nothing) 
cv2.createTrackbar('Upper H', 'HSV', 85, 180, nothing)
cv2.createTrackbar('Upper S', 'HSV', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'HSV', 255, 255, nothing)

'''
# Threshlod
cv2.createTrackbar('Lower Thresh', 'Thresh', 200, 255, nothing) 
cv2.createTrackbar('Upper Thresh', 'Thresh', 255, 255, nothing) 
'''

# Canny
cv2.createTrackbar('Lower Thresh', 'Canny', 50, 255, nothing) # 50 # 0 
cv2.createTrackbar('Upper Thresh', 'Canny', 100, 255, nothing) # 100 # 150

'''
# Yellow
cv2.createTrackbar('Lower B', 'Yellow', 0, 255, nothing) 
cv2.createTrackbar('Lower G', 'Yellow', 200, 255, nothing)
cv2.createTrackbar('Lower R', 'Yellow', 200, 255, nothing)
cv2.createTrackbar('Upper B', 'Yellow', 50, 255, nothing)
cv2.createTrackbar('Upper G', 'Yellow', 255, 255, nothing)
cv2.createTrackbar('Upper R', 'Yellow', 255, 255, nothing)
'''

# warp_src = np.array([
#     # [x_h, Height//2 + y_h], # 좌상단
#     # [-x_l, Height - y_l], # 좌하단
#     # [Width - x_h, Height//2 + y_h], # 우상단
#     # [Width + x_l, Height - y_l] # 우하단
#     [Width * 0.42, Height * 0.55],   # top-left
#     [Width * 0.57, Height * 0.55],   # top-right ✅
#     [Width * 1.0, Height * 1.0],  # bottom-right ✅
#     [Width * 0.0, Height * 1.0],  # bottom-left
# ], dtype=np.float32)

# warp_dst = np.array([
#     [0, 0],                       # top-left
#     [warp_img_w, 0],             # top-right
#     [warp_img_w, warp_img_h],    # bottom-right
#     [0, warp_img_h]              # bottom-left
# ], dtype=np.float32)


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

        # 🔁 기존: uncompressed 이미지 구독
        # rospy.Subscriber("/usb_cam/image_raw/calib", Image, self.callback)

        # ✅ MORAI의 compressed 이미지 토픽 구독
        # rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.prev_fit_left = None
        self.prev_fit_right = None

        self.lane_flag = 1              # ← 전역 대신 멤버
        self.prev_lane_flag = self.lane_flag

        self.boot_right_on = False
        self.boot_right_until = rospy.Time(0)
        self.boot_lock = 0
        self.BOOT_NEED_LOCK = 2
        self.BOOT_SECS = 2.0

        self.switch_slow_speed = rospy.get_param("~switch_slow_speed", 0.4)   # 임시 감속 속도
        self.switch_slow_secs  = rospy.get_param("~switch_slow_secs",  2.0)   # 감속 지속 시간(초)
        self._restore_timer = None
        self._saved_speed_after_switch = None

        topic_name = rospy.get_param("~lane_flag_topic", "/lane_flag")
        rospy.Subscriber(topic_name, Int32, self.lane_switch_cb, queue_size=1, tcp_nodelay=True)


        rospy.Subscriber(
            "/image_jpeg/compressed",
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size=2**16,        # 64KB 정도면 충분 (너무 크게 잡지 말기)
            tcp_nodelay=True,
        )

        rospy.Subscriber("/direction_flag", Int32, self.flag_callback)
        rospy.Subscriber("/parking_flag", Bool, self.parking_callback)

        # 차량이 ackermann 퍼블리셔를 지원하지 않음 
        # self.drive_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_6", AckermannDriveStamped, queue_size=1)
        # self.drive_info = AckermannDriveStamped()

        # low-level 퍼블리싱
        # self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        # self.steer_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        # ✅ Ackermann 메시지 퍼블리셔 추가
        self.ackermann_pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)

        self.parking_flag = False
        self.start_time = time.time()

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

    def change_lane(self): # 주차용 차선 변경
        global lane_flag, lane_width

        rospy.loginfo("AR Change lane")
        lane_flag = 1
        lane_width = 135 

    def parking_callback(self, data): # 주차용 차선 변경
        self.parking_flag = data.data

        if self.parking_flag:
            threading.Timer(9.0, self.change_lane).start()

    def flag_callback(self, data): # 교차로 차선 변경
        global lane_flag

        if not self.parking_flag:  # AR 미션 전에만 변경
            lane_flag = data.data

    
    def lane_switch_cb(self, msg):
        v = int(msg.data)
        if v not in (0,1) or v == self.lane_flag:
            return
        old = self.lane_flag
        self.lane_flag = v

        # 공통 리셋
        self.prev_fit_left = None
        self.prev_fit_right = None
        self.prev_base = None
        self.boot_lock = 0

        # 오른쪽으로 바뀌면 부트스트랩 ON (왼쪽은 필요 없음)
        if self.lane_flag == 1:
            self.boot_right_on = True
            self.boot_right_until = rospy.Time.now() + rospy.Duration.from_sec(self.BOOT_SECS)
            rospy.loginfo(f"[BOOT-R] start {self.BOOT_SECS:.1f}s")
        else:
            self.boot_right_on = False
            rospy.loginfo("[BOOT-R] off (left mode)")

        global speed
        # 진행 중인 복구 타이머가 있으면 취소
        if self._restore_timer is not None:
            try:
                if self._restore_timer.is_alive():
                    self._restore_timer.cancel()
            except Exception:
                pass
            self._restore_timer = None

        # 현재 속도를 저장해 두고, 즉시 임시 감속
        self._saved_speed_after_switch = float(speed)
        speed = float(self.switch_slow_speed)
        rospy.loginfo(f"[LaneSwitch] slow to {speed:.2f} m/s for {self.switch_slow_secs:.1f}s (then restore to {self._saved_speed_after_switch:.2f})")

        # 타이머로 복구 예약
        self._restore_timer = threading.Timer(self.switch_slow_secs, self._restore_speed_after_switch)
        self._restore_timer.daemon = True
        self._restore_timer.start()

    def _restore_speed_after_switch(self):
        """lane_flag 변경 후 임시 감속이 끝나면 원래 속도로 복구"""
        global speed
        if self._saved_speed_after_switch is not None:
            speed = float(self._saved_speed_after_switch)
        self._saved_speed_after_switch = None
        self._restore_timer = None

    def callback(self, data):
        global speed

        def get_lookahead_points(rx, ry, num_points=5, spacing=5):
            """
            곡선을 따라 일정 간격으로 lookahead point 추출
            """
            lookahead_rx = []
            lookahead_ry = []

            for i in range(0, len(rx), spacing):
                if len(lookahead_rx) >= num_points:
                    break
                lookahead_rx.append(rx[i])
                lookahead_ry.append(ry[i])
            
            return lookahead_rx, lookahead_ry

        # ✅ 압축 이미지 처리
        np_arr = np.frombuffer(data.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        blur = cv2.GaussianBlur(self.image, (5, 5), 0)

        '''
        # Thresh ########################
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        low_thresh = cv2.getTrackbarPos('Lower Thresh', 'Thresh')
        high_thresh = cv2.getTrackbarPos('Upper Thresh', 'Thresh')
        _, thresh_img = cv2.threshold(gray, low_thresh, high_thresh, cv2.THRESH_BINARY)
        ###############################
        '''

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
        low_canny_thresh = cv2.getTrackbarPos('Lower Thresh', 'Canny')
        high_canny_thresh = cv2.getTrackbarPos('Upper Thresh', 'Canny')
        canny_img = cv2.Canny(blur, low_canny_thresh, high_canny_thresh)
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


        # === (추가) 색 마스크 ===
        mask = color_mask_by_mode(hsv, self.lane_flag)  # 0=노란, 1=흰
        if mask is not None:
            # 1) 마스크를 살짝 키워서 Canny 경계와 겹치게
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_d = cv2.dilate(mask, k3, iterations=1)

            # 2) Canny와는 AND가 아니라 OR로 결합 (교집합=0 문제 회피)
            sum_img = cv2.bitwise_or(canny_img, mask_d)

            # 3) 마스크의 테두리(edge)도 합쳐서 선을 더 또렷하게
            #    (마스크 내부만 채워진 경우 경계선 생성)
            mask_edge = cv2.morphologyEx(mask_d, cv2.MORPH_GRADIENT, k3)
            sum_img = cv2.bitwise_or(sum_img, mask_edge)

            # 4) 수직선 강화: 세로로 1픽셀 팽창 → 끊김 방지
            kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            sum_img = cv2.dilate(sum_img, kv, iterations=1)

            # (디버그) 합성 결과가 너무 비면 임계 완화
            if cv2.countNonZero(sum_img) < 300:
                # 최후의 안전장치: OR 가중 합 (약간 밝게)
                sum_img = cv2.addWeighted(canny_img, 1.0, mask_d, 0.7, 0.0)
        else:
            # 양쪽 모드일 때의 폴백 (필요시 유지/조정)
            hsv_white = cv2.inRange(hsv, np.array([0, W_S_L, W_V_L]), np.array([180, W_S_U, W_V_U]))
            sum_img = cv2.bitwise_or(canny_img, hsv_white)

        # === BEV ===
        warp_img, M, Minv = warp_image(sum_img, warp_src, warp_dst, (warp_img_w, warp_img_h))
        if warp_img.dtype != np.uint8:
            warp_img = warp_img.astype(np.uint8)
        _, warp_img = cv2.threshold(warp_img, 1, 255, cv2.THRESH_BINARY)

        # ===== 우측-부트스트랩 모드 =====
        boot_active = self.boot_right_on and (rospy.Time.now() < self.boot_right_until)
        if boot_active and self.lane_flag == 1:
            # 1) 우측 가장자리 앵커
            boot_img = roi_edge_anchor(warp_img, side=1)
            # 2) 점선 제거
            boot_img = filter_solid_by_vertical_extent(boot_img, MIN_RUN_RATIO_BOOT)

            # 3) 베이스(중앙 마스킹 없이 전체 히스토그램의 우반부만)
            leftx_base, rightx_base, processed_img = find_base_full(boot_img, lane_flag=1)

            # 4) 슬윈으로 fit 추정
            left_fit, right_fit, avex, avey, tracker_img, rx, ry = sliding_window(
                leftx_base, rightx_base, processed_img, 1, boot_img
            )

            ok = cv2.countNonZero(boot_img) > 400
            if ok and right_fit is not None:
                self.prev_fit_right = right_fit
                self.boot_lock += 1
            else:
                self.boot_lock = 0

            # 조기 종료 조건(락온) 또는 시간 만료
            if self.boot_lock >= self.BOOT_NEED_LOCK or rospy.Time.now() >= self.boot_right_until:
                self.boot_right_on = False
                rospy.loginfo(f"[BOOT-R] done (lock={self.boot_lock})")

            # → 이 프레임은 부트 결과로 바로 조향/퍼블리시하고 return
            angle = math.atan2(avex - 160, 360 - avey)
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = angle
            self.ackermann_pub.publish(drive_msg)

            # 디버그
            # cv2.imshow("BOOT-R", boot_img)
            # cv2.imshow("tracker", tracker_img)
            cv2.waitKey(1)
            return


        # === 코리도어 적용 (자동확장) ===
        corr_applied = False
        masked, corridor = (None, None)

        def try_corridor(prev_fit, lane_flag):
            if prev_fit is None:
                return None, None
            for scale in (1.0, 1.3, 1.7, 2.2):            # 자동 확장 단계
                corr = make_corridor_mask(warp_img_h, warp_img_w, prev_fit,
                                        lane_flag=lane_flag, scale=scale)
                masked = cv2.bitwise_and(warp_img, corr)
                if cv2.countNonZero(masked) > 350:        # 임계(환경 따라 250~600)
                    return masked, corr
            return None, None
        
        fit_ref = self.prev_fit_left if self.lane_flag==0 else self.prev_fit_right
        if self.lane_flag == 0:
            masked, corridor = try_corridor(self.prev_fit_left, 0)
        else:
            masked, corridor = try_corridor(self.prev_fit_right, 1)

        # 4) 적응형 중앙 가드
        guard_mask, guard_on = make_adaptive_center_guard(warp_img_h, warp_img_w, fit_ref, self.lane_flag)

        # 5) 베이스/슬윈
        if corr_applied and fit_ref is not None:
            leftx_base, rightx_base = bases_from_fit(warp_img_h, warp_img_w, fit_ref, self.lane_flag)
            processed_img = np.dstack((warp_img, warp_img, warp_img)) * 255
        else:
            leftx_base, rightx_base, processed_img = find_base_full(warp_img, self.lane_flag)

        left_fit, right_fit, avex, avey, tracker_img, rx, ry = sliding_window(
            leftx_base, rightx_base, processed_img, self.lane_flag, warp_img
        )

        # 6) fit 저장
        if self.lane_flag == 0 and left_fit is not None:
            self.prev_fit_left = left_fit
        elif self.lane_flag == 1 and right_fit is not None:
            self.prev_fit_right = right_fit

        x = avex - 160
        y = 360 - avey
        angle = math.atan2(x, y)

        if is_lane == True:
            # ✅ Ackermann 메시지 구성 및 퍼블리시
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = angle
            self.ackermann_pub.publish(drive_msg)

        # cv2.imshow("HSV", hsv_img)
        cv2.imshow("tracker", tracker_img)
        # cv2.imshow("Canny", canny_img)
        # cv2.imshow("sum_img", sum_img)

        image_with_warp = self.image.copy()

        # 점 찍기
        for pt in warp_src:
            cv2.circle(image_with_warp, tuple(pt.astype(int)), 5, (0, 255, 255), -1)

        # 선 그리기
        for i in range(4):
            pt1 = tuple(warp_src[i].astype(int))
            pt2 = tuple(warp_src[(i + 1) % 4].astype(int))
            cv2.line(image_with_warp, pt1, pt2, (255, 255, 0), 2)

        # ✅ Bird's Eye View (컬러 원본 이미지로부터)
        warp_color, _, _ = warp_image(self.image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        cv2.imshow("Bird's Eye View Color", warp_color)

        # cv2.imshow("Original + warp area", image_with_warp)
        # cv2.imshow("Bird's Eye View", warp_img)
        # cv2.imshow("result", lane_img)
        # cv2.imshow("Thresh", thresh_img)
        #cv2.imshow("Yellow", yellow_img)

        cv2.waitKey(1)


def roi_edge_anchor(binary_bev, side, ratio=EDGE_BAND_RATIO, edge_margin=6):
    h, w = binary_bev.shape[:2]
    band = int(ratio * w)
    mask = np.zeros_like(binary_bev)
    if side == 0:  # 왼쪽 필요시도 사용 가능
        x0, x1 = edge_margin, min(band, w-1)
    else:          # 오른쪽
        x0, x1 = max(0, w - band), w - edge_margin
    mask[:, x0:x1] = 255
    return cv2.bitwise_and(binary_bev, mask)

def filter_solid_by_vertical_extent(binary_bev, min_run_ratio=MIN_RUN_RATIO_BOOT):
    h, w = binary_bev.shape[:2]
    min_h = int(min_run_ratio * h)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_bev, connectivity=8)
    out = np.zeros_like(binary_bev)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bh >= min_h:
            out[labels == i] = 255
    return out


def draw_fit_on_bev(vis, fit, color=(0, 0, 255), thickness=2):
    """ fit=[A,B,C]을 BEV(vis) 위에 곡선으로 그려준다. vis는 3채널 이미지 """
    if fit is None: 
        return vis
    A, B, C = fit
    h, w = vis.shape[:2]
    y = np.arange(h, dtype=np.float32)
    x = (A*y*y + B*y + C).astype(np.int32)
    x = np.clip(x, 0, w-1)
    pts = np.stack([x, y.astype(np.int32)], axis=1)
    for i in range(len(pts)-1):
        cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), color, thickness)
    return vis

def show_corridor_debug(bev_binary, corridor_mask, fit, lane_flag, win_name="BEV+corridor"):
    """
    bev_binary: 0/255 단일 채널 BEV 입력(코리도어 적용 '전' 또는 '후' 아무거나)
    corridor_mask: 0/255 코리도어 마스크
    fit: 이전 프레임 polyfit (해당 차선 쪽)
    lane_flag: 0=왼쪽, 1=오른쪽
    """
    if not DEBUG_VIS:
        return
    # 3채널로 만들고 투명 오버레이
    vis = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()

    # 코리도어 윤곽선(노란색) + 반투명 채움
    contours, _ = cv2.findContours(corridor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), thickness=cv2.FILLED)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

        # 윤곽선 테두리 강조
        cv2.drawContours(vis, contours, -1, (0, 255, 255), thickness=2)

    # 이전 프레임의 fit 곡선(빨강)도 같이 그림
    vis = draw_fit_on_bev(vis, fit, color=(0, 0, 255), thickness=2)

    # 화면 좌하단에 non-zero 카운트 찍기
    nz_bev = int(cv2.countNonZero(bev_binary))
    nz_cor = int(cv2.countNonZero(corridor_mask))
    cv2.putText(vis, f"nz_bev={nz_bev}  nz_corr={nz_cor}  lane={lane_flag}",
                (12, vis.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(win_name, vis)

def make_adaptive_center_guard(h, w, fit, lane_flag,
                               base=18,        # 기본 중앙 여유(px)
                               alpha=0.6,      # 타깃이 센터에서 멀수록 가드 폭 증가
                               gmin=-8, gmax=60,   # 가드 폭 하/상한
                               on_thresh_px=40):   # 평균 센터 거리 이하면 가드 OFF
    """
    fit: np.polyfit(y,x,2) (이전 프레임 타깃 차선)
    lane_flag: 0=왼쪽(노란선), 1=오른쪽(흰선)
    return: (mask, enabled)  # mask: 0/255, enabled: 적용 여부
    """
    if fit is None:
        return np.ones((h, w), np.uint8) * 255, False

    A, B, C = fit
    mid = w // 2
    y = np.arange(h, dtype=np.float32)
    x = (A*y*y + B*y + C)          # 타깃 차선 예측 x(y)

    # 타깃과 센터의 평균 거리
    dist = np.mean(x - mid) if lane_flag == 1 else np.mean(mid - x)
    if dist < on_thresh_px:
        # 타깃이 센터에 가깝다 → 가드 끔 (커브에서 잘리지 않게)
        return np.ones((h, w), np.uint8) * 255, False

    # y별 가드 폭: base + alpha * (타깃-센터), 클립
    g = base + alpha * (x - mid if lane_flag == 1 else mid - x)
    g = np.clip(g, gmin, gmax)

    # 가드 경계 곡선: 오른쪽 모드면 x_guard = mid + g(y) 보다 "왼쪽"을 지움
    x_guard = (mid + g) if lane_flag == 1 else (mid - g)
    x_guard = x_guard.astype(np.int32)
    x_guard = np.clip(x_guard, 0, w-1)

    # 마스크 생성(허용 영역=255)
    mask = np.zeros((h, w), dtype=np.uint8)
    if lane_flag == 1:
        # 중앙~좌측을 지우고 우측만 남김
        for yy in range(h):
            mask[yy, x_guard[yy]:] = 255
    else:
        # 중앙~우측을 지우고 좌측만 남김
        for yy in range(h):
            mask[yy, :x_guard[yy]+1] = 255
    return mask, True

def make_corridor_mask(h, w, fit, lane_flag,
                       base_half=16,      # 직선 기본 반폭
                       curve_gain=0.08,   # 곡률(A) 기반 가산
                       slope_gain=6.0,    # 기울기(|2Ay+B|) 기반 가산
                       center_bias=14,    # 중앙쪽 여유
                       scale=1.0):        # 전체 배율(자동-확장용)
    """
    h,w : BEV 크기
    fit : np.polyfit( y, x, 2 ) 결과 [A,B,C]
    lane_flag : 0=왼쪽(노란선), 1=오른쪽(흰선)
    base_half : 기본 반폭
    curve_gain: |A|에 따른 추가 반폭 계수 (0.05~0.12 사이 튜닝)
    center_bias: 중앙쪽으로 더 넓게 허용(커브에서 중앙쪽으로 말리는 걸 살리기 위함)
    """
    A, B, C = fit
    y = np.arange(h, dtype=np.float32)
    x = (A * y * y + B * y + C)

    # 곡률/기울기 기반 반폭 (y별)
    extra_curve = curve_gain * np.abs(A) * (h**2)
    slope = np.abs(2*A*y + B)                 # |dx/dy|
    extra_slope = slope_gain * (slope / (1 + slope))  # 0~slope_gain로 완만히 증가
    half_y = (base_half + extra_curve + extra_slope) * scale
    half_y = np.clip(half_y, base_half, base_half + 28)  # 상한

    # 중앙쪽 비대칭
    if lane_flag == 0:
        half_left  = half_y
        half_right = half_y + center_bias
    else:
        half_left  = half_y + center_bias
        half_right = half_y

    # 상/하 경계 곡선 만들기
    x_left  = (x - half_left).astype(np.int32)
    x_right = (x + half_right).astype(np.int32)
    x_left  = np.clip(x_left,  0, w-1)
    x_right = np.clip(x_right, 0, w-1)

    # 다각형(띠) 폴리곤 구성: 위쪽부터 아래로 좌변, 아래에서 위로 우변
    pts_left  = np.stack([x_left,  y.astype(np.int32)], axis=1)
    pts_right = np.stack([x_right, y.astype(np.int32)], axis=1)[::-1]
    polygon = np.vstack([pts_left, pts_right]).reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


    
def find_extreme_peak(hist, side="left", min_separation=20, thresh_ratio=0.3):
    """
    hist: 1D 히스토그램 (하단 절반 합)
    side: 'left' -> 가장 왼쪽 피크, 'right' -> 가장 오른쪽 피크
    min_separation: 피크 간 최소 간격(픽셀)
    thresh_ratio: 최대값 대비 최소 높이 비율
    return: 피크 x 인덱스 (없으면 -1)
    """
    # 간단 스무딩(이웃 평균)
    k = 21  # 15~31 사이에서 조정 가능
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(hist.astype(np.float32), kernel, mode='same')

    # 임계치
    th = sm.max() * thresh_ratio

    # 로컬 피크 추출
    peaks = []
    for i in range(1, len(sm)-1):
        if sm[i] > th and sm[i] > sm[i-1] and sm[i] >= sm[i+1]:
            # 가까운 피크끼리 병합(높은 쪽 우선)
            if peaks and i - peaks[-1][0] < min_separation:
                if sm[i] > peaks[-1][1]:
                    peaks[-1] = (i, sm[i])
            else:
                peaks.append((i, sm[i]))

    if not peaks:
        return -1

    if side == "left":
        return min(peaks, key=lambda p: p[0])[0]
    else:
        return max(peaks, key=lambda p: p[0])[0]



def color_mask_by_mode(hsv, lane_flag):
    if lane_flag == 0:  # 왼쪽=노란만
        lower = np.array([Y_H_L, Y_S_L, Y_V_L], dtype=np.uint8)
        upper = np.array([Y_H_U, Y_S_U, Y_V_U], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    elif lane_flag == 1:  # 오른쪽=흰만
        s = hsv[:, :, 1]; v = hsv[:, :, 2]
        mask_s = cv2.inRange(s, W_S_L, W_S_U)
        mask_v = cv2.inRange(v, W_V_L, W_V_U)
        return cv2.bitwise_and(mask_s, mask_v)
    else:
        return None


def bases_from_fit(h, w, fit, lane_flag):
    """코리도어가 적용된 프레임: 이전 fit으로 베이스 바로 산출"""
    A, B, C = fit
    y_eval = h - 1
    x = int(A*y_eval*y_eval + B*y_eval + C)
    x = np.clip(x, 0, w-1)
    if lane_flag == 0:   # 왼쪽 차선
        return x, min(w-1, x + lane_width)
    else:                # 오른쪽 차선
        return max(0, x - lane_width), x

def find_base_full(binary_bev, lane_flag, edge_margin=8, center_bias_px=18):
    """
    중앙 마스킹 없이 전체 폭에서 베이스 탐색.
    lane_flag=0(왼쪽)→ 좌반부에서 argmax, 1(오른쪽)→ 우반부에서 argmax.
    center_bias_px: 중앙 쪽을 약간 가중해서 반대편 차선으로 못 튀게 함.
    """
    h, w = binary_bev.shape[:2]
    mid = w // 2
    hist = np.sum(binary_bev[h//2:, :], axis=0).astype(np.float32)

    if lane_flag == 0:
        x0, x1 = edge_margin, mid
        # 중앙쪽 가중치(+): 중앙으로 올수록 1.0→1.2
        ramp = np.linspace(1.0, 1.0 + 0.2, x1-x0)
        hist[x0:x1] *= ramp
        base = np.argmax(hist[x0:x1]) + x0
        leftx_base, rightx_base = base, min(w-1, base + lane_width)
    else:
        x0, x1 = mid, w - edge_margin
        ramp = np.linspace(1.0 + 0.2, 1.0, x1-x0)  # 중앙쪽에 가중치
        hist[x0:x1] *= ramp
        base = np.argmax(hist[x0:x1]) + x0
        rightx_base, leftx_base = base, max(0, base - lane_width)

    processed_img = np.dstack((binary_bev, binary_bev, binary_bev)) * 255
    return leftx_base, rightx_base, processed_img


        
        
def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def image_processing_canny(img, lane_flag):
    """
    img: BEV 이진(0/255)
    lane_flag: 0=왼쪽 실선만, 1=오른쪽 실선만, 그 외=기존 둘 다
    """
    h, w = img.shape[:2]

    histogram = np.sum(img[h // 2:, :], axis=0)

    if lane_flag in (0, 1):
        side = "left" if lane_flag == 0 else "right"
        base = find_extreme_peak(histogram, side=side, min_separation=20, thresh_ratio=0.25)

        # 피크가 안 잡히면(희미함/가림) 기존 방식 fallback
        if base < 0:
            midpoint = histogram.shape[0] // 2
            if lane_flag == 0:
                base = np.argmax(histogram[:midpoint])
            else:
                base = np.argmax(histogram[midpoint:]) + midpoint

        # 한쪽만 쓰므로 반대쪽은 가상으로 lane_width 만큼 이동
        if lane_flag == 0:
            leftx_base  = base
            rightx_base = min(w-1, base + lane_width)
        else:
            rightx_base = base
            leftx_base  = max(0, base - lane_width)

        processed_img = np.dstack((img, img, img)) * 255
        return leftx_base, rightx_base, processed_img

    else:
        # (기존 양쪽 탐지 루틴)
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
    rospy.spin()


if __name__ == "__main__":
    try:
        run()
        
    except rospy.ROSInterruptException:
        pass