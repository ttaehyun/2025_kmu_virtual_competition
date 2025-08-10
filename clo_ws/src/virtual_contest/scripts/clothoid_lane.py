#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped

# 이미지 크기 (필요시 사용)
Width = 640
Height = 480

# 전방 시야를 넓게 보되, 좌표는 이미지 밖까지 확장
x_h = 220     # 위쪽 좁게
x_l = 600    # 아래쪽 넓게 (이미지보다 크게)
y_h = -10     # 상단 기준선 위치 (멀리 있는 곳)
y_l = 0     # 하단 여백

warp_width = Width-x_h*2
warp_height = (Height//2-y_l + y_h)
class lane_detect:
    def __init__(self):
        rospy.loginfo("lane detect node is Created")
        self.bridge = CvBridge()

        # 이미지 구독자
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        # 주행 명령 발행자
        self.drive_pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()

        # 처리된 이미지를 저장할 변수들
        self.latest_image = None
        self.lane_image = None # <-- 색상 필터링된 이미지를 저장할 변수 추가
        self.sliding_window_img = None # <-- 시각화 이미지를 저장할 변수
        self.final_image = None
        self.roi_img = None
        self.warped_img = None
        
        self.warp_src = np.array([
            [x_h, Height//2 - y_h], # 좌상단
            [-x_l, Height - y_l], # 좌하단
            [Width - x_h, Height//2 - y_h], # 우상단
            [Width + x_l, Height - y_l] # 우하단
        ], dtype=np.float32)

        self.warp_dst = np.array([
            [0, 0],
            [0, warp_height],
            [warp_width, 0],
            [warp_width, warp_height]
        ], dtype=np.float32)

        # 원근 변환 및 역변환 행렬 계산
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.M_inv = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)
    def color_filter(self, image):
        """
        이미지에서 흰색과 노란색 차선을 검출하여 이진(binary) 이미지로 반환합니다.
        """
        # BGR 이미지를 HSV 색 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 노란색 차선을 위한 HSV 범위 설정
        lower_yellow = np.array([0, 100, 134])
        upper_yellow = np.array([50, 255, 255])
        
        # 흰색 차선을 위한 HSV 범위 설정
        # 흰색은 채도(S)가 낮고, 명도(V)가 높은 특징을 이용합니다.
        lower_white = np.array([0, 0, 178])
        upper_white = np.array([60, 56, 255])

        # 각 색상에 대한 마스크 생성
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 두 마스크를 하나로 합침 (OR 연산)
        # 결과물은 차선 부분만 흰색(255), 나머지는 검은색(0)인 이미지가 됩니다.
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        return combined_mask
    
    def perspective_warp(self, image):
        return cv2.warpPerspective(image, self.M, (warp_width, warp_height), flags=cv2.INTER_LINEAR)
    # === 3단계 핵심 함수: ROI 설정 및 차선 피팅 ===
    def find_lane_pixels_and_fit(self, warped_img):
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        
        # 시각화를 위해 흑백 이미지를 3채널 컬러 이미지로 변환
        out_img = np.dstack((warped_img, warped_img, warped_img))
        
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 10
        minpix = 50
        window_height = int(warped_img.shape[0] / nwindows)

        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # 초록색으로 슬라이딩 윈도우 그리기
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 찾아낸 픽셀들을 색칠하기
        out_img[lefty, leftx] = [255, 0, 0]  # 왼쪽 차선은 파란색
        out_img[righty, rightx] = [0, 0, 255] # 오른쪽 차선은 빨간색

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None
        
        # 이제 시각화 이미지도 함께 반환
        return left_fit, right_fit, out_img
    
    def draw_final_result(self, original_image, warped_image, left_fit, right_fit):
        """
        최종 결과를 그리는 함수 (영역 채우기 대신 선 그리기)
        """
        # 그리기용 빈 이미지 생성
        out_img = np.dstack((original_image, original_image, original_image))
        color_warp = np.zeros_like(out_img).astype(np.uint8)

        h, w = warped_image.shape[:2]
        ploty = np.linspace(0, h - 1, h)

        # 왼쪽 차선 그리기 (파란색)
        if left_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)

        # 오른쪽 차선 그리기 (빨간색)
        if right_fit is not None:
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)
        
        # 역변환: BEV 위의 차선 라인을 다시 원본 카메라 시점으로 되돌림
        unwarped = cv2.warpPerspective(color_warp, self.M_inv, (w, h))
        
        # 원본 이미지와 역변환된 라인을 합성
        result = cv2.addWeighted(out_img, 1, unwarped, 1.0, 0)
        return result
    def callback(self, data):
        """이미지를 수신하면 호출되는 함수"""
        try:
            # 1. 원본 이미지 수신
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            
            # 2. 색상 필터링 적용
            self.lane_image = self.color_filter(self.latest_image)
            self.warped_img = self.perspective_warp(self.lane_image)
            # 시각화 이미지를 반환받아 클래스 변수에 저장
            left_fit, right_fit, self.sliding_window_img = self.find_lane_pixels_and_fit(self.warped_img)
            self.final_image = self.draw_final_result(self.warped_img, self.warped_img, left_fit, right_fit)


        except Exception as e:
            rospy.logerr(f"[Image Processing Error] {e}")

 
def run():
    rospy.init_node("Lane_Detect_Node")
    cam = lane_detect()

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # latest_image와 lane_image가 모두 준비되었는지 확인
        if cam.latest_image is not None and cam.lane_image is not None:
            # 원본 이미지와 필터링된 이미지를 함께 표시
            cv2.imshow("Original Image", cam.latest_image)
            cv2.imshow("warp Lanes", cam.warped_img) # <-- 필터링된 이미지 창 추가
            cv2.imshow("Final Image", cam.final_image)
            cv2.imshow("Sliding Window Visualization", cam.sliding_window_img)
            # cv2.imshow("ROI", cam.roi_img)
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        rate.sleep()

    cv2.destroyAllWindows() # 종료 시 모든 창 닫기

if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        pass