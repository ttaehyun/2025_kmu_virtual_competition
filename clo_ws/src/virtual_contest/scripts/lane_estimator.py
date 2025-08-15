#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64

class LaneEstimator:
    def __init__(self):
        self.bridge = CvBridge()

        self.warp_w = 320
        self.warp_h = 240

        # BEV에서 1픽셀 → 미터 환산 (대략값으로 시작 후 튠)
        # 차선 폭(센터-센터) ~ 3.2m 가정, BEV 하단에서 좌/우 차선 간 픽셀폭 측정해서 적절히 세팅
        self.xm_per_pix = 3.2 / 300.0
        #heading 부호/축 정의 보정 플래그
        self.flip_heading_sign = False
    
        # HSV 임계값(흰색/노란색 둘 다 통과시키고 싶으면 두 범위를 OR)
        self.white_lower = np.array([0, 0, 131])
        self.white_upper = np.array([40, 171, 255])
        self.yellow_lower = np.array([15,  60, 120])
        self.yellow_upper = np.array([40, 255, 255])

        # 마스크 후처리
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        # ROI (하단부 가중치 ↑)
        self.roi_y0 = rospy.get_param("~roi_y0", int(self.warp_h*0.45))
        self.roi_y1 = self.warp_h - 30

        # 신뢰도 계산용 최소 픽셀 수
        self.min_lane_px = rospy.get_param("~min_lane_px", 800)

        # 구독/퍼블리시
        self.sub = rospy.Subscriber("/BEV_image", CompressedImage, self.cb, queue_size=1)
        self.pub_ey = rospy.Publisher("/lane/center_offset", Float64, queue_size=1)
        self.pub_epsi = rospy.Publisher("/lane/heading_error", Float64, queue_size=1)
        self.pub_conf = rospy.Publisher("/lane/confidence", Float64, queue_size=1)

        self.pub_mask = rospy.Publisher("/lane/debug_mask", CompressedImage, queue_size=1)
        self.pub_overlay = rospy.Publisher("/lane/debug_overlay", CompressedImage, queue_size=1)

        self.mask = None
        self.edges = None
    def cb(self, msg):
        # 1) 이미지 변환
        bev = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        H, W = bev.shape[:2]

        # 2) ROI 추출 (하단부 중심)
        roi = bev[self.roi_y0:self.roi_y1, :]

        # 3) HSV 색 필터 (흰/노란 차선)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, self.white_lower, self.white_upper)
        mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        self.mask = cv2.bitwise_or(mask_white, mask_yellow)

        # 4) 엣지/밝기 보강 (선택사항) → 흰/노란색 외 라인도 일부 감지되게
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(gray, 60, 150)
        self.mask = cv2.bitwise_or(self.mask, self.edges)

        # 5) 모폴로지로 잡음 정리
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)

        # 6) (선택) 좌우 하단 히스토그램으로 중앙 추정
        hist = np.sum(self.mask[self.mask.shape[0]//2:, :], axis=0)
        # 너무 들쭉날쭉하면 가우시안 블러
        hist_smooth = cv2.GaussianBlur(hist.astype(np.float32), (31,1), 0)

        # 7) 레인 포인트 좌표 추출
        ys, xs = np.where(self.mask > 0)
        # ROI 오프셋 보정 (전체 BEV 좌표계로)
        ys = ys + self.roi_y0

        # 신뢰도: 픽셀 수와 분포로 간단 계산
        lane_px = len(xs)
        conf = float(min(1.0, max(0.0, lane_px / float(self.min_lane_px))))
        # 분포가 너무 한쪽에 몰리면(= 단일 차선만) 보정
        if lane_px > 0:
            left_ratio = (xs < W//2).sum() / float(lane_px)
            right_ratio = 1.0 - left_ratio
            # 좌우 균형이 0.1~0.9 사이면 가산점, 아니면 감점
            balance = 1.0 - abs(left_ratio - 0.5) * 2.0
            conf = 0.7*conf + 0.3*max(0.0, balance)

        # 유효성 체크
        if lane_px < 50:
            self.publish(0.0, 0.0, 0.0, bev, self.mask, hist_smooth)
            return

        # 8) 차선 방향(heading) 추정: cv2.fitLine (y-증가: 아래방향)
        #    fitLine은 (vx,vy), (x0,y0)를 줌. 세로(전진)축은 +y라고 보면,
        #    lane angle = atan2(vx, vy). vy가 1이면 수직(각도 0)과 유사.
        line = cv2.fitLine(np.column_stack((xs, ys)).astype(np.float32),
                           distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = [float(v) for v in line]
        lane_angle = np.arctan2(vx, vy)  # rad, 0이면 세로 정렬
        epsi = -lane_angle  # 차량 헤딩이 +y(아래)로 정렬 가정 → 부호 보정
        if self.flip_heading_sign:
            epsi = -epsi

        # 9) 중심 오프셋(e_y) 추정: 하단(y=H-1)에서 센터라인 x와 이미지 중앙의 차
        #    fit line으로 y=H-1일 때 x = x0 + vx/vy * (y - y0)
        y_eval = H - 1
        if abs(vy) < 1e-6:
            x_at_bottom = x0  # 수평에 가까울 때는 근사
        else:
            x_at_bottom = x0 + (vx / vy) * (y_eval - y0)
        # 차량 중심(카메라 중심) 기준: W/2
        dx_pix = float(x_at_bottom - (W / 2.0))
        ey_m = dx_pix * self.xm_per_pix  # +: 오른쪽(+x) 이탈로 정의

        # 10) 디버그 오버레이
        overlay = bev.copy()
        # ROI 사각형
        cv2.rectangle(overlay, (0, self.roi_y0), (W-1, self.roi_y1-1), (64,64,64), 1)
        # 센터라인 & 차량센터
        cv2.line(overlay, (W//2, self.roi_y0), (W//2, self.roi_y1-1), (255,0,0), 1)
        # 적중점
        cv2.circle(overlay, (int(np.clip(x_at_bottom,0,W-1)), y_eval), 3, (0,0,255), -1)
        # 방향 표시 (y_eval에서 위로 40px)
        x2 = int(np.clip(x_at_bottom + 40*np.tan(lane_angle), 0, W-1))
        y2 = int(max(0, y_eval-40))
        cv2.line(overlay, (int(np.clip(x_at_bottom,0,W-1)), y_eval), (x2, y2), (0,255,0), 2)

        # 텍스트
        cv2.putText(overlay, f"ey={ey_m:+.2f} m", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
        cv2.putText(overlay, f"epsi={epsi:+.3f} rad", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
        cv2.putText(overlay, f"conf={conf:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)

        # 11) 퍼블리시
        self.publish(ey_m, epsi, conf, overlay, self.mask, hist_smooth)

    def publish(self, ey, epsi, conf, overlay_bgr, mask_roi, hist):
        # 수치 정보 출력
        rospy.loginfo(f"ey={ey:+.2f} m, epsi={epsi:+.3f} rad, conf={conf:.2f}")

        # 디버그 표시
        cv2.imshow("Lane Mask", mask_roi)      # 이진 마스크
        cv2.imshow("Lane Overlay", overlay_bgr)  # BEV + 차선 중심/방향 표시
        cv2.imshow("mask", self.mask)  # 전체 마스크
        cv2.imshow("edges", self.edges)  # 엣지 검출 결과
        cv2.waitKey(1)

def main():
    rospy.init_node("lane_estimator")
    LaneEstimator()
    rospy.loginfo("lane_estimator node started. Subscribing /BEV_image")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass