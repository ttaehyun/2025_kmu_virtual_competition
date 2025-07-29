import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map
from skimage import exposure

# ==================== 설정 ====================
input_type = 'image'  # 'video'
input_name = '12.jpg'

left_line = Line()
right_line = Line()

# Threshold 최적값
th_sobelx = (37, 111)
th_sobely = (138, 195)
th_mag = (148, 232)
th_dir = (0.86, 1.06)
th_h = (180, 180)
th_l = (195, 232)
th_s = (79, 183)

# src 좌표 (고정값)
src = np.float32([
    [0, 479],       # LB
    [223, 244],     # LT
    [441, 251],     # RT
    [639, 479]      # RB
])

# dst 좌표
def get_dst_for_src(w, h):
    return np.float32([
        [w // 4, h], [w // 4, 0], [3 * w // 4, 0], [3 * w // 4, h]
    ])

warp_size = (640, 480)
dst = get_dst_for_src(*warp_size)

# ==================== 실행 ====================
mtx, dist = calib()

if __name__ == '__main__':
    if input_type == 'image':
        img = cv2.imread(input_name)
        undist_img = img
        rows, cols = undist_img.shape[:2]

        combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
        combined_result = comb_result(combined_gradient, combined_hls)

        warp_img, M, Minv = warp_image(combined_result, src, dst, warp_size)
        searching_img = find_LR_lines(warp_img, left_line, right_line)
        w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)

        # 최종 결과 합성
        color_result = cv2.warpPerspective(w_color_result, Minv, (cols, rows))
        comb_result_img = np.zeros_like(undist_img)
        height = color_result.shape[0]
        comb_result_img[rows - height:rows, 0:cols] = color_result
        result = cv2.addWeighted(undist_img, 1, comb_result_img, 0.3, 0)

        for pt in src.astype(int):
            cv2.circle(result, tuple(pt), 5, (0, 0, 255), -1)

        # 시각화
        # cv2.imshow('BEV', warp_img)
        # cv2.imshow("SRC Points", combined_hls)
        cv2.imshow('Result Overlay', result)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_type == 'video':
        cap = cv2.VideoCapture(input_name)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            undist_img = undistort(frame, mtx, dist)
            undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2)
            rows, cols = undist_img.shape[:2]

            combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
            combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
            combined_result = comb_result(combined_gradient, combined_hls)

            warp_img, M, Minv = warp_image(combined_result, src, dst, warp_size)
            searching_img = find_LR_lines(warp_img, left_line, right_line)
            w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)

            color_result = cv2.warpPerspective(w_color_result, Minv, (cols, rows))
            lane_color = np.zeros_like(undist_img)
            lane_color = color_result
            result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)

            info, info2 = np.zeros_like(result), np.zeros_like(result)
            info[5:110, 5:190] = (255, 255, 255)
            info2[5:110, cols - 111:cols - 6] = (255, 255, 255)
            info = cv2.addWeighted(result, 1, info, 0.2, 0)
            info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
            road_map = print_road_map(w_color_result, left_line, right_line)
            info2[10:105, cols - 106:cols - 11] = road_map
            info2 = print_road_status(info2, left_line, right_line)
            cv2.imshow('road info', info2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
