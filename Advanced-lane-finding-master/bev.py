import cv2
import numpy as np

# 🔧 이미지 불러오기 및 리사이즈
img = cv2.imread("2.jpg")  # 원하는 이미지 넣기
img = cv2.resize(img, (640, 480))   # BEV 대상 사이즈 고정

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

# 🔘 트랙바 생성 (좌상, 우상, 좌하, 우하)
cv2.createTrackbar("LT x", "Trackbars", 223, 639, nothing)
cv2.createTrackbar("LT y", "Trackbars", 244, 479, nothing)
cv2.createTrackbar("RT x", "Trackbars", 441, 639, nothing)
cv2.createTrackbar("RT y", "Trackbars", 251, 479, nothing)
cv2.createTrackbar("LB x", "Trackbars", 0, 639, nothing)
cv2.createTrackbar("LB y", "Trackbars", 479, 479, nothing)
cv2.createTrackbar("RB x", "Trackbars", 639, 639, nothing)
cv2.createTrackbar("RB y", "Trackbars", 479, 479, nothing)

while True:
    # 🎛 트랙바 좌표값 가져오기
    ltx = cv2.getTrackbarPos("LT x", "Trackbars")
    lty = cv2.getTrackbarPos("LT y", "Trackbars")
    rtx = cv2.getTrackbarPos("RT x", "Trackbars")
    rty = cv2.getTrackbarPos("RT y", "Trackbars")
    lbx = cv2.getTrackbarPos("LB x", "Trackbars")
    lby = cv2.getTrackbarPos("LB y", "Trackbars")
    rbx = cv2.getTrackbarPos("RB x", "Trackbars")
    rby = cv2.getTrackbarPos("RB y", "Trackbars")

    # 🟠 원본 복사본 (빨간 점 표시용)
    preview = img.copy()

    # 🔺 src / dst 좌표 설정
    src = np.float32([[lbx, lby], [ltx, lty], [rtx, rty], [rbx, rby]])
    dst = np.float32([[151, 480], [151, 0], [489, 0], [489, 480]])  # BEV 정사각형 매핑

    # 🔁 BEV 변환 수행
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (640, 480))

    # 🔴 원본 이미지에 점 표시
    for pt in src.astype(int):
        cv2.circle(preview, tuple(pt), 6, (0, 0, 255), -1)

    # 🪞 시각화
    cv2.imshow("Original + src pts", preview)
    cv2.imshow("Warped BEV", warped)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
