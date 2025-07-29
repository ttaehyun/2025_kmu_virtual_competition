import cv2
import numpy as np

# 🔧 이미지 로드 및 전처리
# img = cv2.imread("test_images/straight_lines1.jpg")  # ← 여기 원하는 이미지 경로 넣기
img = cv2.imread("3.jpg")  # ← 여기 원하는 이미지 경로 넣기
blur = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def nothing(x):
    pass

# 🎛 트랙바 창 생성
cv2.namedWindow("Gradients")
cv2.namedWindow("HLS")

# Sobel X, Y Threshold
cv2.createTrackbar("SobelX Min", "Gradients", 37, 255, nothing)
cv2.createTrackbar("SobelX Max", "Gradients", 111, 255, nothing)
cv2.createTrackbar("SobelY Min", "Gradients", 138, 255, nothing)
cv2.createTrackbar("SobelY Max", "Gradients", 195, 255, nothing)

# Magnitude
cv2.createTrackbar("Mag Min", "Gradients", 148, 255, nothing)
cv2.createTrackbar("Mag Max", "Gradients", 232, 255, nothing)

# Direction (x100으로 조절)
cv2.createTrackbar("Dir Min (x100)", "Gradients", int(0.86*100), 300, nothing)
cv2.createTrackbar("Dir Max (x100)", "Gradients", int(1.06*100), 300, nothing)

# HLS
cv2.createTrackbar("H Min", "HLS", 175, 180, nothing)
cv2.createTrackbar("H Max", "HLS", 180, 180, nothing)
cv2.createTrackbar("L Min", "HLS", 195, 255, nothing)
cv2.createTrackbar("L Max", "HLS", 232, 255, nothing)
cv2.createTrackbar("S Min", "HLS", 79, 255, nothing)
cv2.createTrackbar("S Max", "HLS", 183, 255, nothing)

while True:
    # 🔁 Trackbar 값 읽기
    sx_min = cv2.getTrackbarPos("SobelX Min", "Gradients")
    sx_max = cv2.getTrackbarPos("SobelX Max", "Gradients")
    sy_min = cv2.getTrackbarPos("SobelY Min", "Gradients")
    sy_max = cv2.getTrackbarPos("SobelY Max", "Gradients")
    mag_min = cv2.getTrackbarPos("Mag Min", "Gradients")
    mag_max = cv2.getTrackbarPos("Mag Max", "Gradients")
    dir_min = cv2.getTrackbarPos("Dir Min (x100)", "Gradients") / 100.0
    dir_max = cv2.getTrackbarPos("Dir Max (x100)", "Gradients") / 100.0

    h_min = cv2.getTrackbarPos("H Min", "HLS")
    h_max = cv2.getTrackbarPos("H Max", "HLS")
    l_min = cv2.getTrackbarPos("L Min", "HLS")
    l_max = cv2.getTrackbarPos("L Max", "HLS")
    s_min = cv2.getTrackbarPos("S Min", "HLS")
    s_max = cv2.getTrackbarPos("S Max", "HLS")

    # 📐 Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.uint8(255 * np.abs(sobelx) / np.max(np.abs(sobelx)))
    sobelx_bin = cv2.inRange(abs_sobelx, sx_min, sx_max)

    # 📐 Sobel Y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.uint8(255 * np.abs(sobely) / np.max(np.abs(sobely)))
    sobely_bin = cv2.inRange(abs_sobely, sy_min, sy_max)

    # 📐 Magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(255 * mag / np.max(mag))
    mag_bin = cv2.inRange(mag, mag_min, mag_max)

    # 📐 Direction
    dir = np.arctan2(np.abs(sobely), np.abs(sobelx))
    dir_bin = ((dir >= dir_min) & (dir <= dir_max)).astype(np.uint8) * 255

    # 🎨 HLS Threshold
    H, L, S = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    h_bin = cv2.inRange(H, h_min, h_max)
    l_bin = cv2.inRange(L, l_min, l_max)
    s_bin = cv2.inRange(S, s_min, s_max)

    hls_comb = cv2.bitwise_or(h_bin, s_bin)
    hls_comb = cv2.bitwise_and(hls_comb, l_bin)

    # 💡 Visualization
    cv2.imshow("SobelX", sobelx_bin)
    cv2.imshow("SobelY", sobely_bin)
    cv2.imshow("Magnitude", mag_bin)
    cv2.imshow("Direction", dir_bin.astype(np.uint8))
    cv2.imshow("HLS Combined", hls_comb)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
