#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import LaserScan, CompressedImage
from numpy.linalg import inv
import torch  # YOLOv5를 위해 추가
from ackermann_msgs.msg import AckermannDriveStamped

# 사용자 정의: YOLOv5 모델(.pt 파일)의 절대 경로를 지정하세요.
# 예: '/home/user/catkin_ws/src/my_package/models/yolov5s.pt'
YOLO_MODEL_PATH = '/home/a/2025_kmu_virtual_competition/clo_ws/src/virtual_contest/model/best.pt'

# 파라미터는 2D 라이다의 장착 위치와 방향에 맞게 수정해야 합니다.
parameters_cam = {
    "WIDTH": 640,  # image width
    "HEIGHT": 480,  # image height
    "X": 0.30,  # meter
    "Y": 0,
    "Z": 0.11,
    "YAW": 0,  # radian
    "PITCH": 0.0,
    "ROLL": 0
}
parameters_lidar = {
    "X": 0.11,  # meter
    "Y": 0,
    "Z": 0.13,
    "YAW": 0,  # radian
    "PITCH": 0,  # 2D 라이다의 경우 기울기(PITCH)가 중요할 수 있습니다.
    "ROLL": 0
}


def getRotMat(RPY):
    cosR = math.cos(RPY[0])
    cosP = math.cos(RPY[1])
    cosY = math.cos(RPY[2])
    sinR = math.sin(RPY[0])
    sinP = math.sin(RPY[1])
    sinY = math.sin(RPY[2])
    rotRoll = np.array([1, 0, 0, 0, cosR, -sinR, 0, sinR, cosR]).reshape(3, 3)
    rotPitch = np.array([cosP, 0, sinP, 0, 1, 0, -sinP, 0, cosP]).reshape(3, 3)
    rotYaw = np.array([cosY, -sinY, 0, sinY, cosY, 0, 0, 0, 1]).reshape(3, 3)
    return rotYaw.dot(rotPitch.dot(rotRoll))


def getTransformMat(params_cam, params_lidar):
    lidarPosition = np.array([params_lidar.get(i) for i in ("X","Y","Z")])
    camPosition = np.array([params_cam.get(i) for i in ("X","Y","Z")])
    lidarRPY = np.array([params_lidar.get(i) for i in ("ROLL","PITCH","YAW")])
    camRPY = np.array([params_cam.get(i) for i in ("ROLL","PITCH","YAW")])
    camRPY += np.array([-90 * math.pi/180, 0, -90 * math.pi/180])
    camRot = getRotMat(camRPY)
    camTransl = np.array([camPosition])
    Tr_cam_to_vehicle = np.vstack((np.hstack((camRot, camTransl.T)), [0,0,0,1]))
    lidarRot = getRotMat(lidarRPY)
    lidarTransl = np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.vstack((np.hstack((lidarRot, lidarTransl.T)), [0,0,0,1]))
    return inv(Tr_cam_to_vehicle).dot(Tr_lidar_to_vehicle).round(6)


def getCameraMat(params_cam):
    # 제공된 Intrinsic Matrix (핀홀 카메라 모델)
    return np.array([[230.0, 0.0, 320.0],
                     [0.0, 230.0, 240.0],
                     [0.0, 0.0, 1.0]])


class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):
        self.pc_np = None
        self.img = None
        self.ranges = None
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]

        rospy.Subscriber("/lidar2D", LaserScan, self.scan_callback)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.img_callback)
        rospy.Subscriber("/ackermann_cmd_mux/input/nav_2", AckermannDriveStamped, self.ack_callback)
        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, queue_size=1)
        self.TransformMat = getTransformMat(params_cam, params_lidar)
        self.CameraMat = getCameraMat(params_cam)
        
        self.ack_nav2 = AckermannDriveStamped()
        self.is_ack_nav2 = False
    def img_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def scan_callback(self, msg):
        pts, rngs = [], []
        angle = msg.angle_min
        for r in msg.ranges:
            if not np.isinf(r) and not np.isnan(r) and msg.range_min < r < msg.range_max:
                pts.append([r*math.cos(angle), r*math.sin(angle), 0.0, 1.0])
                rngs.append(r)
            angle += msg.angle_increment
        self.pc_np = np.array(pts, dtype=np.float32)
        self.ranges = np.array(rngs, dtype=np.float32)

    def ack_callback(self, msg):
        if not self.is_ack_nav2:
            self.is_ack_nav2 = True
        self.ack_nav2 = msg
    def transformLiDARToCamera(self, pc_lidar):
        cam = self.TransformMat.dot(pc_lidar.T)
        return np.delete(cam, 3, axis=0)

    def transformCameraToImage(self, pc_camera):
        cam = self.CameraMat.dot(pc_camera)
        mask_front = cam[2, :] < 0
        proj = cam.copy()
        np.divide(proj, proj[2, :], out=proj, where=mask_front)
        mask_frame = ((proj[0, :] >= 0) & (proj[0, :] < self.width) &
                      (proj[1, :] >= 0) & (proj[1, :] < self.height))
        final = mask_front & mask_frame
        return proj[:2, final], final


def draw_pts_img(img, xi, yi, distances, threshold):
    out = img.copy()
    for (x, y), d in zip(zip(xi, yi), distances):
        color = (0,255,0) if d >= threshold else (0,0,255)
        cv2.circle(out, (x,y), 2, color, -1)
    return out


if __name__ == '__main__':
    rospy.init_node('ex_calib_2d_yolo', anonymous=True)
    Transformer = LiDARToCameraTransform(parameters_cam, parameters_lidar)
    rate = rospy.Rate(15)

    distance_threshold = 2.5  # 빨간점 임계 거리 (m)
    stop_margin = 10         # 바운딩 박스 주변 정지 감지 영역 (픽셀)

    # YOLOv5 모델 로드
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
        model.conf = 0.5
        rospy.loginfo("YOLOv5 model loaded successfully.")
    except Exception as e:
        rospy.logerr(f"Failed to load YOLOv5 model: {e}")
        rospy.logerr(f"Please check the path: {YOLO_MODEL_PATH}")
        exit()

    while not rospy.is_shutdown():
    
        if Transformer.img is None:
            continue

        yolo_frame = Transformer.img.copy()
        results = model(yolo_frame)
        detections = results.xyxy[0]
    
        # YOLO 박스 그리기
        for *box, conf, cls in detections:
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(yolo_frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(yolo_frame, f"{model.names[int(cls)]} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # LiDAR 투영 및 그리기
        if Transformer.pc_np is not None and Transformer.pc_np.shape[0] > 0:
            xyz_c = Transformer.transformLiDARToCamera(Transformer.pc_np)
            xy_i, mask = Transformer.transformCameraToImage(xyz_c)
            # range mismatch 처리
            if len(mask) != len(Transformer.ranges):
                min_len = min(len(mask), len(Transformer.ranges))
                mask = mask[:min_len]
                filtered_ranges = Transformer.ranges[:min_len][mask]
            else:
                filtered_ranges = Transformer.ranges[mask]

            if xy_i.shape[1] > 0:
                xy_i = xy_i.astype(np.int32)
                projectionImage = draw_pts_img(yolo_frame, xy_i[0,:], xy_i[1,:], filtered_ranges, distance_threshold)

                # 정지 조건 검사
                if Transformer.is_ack_nav2:
                    for (px,py), d in zip(zip(xy_i[0,:], xy_i[1,:]), filtered_ranges):
                        if d < distance_threshold:
                            for *box, conf, cls in detections:
                                bx1,by1,bx2,by2 = map(int, box)
                                if bx1-stop_margin <= px <= bx2+stop_margin and by1-stop_margin <= py <= by2+stop_margin:
                                    ack_msg = AckermannDriveStamped()
                                    ack_msg.drive.speed = 0.0  # 정지 속도
                                    ack_msg.drive.steering_angle = Transformer.ack_nav2.drive.steering_angle  # 조향 각도
                                    Transformer.ack_pub.publish(ack_msg)
                                    print("정지!!!!!!!!!!!")
                                    break
                            else:
                                continue
                            break
            else:
                projectionImage = yolo_frame
        else:
            projectionImage = yolo_frame

        cv2.imshow("YOLOv5 + 2D Lidar to Camera Projection", projectionImage)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        rate.sleep()

    cv2.destroyAllWindows()
