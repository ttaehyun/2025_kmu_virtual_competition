#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import LaserScan, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from numpy.linalg import inv
import torch  # YOLOv5를 위해 추가
import message_filters  # ★ 추가: 동기화용

# 사용자 정의: YOLOv5 모델(.pt 파일)의 절대 경로를 지정하세요.
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
    cosR = math.cos(RPY[0]); sinR = math.sin(RPY[0])
    cosP = math.cos(RPY[1]); sinP = math.sin(RPY[1])
    cosY = math.cos(RPY[2]); sinY = math.sin(RPY[2])
    rotRoll  = np.array([1,0,0, 0,cosR,-sinR, 0,sinR,cosR]).reshape(3,3)
    rotPitch = np.array([cosP,0,sinP, 0,1,0, -sinP,0,cosP]).reshape(3,3)
    rotYaw   = np.array([cosY,-sinY,0, sinY,cosY,0, 0,0,1]).reshape(3,3)
    return rotYaw.dot(rotPitch.dot(rotRoll))

def getTransformMat(params_cam, params_lidar):
    lidarPosition = np.array([params_lidar.get(i) for i in ("X","Y","Z")])
    camPosition   = np.array([params_cam.get(i) for i in ("X","Y","Z")])
    lidarRPY      = np.array([params_lidar.get(i) for i in ("ROLL","PITCH","YAW")])
    camRPY        = np.array([params_cam.get(i) for i in ("ROLL","PITCH","YAW")])
    camRPY += np.array([-90 * math.pi/180, 0, -90 * math.pi/180])
    camRot   = getRotMat(camRPY)
    camTrans = np.array([camPosition])
    Tr_cam_to_vehicle   = np.vstack((np.hstack((camRot, camTrans.T)), [0,0,0,1]))
    lidarRot  = getRotMat(lidarRPY)
    lidarTrans= np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.vstack((np.hstack((lidarRot, lidarTrans.T)), [0,0,0,1]))
    return inv(Tr_cam_to_vehicle).dot(Tr_lidar_to_vehicle).round(6)

def getCameraMat(params_cam):
    # 제공된 Intrinsic Matrix (핀홀 카메라 모델)
    return np.array([[230, 0., 320.],
                     [0., 230, 240.],
                     [0., 0., 1.]])

class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):
        self.pc_np = None
        self.img = None
        self.ranges = None
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]

        # ★★★ 메시지 동기화 (ApproximateTimeSynchronizer)
        img_sub   = message_filters.Subscriber("/image_jpeg/compressed", CompressedImage)
        lidar_sub = message_filters.Subscriber("/lidar2D", LaserScan)
        rospy.Subscriber("/ackermann_cmd_mux/input/nav_2", AckermannDriveStamped, self.ack_callback)
        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, queue_size=1)
        # queue_size: 내부 버퍼 크기, slop: 허용 시간차(초)
        # 카메라 30Hz, 라이다 10Hz → 0.05~0.1 정도가 보통 잘 맞음
        ats = message_filters.ApproximateTimeSynchronizer(
            [img_sub, lidar_sub], queue_size=10, slop=0.1, allow_headerless=False
        )
        ats.registerCallback(self.sync_callback)

        self.TransformMat = getTransformMat(params_cam, params_lidar) # lidar -> camera matrix
        self.CameraMat = getCameraMat(params_cam)

        self.ack_nav2 = AckermannDriveStamped()
        self.is_ack_nav2 = False

    def ack_callback(self, msg):
        if not self.is_ack_nav2:
            self.is_ack_nav2 = True
        self.ack_nav2 = msg

    def sync_callback(self, img_msg, scan_msg):
        """카메라-라이다를 타임스탬프 기준으로 동기화한 콜백."""
        # 이미지 디코드
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # LaserScan → 포인트/거리 배열 구성 (기존 scan_callback 내용)
        pts, rngs = [], []
        angle = scan_msg.angle_min
        for r in scan_msg.ranges:
            if not np.isinf(r) and not np.isnan(r) and scan_msg.range_min < r < scan_msg.range_max:
                pts.append([r*math.cos(angle), r*math.sin(angle), 0.0, 1.0])
                rngs.append(r)
            angle += scan_msg.angle_increment
        if pts:
            self.pc_np = np.array(pts, dtype=np.float32)
            self.ranges = np.array(rngs, dtype=np.float32)
        else:
            self.pc_np = None
            self.ranges = None

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

    # ---- 파라미터 ----
    distance_threshold = 1.0       # (m) 빨간점 임계 거리
    stop_margin = 10               # (px) 바운딩 박스 주변 여유
    target_class_name = "person_cycle"  # 대상 클래스명
    stationary_time_threshold = 2.0     # (s) 이 시간 이상 거의 안 움직이면 "그냥가!!!!!"
    pixel_movement_epsilon = 2.0        # (px) 화면상 이동량이 이 값 미만이면 '거의 안 움직임'으로 간주

    # 포인트 정지 판별용 상태 저장: {obj_id: (px, py, start_time_of_stationary_period)}
    last_positions = {}

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
        # 동기화된 세트가 들어올 때까지 대기
        if Transformer.img is None:
            rate.sleep()
            continue

        frame = Transformer.img.copy()
        results = model(frame)
        detections = results.xyxy[0]

        # LiDAR 투영 먼저 계산 (동기화된 동일 시각의 포인트 사용)
        xy_i = None
        filtered_ranges = None
        if Transformer.pc_np is not None and Transformer.pc_np.shape[0] > 0:
            xyz_c = Transformer.transformLiDARToCamera(Transformer.pc_np)
            xy_i, mask = Transformer.transformCameraToImage(xyz_c)
            if Transformer.ranges is not None:
                if len(mask) != len(Transformer.ranges):
                    min_len = min(len(mask), len(Transformer.ranges))
                    mask = mask[:min_len]
                    filtered_ranges = Transformer.ranges[:min_len][mask]
                else:
                    filtered_ranges = Transformer.ranges[mask]

        # 포인트 시각화
        if xy_i is not None and filtered_ranges is not None and xy_i.shape[1] > 0:
            frame = draw_pts_img(frame, xy_i[0, :].astype(np.int32),
                                       xy_i[1, :].astype(np.int32),
                                       filtered_ranges, distance_threshold)

        # 이번 프레임에서 관측된 obj_id 집합(유실된 트랙 정리용)
        seen_ids = set()

        # 각 탐지에 대해 처리
        for det_idx, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            cls_name = model.names[int(cls)]

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # 대상 클래스만 처리
            if cls_name != target_class_name:
                continue

            obj_id = det_idx
            seen_ids.add(obj_id)

            # LiDAR 포인트가 없으면 판단 불가
            if xy_i is None or filtered_ranges is None or xy_i.shape[1] == 0:
                continue

            # 바운딩박스(+margin) & 거리 임계 내 포인트 중 하나 선택
            found_close_point = False
            stationary = False
            current_time = time.time()
            
            if Transformer.is_ack_nav2:
                for (px, py), d in zip(zip(xy_i[0, :].astype(np.int32), xy_i[1, :].astype(np.int32)), filtered_ranges):
                    if d < distance_threshold:
                        if (x1 - stop_margin <= px <= x2 + stop_margin) and (y1 - stop_margin <= py <= y2 + stop_margin):
                            found_close_point = True

                            if obj_id in last_positions:
                                prev_x, prev_y, start_t = last_positions[obj_id]
                                movement = math.hypot(px - prev_x, py - prev_y)

                                if movement < pixel_movement_epsilon:
                                    if current_time - start_t >= stationary_time_threshold:
                                        stationary = True
                                    last_positions[obj_id] = (px, py, start_t)
                                else:
                                    last_positions[obj_id] = (px, py, current_time)
                            else:
                                last_positions[obj_id] = (px, py, current_time)
                            break

            if found_close_point:
                if stationary:
                    print("그냥가!!!!!")
                else:
                    ack_msg = AckermannDriveStamped()
                    ack_msg.drive.speed = 0.0  # 정지 속도
                    ack_msg.drive.steering_angle = Transformer.ack_nav2.drive.steering_angle  # 조향 각도
                    Transformer.ack_pub.publish(ack_msg)
                    print("정지!!!!!!!!!!!")

        # 프레임에서 사라진 객체의 상태 제거
        stale_ids = set(last_positions.keys()) - seen_ids
        for sid in stale_ids:
            del last_positions[sid]

        # 디스플레이
        cv2.imshow("YOLOv5 + 2D Lidar to Camera Projection (Synced)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        rate.sleep()

    cv2.destroyAllWindows()