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
import torch
import message_filters
import sys
from pathlib import Path

# =========================
# YOLO 경로/설정 (오프라인)
# =========================
# ★ 여기를 실제 YOLOv5 레포지토리 경로로 바꿔주세요
YOLO_DIR = "/home/a/2025_kmu_virtual_competition/yolov5"  # ex) /.../virtual_contest/yolov5
YOLO_MODEL_PATH = "/home/a/2025_kmu_virtual_competition/clo_ws/src/virtual_contest/model/best.pt"

# 추론 하이퍼파라미터
IMG_SIZE = (640, 640)   # 학습/추론 해상도
CONF_THRES = 0.5
IOU_THRES  = 0.45
MAX_DET    = 1000

# 대상 클래스명 (YOLO 모델의 names와 일치)
TARGET_CLASS_NAME = "person_cycle"

# 카메라/라이다 파라미터
parameters_cam = {
    "WIDTH": 640,
    "HEIGHT": 480,
    "X": 0.30,
    "Y": 0,
    "Z": 0.11,
    "YAW": 0,
    "PITCH": 0.0,
    "ROLL": 0
}
parameters_lidar = {
    "X": 0.11,
    "Y": 0,
    "Z": 0.13,
    "YAW": math.pi,  #### 라이다 
    "PITCH": 0,
    "ROLL": 0
}

# =========================
# 수학/변환 유틸
# =========================
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
    # 카메라 좌표계 보정
    camRPY += np.array([-90 * math.pi/180, 0, -90 * math.pi/180])
    camRot   = getRotMat(camRPY)
    camTrans = np.array([camPosition])
    Tr_cam_to_vehicle   = np.vstack((np.hstack((camRot, camTrans.T)), [0,0,0,1]))
    lidarRot  = getRotMat(lidarRPY)
    lidarTrans= np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.vstack((np.hstack((lidarRot, lidarTrans.T)), [0,0,0,1]))
    return inv(Tr_cam_to_vehicle).dot(Tr_lidar_to_vehicle).round(6)

def getCameraMat(params_cam):
    return np.array([[230, 0., 320.],
                     [0., 230, 240.],
                     [0., 0., 1.]])

# =========================
# LiDAR → Camera 투영
# =========================
class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):
        self.pc_np = None
        self.img = None
        self.ranges = None
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]

        img_sub   = message_filters.Subscriber("/image_jpeg/compressed", CompressedImage)
        lidar_sub = message_filters.Subscriber("/lidar2D", LaserScan)
        rospy.Subscriber("/ackermann_cmd_mux/input/nav_2", AckermannDriveStamped, self.ack_callback)
        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, queue_size=1)

        ats = message_filters.ApproximateTimeSynchronizer(
            [img_sub, lidar_sub], queue_size=10, slop=0.1, allow_headerless=False
        )
        ats.registerCallback(self.sync_callback)

        self.TransformMat = getTransformMat(params_cam, params_lidar)
        self.CameraMat = getCameraMat(params_cam)

        self.ack_nav2 = AckermannDriveStamped()
        self.is_ack_nav2 = False

    def ack_callback(self, msg):
        if not self.is_ack_nav2:
            self.is_ack_nav2 = True
        self.ack_nav2 = msg

    def sync_callback(self, img_msg, scan_msg):
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
        mask_front = cam[2, :] < 0  ########################################
        proj = cam.copy()
        np.divide(proj, proj[2, :], out=proj, where=mask_front)
        mask_frame = ((proj[0, :] >= 0) & (proj[0, :] < self.width) &
                      (proj[1, :] >= 0) & (proj[1, :] < self.height))
        final = mask_front & mask_frame
        return proj[:2, final], final

# =========================
# 시각화
# =========================
def draw_pts_img(img, xi, yi, distances, threshold):
    out = img.copy()
    for (x, y), d in zip(zip(xi, yi), distances):
        color = (0,255,0) if d >= threshold else (0,0,255)
        cv2.circle(out, (x,y), 2, color, -1)
    return out

# =========================
# 메인
# =========================
if __name__ == '__main__':
    rospy.init_node('ex_calib_2d_yolo', anonymous=True)
    Transformer = LiDARToCameraTransform(parameters_cam, parameters_lidar)
    # Transformer.is_ack_nav2 = True  ####################상위 제어기?
    rate = rospy.Rate(15)

    # ---- 제어 파라미터 ----
    distance_threshold = 1.2       # (m)
    stop_margin = 5               # (px)
    stationary_time_threshold = 2.0     # (s)
    pixel_movement_epsilon = 2.0        # (px)

    last_positions = {}  # {obj_id: (px, py, start_time_of_stationary_period)}

    # =========================
    # YOLOv5 오프라인 로드 (FP32 고정)
    # =========================
    try:
        if YOLO_DIR not in sys.path:
            sys.path.append(YOLO_DIR)

        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # GPU 있으면 CUDA 사용
        model = DetectMultiBackend(YOLO_MODEL_PATH, device=device, dnn=False)

        # 🔧 FP16 완전 비활성화 + 모델을 float32로 고정
        model.fp16 = False
        if hasattr(model, 'model'):
            model.model.float()

        stride, names = int(model.stride), model.names
        model.warmup(imgsz=(1, 3, *IMG_SIZE))  # CUDA면 GPU에서 워밍업 수행

        rospy.loginfo("YOLOv5 (offline, FP32) loaded successfully.")

    except Exception as e:
        rospy.logerr(f"Failed to load YOLOv5 model offline: {e}")
        rospy.logerr(f"Check YOLO_DIR: {YOLO_DIR}")
        rospy.logerr(f"Check weights: {YOLO_MODEL_PATH}")
        raise SystemExit

    while not rospy.is_shutdown():
        if Transformer.img is None:
            rate.sleep()
            continue

        frame = Transformer.img.copy()

        # =========================
        # LiDAR → 이미지 투영
        # =========================
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

        if xy_i is not None and filtered_ranges is not None and xy_i.shape[1] > 0:
            frame = draw_pts_img(frame,
                                 xy_i[0, :].astype(np.int32),
                                 xy_i[1, :].astype(np.int32),
                                 filtered_ranges,
                                 distance_threshold)

        # =========================
        # YOLO 추론 (FP32, GPU/CPU 자동)
        # =========================
        # 1) 전처리
        im = letterbox(frame, IMG_SIZE, stride=stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()            # ★ 항상 FP32
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # 2) 추론 + NMS
        with torch.no_grad():
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred,
                                       conf_thres=CONF_THRES,
                                       iou_thres=IOU_THRES,
                                       classes=None,
                                       agnostic=False,
                                       max_det=MAX_DET)

        # 이번 프레임에서 관측된 obj_id 집합
        seen_ids = set()

        # 3) 박스 후처리 및 로직
        for det in pred:  # det: [N,6] (x1,y1,x2,y2,conf,cls)
            if len(det):
                # 모델 입력 크기 -> 원본 프레임 크기로 스케일 복원
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                for det_idx, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_name = names[int(cls.item())] if isinstance(names, (list, tuple)) else names[int(cls.item())]

                    # 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(frame, f"{cls_name} {float(conf):.2f}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    # 대상 클래스만 처리
                    if cls_name != TARGET_CLASS_NAME:
                        continue

                    obj_id = det_idx
                    seen_ids.add(obj_id)

                    # LiDAR 포인트 없으면 판단 불가
                    if xy_i is None or filtered_ranges is None or xy_i.shape[1] == 0:
                        continue

                    # 바운딩박스(+margin) & 거리 임계 내 포인트 중 하나 선택
                    found_close_point = False
                    stationary = False
                    current_time = time.time()

                    if Transformer.is_ack_nav2:
                        for (px, py), d in zip(
                            zip(xy_i[0, :].astype(np.int32), xy_i[1, :].astype(np.int32)),
                            filtered_ranges
                        ):
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
                            ack_msg.drive.speed = 0.0  # 정지
                            ack_msg.drive.steering_angle = Transformer.ack_nav2.drive.steering_angle
                            Transformer.ack_pub.publish(ack_msg)
                            print("정지!!!!!!!!!!!")

        # 프레임에서 사라진 객체의 상태 제거
        stale_ids = set(last_positions.keys()) - seen_ids
        for sid in stale_ids:
            del last_positions[sid]

        # 디스플레이
        cv2.imshow("YOLOv5 + 2D Lidar to Camera Projection (Synced, Offline FP32)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        rate.sleep()

    cv2.destroyAllWindows()
