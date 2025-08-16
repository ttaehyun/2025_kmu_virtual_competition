#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import LaserScan, CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from numpy.linalg import inv
import torch
import message_filters
from sklearn.cluster import DBSCAN
import tf2_ros
from tf.transformations import quaternion_matrix

# 사용자 정의: YOLOv5 모델(.pt 파일)의 절대 경로를 지정하세요.
YOLO_MODEL_PATH = '/media/a/SSD/virtual_competition/best.pt'

# 파라미터
parameters_cam = {
    "WIDTH": 640, "HEIGHT": 480, "X": 0.30, "Y": 0, "Z": 0.11,
    "YAW": 0, "PITCH": 0.0, "ROLL": 0
}
parameters_lidar = {
    "X": 0.11, "Y": 0, "Z": 0.13,
    "YAW": 0, "PITCH": 0, "ROLL": 0
}

def tf_to_mat(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])  # 4x4
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T

def transform_points_np(pts_xyz, T):  # pts: (N,3)
    if pts_xyz.size == 0:
        return pts_xyz
    ones = np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)
    pts_h = np.hstack([pts_xyz, ones])            # (N,4)
    out = (T.dot(pts_h.T)).T[:, :3]               # (N,3)
    return out

def transform_point_np(p_xyz, T):  # p: (3,)
    ph = np.array([p_xyz[0], p_xyz[1], p_xyz[2], 1.0], dtype=np.float64)
    out = T.dot(ph)[:3]
    return out

# (getRotMat, getTransformMat, getCameraMat, LiDARToCameraTransform, draw_pts_img 함수는 이전과 동일)
def getRotMat(RPY):
    cosR, sinR = math.cos(RPY[0]), math.sin(RPY[0])
    cosP, sinP = math.cos(RPY[1]), math.sin(RPY[1])
    cosY, sinY = math.cos(RPY[2]), math.sin(RPY[2])
    rotRoll  = np.array([[1,0,0], [0,cosR,-sinR], [0,sinR,cosR]])
    rotPitch = np.array([[cosP,0,sinP], [0,1,0], [-sinP,0,cosP]])
    rotYaw   = np.array([[cosY,-sinY,0], [sinY,cosY,0], [0,0,1]])
    return rotYaw.dot(rotPitch.dot(rotRoll))

def getTransformMat(params_cam, params_lidar):
    lidarPosition = np.array([params_lidar[i] for i in ("X","Y","Z")])
    camPosition   = np.array([params_cam[i] for i in ("X","Y","Z")])
    lidarRPY      = np.array([params_lidar[i] for i in ("ROLL","PITCH","YAW")])
    camRPY        = np.array([params_cam[i] for i in ("ROLL","PITCH","YAW")])
    camRPY += np.array([-90 * math.pi/180, 0, -90 * math.pi/180])

    camRot   = getRotMat(camRPY)
    Tr_cam_to_vehicle   = np.eye(4)
    Tr_cam_to_vehicle[:3,:3] = camRot
    Tr_cam_to_vehicle[:3, 3] = camPosition

    lidarRot  = getRotMat(lidarRPY)
    Tr_lidar_to_vehicle = np.eye(4)
    Tr_lidar_to_vehicle[:3,:3] = lidarRot
    Tr_lidar_to_vehicle[:3, 3] = lidarPosition

    return np.linalg.inv(Tr_cam_to_vehicle).dot(Tr_lidar_to_vehicle)

def getCameraMat(params_cam):
    return np.array([[230, 0., 320.], [0., 230, 240.], [0., 0., 1.]])

class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):
        self.pc_np = None
        self.img = None
        self.ranges = None
        self.header = None
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]

        img_sub   = message_filters.Subscriber("/image_jpeg/compressed", CompressedImage)
        lidar_sub = message_filters.Subscriber("/lidar2D", LaserScan)
        ats = message_filters.ApproximateTimeSynchronizer(
            [img_sub, lidar_sub], queue_size=10, slop=0.1
        )
        ats.registerCallback(self.sync_callback)

        self.TransformMat = getTransformMat(params_cam, params_lidar)
        self.CameraMat = getCameraMat(params_cam)

    def sync_callback(self, img_msg, scan_msg):
        self.header = scan_msg.header
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        pts, rngs = [], []
        angle = scan_msg.angle_min
        for r in scan_msg.ranges:
            if scan_msg.range_min < r < scan_msg.range_max:
                pts.append([r*math.cos(angle), r*math.sin(angle), 0.0, 1.0])
                rngs.append(r)
            angle += scan_msg.angle_increment
        self.pc_np = np.array(pts, dtype=np.float32) if pts else None
        self.ranges = np.array(rngs, dtype=np.float32) if rngs else None

    def transformLiDARToCamera(self, pc_lidar):
        return self.TransformMat.dot(pc_lidar.T)[:3, :]

    def transformCameraToImage(self, pc_camera):
        cam_pts = self.CameraMat.dot(pc_camera)
        mask_front = cam_pts[2, :] < -0.05

        proj = cam_pts.copy()
        np.divide(proj, proj[2, :], out=proj, where=mask_front)

        mask_frame = (proj[0, :] >= 0) & (proj[0, :] < self.width) & \
                     (proj[1, :] >= 0) & (proj[1, :] < self.height)

        final_mask = mask_front & mask_frame
        return proj[:2, final_mask], final_mask

def draw_pts_img(img, xi, yi, distances, threshold):
    out = img.copy()
    for (x, y), d in zip(zip(xi, yi), distances):
        color = (0, 255, 0) if d >= threshold else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 2, color, -1)
    return out


if __name__ == '__main__':
    rospy.init_node('lidar_cam_fusion_node', anonymous=True)
    Transformer = LiDARToCameraTransform(parameters_cam, parameters_lidar)
    pc_pub = rospy.Publisher('/colored_lidar_points', PointCloud2, queue_size=1)
    marker_pub = rospy.Publisher('/bus_markers', MarkerArray, queue_size=1)
    rate = rospy.Rate(15)
    MARKER_TARGET_FRAME = rospy.get_param('~marker_frame', 'base_link')

    # ★ 추가: tf2 버퍼/리스너
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # ---- 파라미터 ----
    distance_threshold = 2.0
    stop_margin = 10
    stationary_time_threshold = 2.0
    pixel_movement_epsilon = 2.0
    last_positions = {}

    target_class_names_for_viz = {"person_cycle", "bus"}
    class_colors = {"person_cycle": (0, 255, 0), "bus": (255, 0, 0)}
    color_change_distance_threshold = 5.0

    # 버스 트래킹 관련 파라미터
    NUM_BUSES_TO_TRACK = 2
    dbscan_eps = 0.5
    dbscan_min_samples = 3
    TRACKING_MAX_DIST = 2.0
    TRACKING_MAX_AGE_SEC = 2.0
    STATIONARY_EPSILON_M = 0.05
    # ★★★ 추가: 마커가 처음 생성될 수 있는 최대 거리 (미터) ★★★
    MARKER_CREATION_MAX_DIST = 5.0

    tracked_buses = {}
    next_track_id = 0

    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)
        model.conf = 0.5
        rospy.loginfo("YOLOv5 model loaded successfully.")
    except Exception as e:
        rospy.logerr(f"Failed to load YOLOv5 model: {e}")
        exit()

    while not rospy.is_shutdown():
        if Transformer.img is None or Transformer.pc_np is None:
            rate.sleep()
            continue

        current_time = rospy.get_time()
        frame = Transformer.img.copy()
        pc_lidar = Transformer.pc_np.copy()
        lidar_ranges = Transformer.ranges.copy()
        lidar_header = Transformer.header

        results = model(frame)
        detections = results.xyxy[0]

        xyz_c = Transformer.transformLiDARToCamera(pc_lidar)
        xy_i, mask = Transformer.transformCameraToImage(xyz_c)
        filtered_ranges = lidar_ranges[mask]

        if xy_i is not None and filtered_ranges is not None and xy_i.shape[1] > 0:
            frame = draw_pts_img(frame, xy_i[0, :], xy_i[1, :], filtered_ranges, distance_threshold)

        colors_rgb = np.full((pc_lidar.shape[0], 3), 255, dtype=np.uint8)
        original_indices = np.where(mask)[0]

        seen_ids = set()

        bus_clusters = []
        for det_idx, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            cls_name = model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            proj_pts_in_bbox_mask = (xy_i[0, :] >= x1) & (xy_i[0, :] <= x2) & \
                                    (xy_i[1, :] >= y1) & (xy_i[1, :] <= y2)
            original_indices_in_bbox = original_indices[proj_pts_in_bbox_mask]

            if cls_name in target_class_names_for_viz:
                distances_of_points_in_bbox = lidar_ranges[original_indices_in_bbox]
                distance_mask = distances_of_points_in_bbox < color_change_distance_threshold
                final_indices_to_color = original_indices_in_bbox[distance_mask]
                if len(final_indices_to_color) > 0:
                    colors_rgb[final_indices_to_color] = class_colors[cls_name]
            
            if cls_name == 'bus' and len(original_indices_in_bbox) > dbscan_min_samples:
                points_in_box = pc_lidar[original_indices_in_bbox, :3]
                clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points_in_box)
                labels = clustering.labels_
                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                if len(unique_labels) > 0:
                    largest_cluster_label = unique_labels[np.argmax(counts)]
                    cluster_points = points_in_box[labels == largest_cluster_label]
                    bus_clusters.append({'centroid': np.mean(cluster_points, axis=0), 'points': cluster_points})

            if cls_name == "person_cycle":
                obj_id = det_idx
                seen_ids.add(obj_id)
                found_close_point = False
                stationary = False

                for i, ((px, py), d) in enumerate(zip(xy_i.T, filtered_ranges)):
                    if d < distance_threshold and \
                       (x1 - stop_margin <= px <= x2 + stop_margin) and \
                       (y1 - stop_margin <= py <= y2 + stop_margin):
                        
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
                    if stationary: print("그냥가!!!!!")
                    else: print("정지!!!!!!!!!!!")

        stale_ids = set(last_positions.keys()) - seen_ids
        for sid in stale_ids: del last_positions[sid]

        matched_track_ids = set()

        for track_id, track_data in tracked_buses.items():
            min_dist = float('inf')
            best_match_idx = -1
            for i, cluster in enumerate(bus_clusters):
                dist = np.linalg.norm(track_data['centroid'] - cluster['centroid'])
                if dist < TRACKING_MAX_DIST and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                matched_cluster = bus_clusters.pop(best_match_idx)
                tracked_buses[track_id]['previous_centroid'] = tracked_buses[track_id]['centroid']
                tracked_buses[track_id]['centroid'] = matched_cluster['centroid']
                tracked_buses[track_id]['points'] = matched_cluster['points']
                tracked_buses[track_id]['last_seen'] = current_time
                matched_track_ids.add(track_id)

        # ★★★ 변경: 새 트랙 생성 시 거리 제한 조건 추가 ★★★
        if len(bus_clusters) > 0 and len(tracked_buses) < NUM_BUSES_TO_TRACK:
            for cluster in bus_clusters:
                # 라이다 원점(0,0,0)으로부터의 거리 확인
                dist_from_origin = np.linalg.norm(cluster['centroid'])
                
                # 설정된 거리 내에 있을 때만 새 트랙으로 추가
                if dist_from_origin < MARKER_CREATION_MAX_DIST:
                    tracked_buses[next_track_id] = {
                        'centroid': cluster['centroid'],
                        'previous_centroid': cluster['centroid'],
                        'points': cluster['points'],
                        'last_seen': current_time
                    }
                    next_track_id += 1
                    if len(tracked_buses) >= NUM_BUSES_TO_TRACK:
                        break
        
        expired_ids = [tid for tid, tdata in tracked_buses.items() if current_time - tdata['last_seen'] > TRACKING_MAX_AGE_SEC]
        for tid in expired_ids:
            del tracked_buses[tid]
        marker_array = MarkerArray()# ★ 추가: 라이다 프레임 → MARKER_TARGET_FRAME 변환 가져오기
        have_tf = False
        T_lidar_to_base = np.eye(4)
        if lidar_header:
            src_frame = lidar_header.frame_id
            try:
                tf_stamped = tf_buffer.lookup_transform(
                    MARKER_TARGET_FRAME,  # target
                    src_frame,            # source
                    lidar_header.stamp,
                    rospy.Duration(0.1)
                )
                T_lidar_to_base = tf_to_mat(tf_stamped)
                have_tf = True
            except Exception as e:
                rospy.logwarn_throttle(5.0, "TF not ready for %s -> %s: %s",
                                       src_frame, MARKER_TARGET_FRAME, str(e))
        
        for track_id, track_data in tracked_buses.items():
            movement = np.linalg.norm(track_data['centroid'] - track_data['previous_centroid'])
            if movement < STATIONARY_EPSILON_M:
                continue

            centroid_lidar = track_data['centroid']          # (3,)
            points_lidar   = track_data['points']            # (N,3)

            # ★ 변환 적용: base_link 좌표계로
            if have_tf:
                centroid_base = transform_point_np(centroid_lidar, T_lidar_to_base)
                points_base   = transform_points_np(points_lidar, T_lidar_to_base)
            else:
                centroid_base = centroid_lidar
                points_base   = points_lidar
                
            marker = Marker()
            
            marker.header.frame_id = MARKER_TARGET_FRAME
            marker.header.stamp = lidar_header.stamp if lidar_header else rospy.Time.now()
            marker.ns = "bus_tracker"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(centroid_base[0])
            marker.pose.position.y = float(centroid_base[1])
            marker.pose.position.z = float(centroid_base[2])
            marker.pose.orientation.w = 1.0

            if points_base.size > 0:
                ptp = np.ptp(points_base, axis=0)
                marker.scale.x = max(float(ptp[0]), 0.3)
                marker.scale.y = max(float(ptp[1]), 0.3)
                marker.scale.z = max(float(ptp[2]), 0.3)
            else:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.3

           
            marker.color.a = 0.7
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0

            marker.lifetime = rospy.Duration(TRACKING_MAX_AGE_SEC)
            marker_array.markers.append(marker)
        
        marker_pub.publish(marker_array)

        if lidar_header:
            r, g, b = colors_rgb[:, 0].astype(np.uint32), colors_rgb[:, 1].astype(np.uint32), colors_rgb[:, 2].astype(np.uint32)
            rgb_packed = np.left_shift(r, 16) | np.left_shift(g, 8) | b
            rgb_packed_float = rgb_packed.copy().view(np.float32)

            points_with_color = np.hstack([pc_lidar[:, :3], rgb_packed_float.reshape(-1, 1)])
            fields = [PointField('x', 0, 7, 1), PointField('y', 4, 7, 1), PointField('z', 8, 7, 1), PointField('rgb', 12, 7, 1)]
            pc_msg = point_cloud2.create_cloud(lidar_header, fields, points_with_color)
            pc_pub.publish(pc_msg)

        cv2.imshow("YOLOv5 + 2D Lidar (Fusion)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        rate.sleep()

    cv2.destroyAllWindows()
