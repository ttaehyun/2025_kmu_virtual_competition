"""
focal_length_tuner.py – LiDAR-to-Camera 투영 테스트용 초점거리 실시간 조정 툴
작성: 2025-08-06
"""

import threading, math, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import LaserScan, CompressedImage
from numpy.linalg import inv
from tkinter import Tk, Entry, Button, Label

# ---------- 카메라·LiDAR 파라미터 ----------
CAM_PARAMS = dict(WIDTH=640, HEIGHT=480, X=0.30, Y=0, Z=0.11,
                  YAW=0, PITCH=0.0, ROLL=0)
LIDAR_PARAMS = dict(X=0.11, Y=0, Z=0.13, YAW=0, PITCH=0, ROLL=0)

# --------- 좌표 변환 유틸리티 ----------
def rot_mat(rpy):
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0, math.cos(r),-math.sin(r)],[0, math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(p),0, math.sin(p)],[0,1,0],[-math.sin(p),0, math.cos(p)]])
    Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def make_T(params_cam, params_lidar):
    cam_RPY = np.array([params_cam['ROLL'], params_cam['PITCH'], params_cam['YAW']]) \
              + np.deg2rad([-90, 0, -90])
    lidar_RPY = np.array([params_lidar['ROLL'], params_lidar['PITCH'], params_lidar['YAW']])

    T_cam = np.eye(4);  T_lidar = np.eye(4)
    T_cam[:3,:3]   = rot_mat(cam_RPY)
    T_cam[:3, 3]   = [params_cam['X'], params_cam['Y'], params_cam['Z']]
    T_lidar[:3,:3] = rot_mat(lidar_RPY)
    T_lidar[:3,3]  = [params_lidar['X'], params_lidar['Y'], params_lidar['Z']]
    return inv(T_cam) @ T_lidar          # lidar → cam

# ---------- 변환 클래스 ----------
class Projector:
    def __init__(self, cam_params, lidar_params, init_f=250.0):
        self.width, self.height = cam_params['WIDTH'], cam_params['HEIGHT']
        self.T = make_T(cam_params, lidar_params)
        self.set_focal(init_f)

        self.scan_pts = None
        self.scan_ranges = None
        self.img = None

        rospy.Subscriber('/lidar2D', LaserScan,  self.scan_cb,  queue_size=1)
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage,
                         self.img_cb,   queue_size=1)

    # --- ROS 콜백 ---
    def img_cb(self, msg):
        self.img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    def scan_cb(self, msg):
        pts, rng = [], []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and not (math.isinf(r) or math.isnan(r)):
                pts.append([r*math.cos(angle), r*math.sin(angle), 0, 1])
                rng.append(r)
            angle += msg.angle_increment
        self.scan_pts    = np.asarray(pts, dtype=np.float32)
        self.scan_ranges = np.asarray(rng, dtype=np.float32)

    # --- 투영 계산 ---
    def lidar_to_cam(self, pts4):
        return (self.T @ pts4.T)[:3]  # 3×N
    def cam_to_img(self, pc_cam):
        cam = self.K @ pc_cam
        mask = cam[2] < 0
        proj = cam.copy()
        np.divide(proj, proj[2], out=proj, where=mask)
        in_img = mask & (proj[0]>=0) & (proj[0]<self.width) & (proj[1]>=0) & (proj[1]<self.height)
        return proj[:2, in_img].astype(np.int32), in_img

    # --- 초점거리 업데이트 ---
    def set_focal(self, f):
        self.K = np.array([[f,0,self.width/2], [0,f,self.height/2], [0,0,1]], dtype=np.float32)

# ---------- UI 스레드 ----------
def start_ui(proj):
    def apply():
        try:
            f = float(entry.get())
            proj.set_focal(f)
        except ValueError:
            pass
    root = Tk(); root.title('Focal Length (fx=fy) Tuner')
    Label(root,text='fx = fy').pack(pady=4)
    entry = Entry(root, width=10); entry.insert(0,'250'); entry.pack()
    Button(root,text='Apply', command=apply).pack(pady=4)
    root.mainloop()

# ---------- 메인 ----------
def main():
    rospy.init_node('focal_length_tuner', anonymous=True)
    proj = Projector(CAM_PARAMS, LIDAR_PARAMS)

    # UI를 별도 스레드로 실행
    threading.Thread(target=start_ui, args=(proj,), daemon=True).start()

    r = rospy.Rate(15)
    while not rospy.is_shutdown():
        if proj.img is None:
            r.sleep();  continue

        frame = proj.img.copy()
        if proj.scan_pts is not None and proj.scan_pts.size:
            pc_cam = proj.lidar_to_cam(proj.scan_pts)
            xy_img, mask = proj.cam_to_img(pc_cam)
            ranges = proj.scan_ranges[mask]
            for (x,y), d in zip(xy_img.T, ranges):
                color = (0,255,0) if d>2.5 else (0,0,255)
                cv2.circle(frame, (x,y), 2, color, -1)

        cv2.imshow('Projection (fx={:.1f})'.format(proj.K[0,0]), frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC 로 종료
            break
        r.sleep()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
