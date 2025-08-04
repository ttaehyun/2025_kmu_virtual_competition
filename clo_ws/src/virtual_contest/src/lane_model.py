#!/usr/bin/env python3
import sys
sys.path.append('/home/a/Ultra-Fast-Lane-Detection')  # UFLD 루트 경로
sys.argv += ['/home/a/Ultra-Fast-Lane-Detection/configs/tusimple.py']

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from torchvision import transforms
from model.model import parsingNet
from utils.common import merge_config
import scipy.special
from PIL import Image
from data.constant import tusimple_row_anchor  # row_anchor 직접 불러오기

# 설정 및 모델 로딩
args, cfg = merge_config()
cfg.row_anchor = tusimple_row_anchor
cls_num_per_lane = len(cfg.row_anchor)

net = parsingNet(pretrained=False, backbone=cfg.backbone,
                 cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                 use_aux=True).cuda()
state_dict = torch.load(cfg.test_model, map_location='cuda')['model']
net.load_state_dict({k[7:] if 'module.' in k else k: v for k, v in state_dict.items()})
net.eval()

# 전처리 transform
transform = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

bridge = CvBridge()

colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]  # 차선별 색

def callback(msg):
    # ROS 이미지 → OpenCV 이미지 → PIL 이미지 변환
    img_raw = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    img_resized = cv2.resize(img_raw, (800, 288))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).cuda()

    with torch.no_grad():
        out = net(img_tensor)
        if isinstance(out, tuple):  # use_aux=True일 때는 (lane_output, seg_output) 형태
            out = out[0]

    out = out[0].data.cpu().numpy()  # shape: (griding_num+1, cls_per_lane, num_lanes)
    out = out[:, ::-1, :]  # flip vertically
    # print(f"[DEBUG] out.shape: {out.shape}")
    # softmax + weighted sum
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)  # exclude last row (background)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)

    argmax_out = np.argmax(out, axis=0)  # shape: (cls_num_per_lane, num_lanes)
    mask = argmax_out == cfg.griding_num
    loc[mask] = 0  # invalid한 점 제거

    # 시각화
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # for i in range(loc.shape[1]):  # num_lanes
    #     for j in range(loc.shape[0]):  # cls_num_per_lane
    #         if loc[j, i] > 0:
    #             x = int(loc[j, i] * col_sample_w * img_raw.shape[1] / 800)
    #             y = int(img_raw.shape[0] * (cfg.row_anchor[cls_num_per_lane - 1 - j] / 288))
    #             cv2.circle(img_raw, (x, y), 5, (0, 255, 0), -1)

    for i in range(loc.shape[1]):  # num_lanes
        lane_points = []
        for j in range(loc.shape[0]):  # cls_num_per_lane
            if loc[j, i] > 0:
                x = int(loc[j, i] * col_sample_w * img_raw.shape[1] / 800)
                y = int(img_raw.shape[0] * (cfg.row_anchor[cls_num_per_lane - 1 - j] / 288))
                lane_points.append((x, y))

        if len(lane_points) >= 2:
            cv2.polylines(img_raw, [np.array(lane_points, dtype=np.int32)], isClosed=False, color=colors[i % len(colors)], thickness=3)

    cv2.imshow("Lane Detection", img_raw)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('ufld_lane_detector')
    rospy.Subscriber('/image_jpeg/compressed', CompressedImage, callback)
    rospy.spin()
