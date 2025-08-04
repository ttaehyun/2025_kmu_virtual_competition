#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class CompressedImageSaver:
    def __init__(self):
        rospy.init_node('compressed_image_saver', anonymous=True)

        # 사용자 설정
        self.image_topic = rospy.get_param('~image_topic', '/image_jpeg/compressed')
        self.save_dir = rospy.get_param('~save_dir', '/home/a/compressed_images')
        self.frame_interval = rospy.get_param('~frame_interval', 10)

        os.makedirs(self.save_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.count = 0
        self.save_count = 0

        rospy.Subscriber(self.image_topic, CompressedImage, self.callback, queue_size=10)
        rospy.loginfo(f"✅ 구독 시작: {self.image_topic}")
        rospy.spin()

    def callback(self, msg):
        if self.count % self.frame_interval != 0:
            self.count += 1
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            filename = os.path.join(self.save_dir, f"frame_{self.save_count:05d}.jpg")
            cv2.imwrite(filename, cv_img)
            rospy.loginfo(f"✔ 저장됨: {filename}")
            self.save_count += 1

        except Exception as e:
            rospy.logwarn(f"❌ 오류 발생 (frame {self.count}): {e}")

        self.count += 1

if __name__ == '__main__':
    try:
        CompressedImageSaver()
    except rospy.ROSInterruptException:
        pass
