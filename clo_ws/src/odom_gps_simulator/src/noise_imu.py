#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_about_axis, quaternion_multiply, euler_from_quaternion, quaternion_from_euler

class ImuRepublisher:
    def __init__(self):
        # Publisher
        self.pub = rospy.Publisher('/noise_imu', Imu, queue_size=10)
        # Subscriber
        rospy.Subscriber('/imu', Imu, self.imu_callback)

        # 분산 및 표준편차 설정
        # self.gyro_variance_        = 5e-4       # 각속도 분산
        # self.acceleration_variance_= 0.05       # 가속도 분산
        # self.mean_                 = 0.0
        # self.stddev_vel_           = np.sqrt(self.gyro_variance_)
        # self.stddev_accel_         = np.sqrt(self.acceleration_variance_)
        # self.noise_std_orientation = 0.05

        # 강한 노이즈용 설정
        # self.gyro_variance_        = 0.1   # 각속도 분산 (std ≈ 0.07 rad/s)
        # self.acceleration_variance_= 10.0    # 가속도 분산  (std ≈ 0.71 m/s²)
        # self.mean_                 = 0.0
        # self.stddev_vel_           = np.sqrt(self.gyro_variance_)
        # self.stddev_accel_         = np.sqrt(self.acceleration_variance_)
        # self.noise_std_orientation = 0.3    # std ≈ 17° (완전 강하게)

        # 현실적인 노이즈용 설정
        self.gyro_variance_        = 0.01
        self.acceleration_variance_= 2.0
        self.mean_                 = 0.0
        self.stddev_vel_           = np.sqrt(self.gyro_variance_)
        self.stddev_accel_         = np.sqrt(self.acceleration_variance_)
        self.noise_std_orientation = 0.1

        # IMU 메시지 저장용(공분산 필드 초기화)
        self.imu_data_ = Imu()
        self.imu_data_.angular_velocity_covariance = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, self.gyro_variance_
        ]
        self.imu_data_.linear_acceleration_covariance = [
            self.acceleration_variance_, 0.0, 0.0,
            0.0, self.acceleration_variance_, 0.0,
            0.0, 0.0, self.acceleration_variance_
        ]

        # 드리프트 노이즈 파라미터
        # self.drift_bias_z_        = 0.003     # rad/s·s
        # self.drift_random_walk_z_ = 0.0001    # rad/sqrt(s)
        # self.drift_accum_z_       = 0.0

        # 강한 드리프트 노이즈
        # self.drift_bias_z_        = 0.05    # rad/s·s
        # self.drift_random_walk_z_ = 0.01   # rad/sqrt(s)
        # self.drift_accum_z_       = 0.0

        # 현실적인 드리프트 노이즈
        self.drift_bias_z_        = 0.02    # rad/s·s
        self.drift_random_walk_z_ = 0.002   # rad/sqrt(s)
        self.drift_accum_z_       = 0.0

        # 시간 및 초기 플래그
        self.prev_time_  = rospy.Time.now()
        self.start_flag_ = True

    def imu_callback(self, msg):
        # 시간차 계산
        now = rospy.Time.now()
        dt = (now - self.prev_time_).to_sec()
        if dt < 0.0:
            dt = 0.0
        self.prev_time_ = now

        # 1) 각속도 Z 드리프트 + 가우시안 노이즈
        drift_noise = np.random.normal(0.0, self.drift_random_walk_z_ * np.sqrt(dt))
        self.drift_accum_z_ += drift_noise
        bias_z = self.drift_bias_z_ * dt
        wz_noisy = (msg.angular_velocity.z +
                    bias_z +
                    self.drift_accum_z_ +
                    np.random.normal(self.mean_, self.stddev_vel_))

        # 2) 각속도와 가속도 필드 설정
        self.imu_data_.angular_velocity.x = msg.angular_velocity.x
        self.imu_data_.angular_velocity.y = msg.angular_velocity.y
        self.imu_data_.angular_velocity.z = wz_noisy

        self.imu_data_.linear_acceleration.x = msg.linear_acceleration.x + np.random.normal(self.mean_, self.stddev_accel_)
        self.imu_data_.linear_acceleration.y = msg.linear_acceleration.y + np.random.normal(self.mean_, self.stddev_accel_)
        self.imu_data_.linear_acceleration.z = msg.linear_acceleration.z + np.random.normal(self.mean_, self.stddev_accel_)

        # 3) 쿼터니언 적분으로 orientation 업데이트
        if self.start_flag_:
            # 첫 메시지엔 단위쿼터니언
            self.imu_data_.orientation.x = 0.0
            self.imu_data_.orientation.y = 0.0
            self.imu_data_.orientation.z = 0.0
            self.imu_data_.orientation.w = 1.0
            self.q_orig = np.array([0.0, 0.0, 0.0, 1.0])
            self.start_flag_ = False
        else:
            wx, wy, wz = (self.imu_data_.angular_velocity.x,
                          self.imu_data_.angular_velocity.y,
                          self.imu_data_.angular_velocity.z)
            norm_w = np.linalg.norm([wx, wy, wz])
            if norm_w > 1e-6:
                angle = norm_w * dt
                axis  = np.array([wx, wy, wz]) / norm_w
                q_rot = quaternion_about_axis(angle, axis)
                q_new = quaternion_multiply(self.q_orig, q_rot)
                q_new /= np.linalg.norm(q_new)
                roll, pitch, yaw = euler_from_quaternion([
                    q_new[0], q_new[1], q_new[2], q_new[3]
                ])
                roll  += np.random.normal(0.0, self.noise_std_orientation)
                pitch += np.random.normal(0.0, self.noise_std_orientation)
                yaw   += np.random.normal(0.0, self.noise_std_orientation)
                q_noisy = quaternion_from_euler(roll, pitch, yaw)
                q_noisy /= np.linalg.norm(q_noisy)
                self.q_orig = np.array(q_noisy)
                self.imu_data_.orientation.x = q_noisy[0]
                self.imu_data_.orientation.y = q_noisy[1]
                self.imu_data_.orientation.z = q_noisy[2]
                self.imu_data_.orientation.w = q_noisy[3]

        # 4) orientation 공분산 (yaw 방향만 증분)
        orientation_var_yaw = self.gyro_variance_ * dt * dt * 200 # 10
        self.imu_data_.orientation_covariance = [
            1e-6, 0.0,    0.0,
            0.0,   1e-6,  0.0,
            0.0,   0.0, orientation_var_yaw
        ]

        # 5) 헤더 갱신 후 퍼블리시
        self.imu_data_.header.stamp    = now
        self.imu_data_.header.frame_id = msg.header.frame_id
        self.pub.publish(self.imu_data_)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('imu_republisher', anonymous=True)
    node = ImuRepublisher()
    rospy.loginfo("IMU Republisher Node Started")
    node.spin()
