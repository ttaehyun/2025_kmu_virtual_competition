#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, os
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64

class PIDController:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class control:
    def __init__(self):
        ### 노드 초기화 ###
        rospy.init_node('webot_main', anonymous=True)

        ### nodelet에서 출력한 제어값 입력받음 ###
        rospy.Subscriber("/high_level/ackermann_cmd_mux/target_control", AckermannDriveStamped, self.target_callback)
        self.is_target_input = False
        self.target_input = None
        ####################################

        ### 현재 webot 속도 ###
        # rospy.Subscriber("current_speed", Float32, self.current_speed_callback)
        # self.is_current_speed = False
        # self.current_speed = None
        ####################################

        ### 현재 webot 바퀴 각도 ###
        # rospy.Subscriber("current_angle", Float32, self.current_angle_callback)
        # self.is_current_angle = False
        # self.current_angle = None
        ####################################

        ### 목표 제어값 출력 ###
        # self.target_control_pub = rospy.Publisher("high_level/ackermann_cmd_mux/output", AckermannDriveStamped, queue_size=1)
        # self.target_control = AckermannDriveStamped()
        ####################################

        ### 로우레벨 제어 퍼블리셔 ###
        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        ### pid 초기화 ###
        self.pid = PIDController(1.0, 0, 0)
        ####################################

        rate = rospy.Rate(10)  # 10hz

        ### main ###
        while not rospy.is_shutdown():
            os.system('clear')
            if self.is_target_input: # and self.is_current_speed and self.is_current_angle:

                #self.target_control.drive.steering_angle = self.pid.compute(self.target_input.drive.steering_angle,self.current_angle)
                # self.target_control_pub.publish(self.target_control)

                speed = self.target_input.drive.speed
                angle = self.target_input.drive.steering_angle

                # 필요 시 정규화 조향값
                # max_angle = 0.5  # rad 기준 최대 조향각
                # norm_angle = max(min(angle / max_angle, 1.0), -1.0)

                # 로우레벨 명령 퍼블리시
                self.speed_pub.publish(Float64(speed))
                self.steer_pub.publish(Float64(angle))

                rospy.loginfo(f"🚗 Speed: {speed:.2f}, Angle: {angle:.2f}")

            else:
                print("target_input: ", self.is_target_input)
                rospy.logwarn("❗ target_control 메시지 대기 중...")
                # print("current_speed: ", self.is_current_speed)
                # print("current_angle: ", self.is_current_angle)

            self.is_target_input = False
            # self.is_current_speed = False
            # self.is_current_angle = False
            rate.sleep()

    def target_callback(self, data):
        self.is_target_input = True
        self.target_input = data
        self.target_control.drive.speed = data.drive.speed
        self.target_control.drive.steering_angle = data.drive.steering_angle

    def current_speed_callback(self, data):
        self.is_current_speed = True
        self.current_speed = data.data

    def current_angle_callback(self, data):
        self.is_current_angle = True
        self.current_angle = data.data

if __name__ == '__main__':
    try:
        control()
    except rospy.ROSInterruptException:
        pass