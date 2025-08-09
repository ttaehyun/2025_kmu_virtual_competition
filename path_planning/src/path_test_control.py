#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import numpy as np
from math import cos, sin, pi, sqrt, atan2

from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion
import tf2_ros

class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        # Subscribers
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/target_v", Float32, self.target_v_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_callback)

        # Publisher: Ackermann command topic
        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/nav_2", AckermannDriveStamped, queue_size=1)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State flags
        self.is_path = False

        # Parameters
        self.target_v = 0.0        # 목표 속도 (m/s)
        self.current_speed = 0.0   # 현재 속도 (m/s)

        # Vehicle & lookahead
        self.vehicle_length = 0.26  # 차량 축간 거리 (m)
        self.lfd = 0.8              # look‑ahead distance (m)

        # Pose
        self.path = None
        self.vehicle_yaw = 0.0
        self.current_position = Point()

        # Control loop
        rate_hz = 20
        rate = rospy.Rate(rate_hz)

        while not rospy.is_shutdown():
            # lookup current pose from tf
            try:
                trans = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
                self.current_position.x = trans.transform.translation.x
                self.current_position.y = trans.transform.translation.y
                q = trans.transform.rotation
                _, _, self.vehicle_yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                rospy.logwarn_throttle(1.0, 'TF lookup map->base_link failed')
                rate.sleep()
                continue

            if self.is_path:
                # 1) Pure pursuit: find look‑ahead point in local coordinates
                T = np.array([
                    [cos(self.vehicle_yaw), -sin(self.vehicle_yaw), self.current_position.x],
                    [sin(self.vehicle_yaw),  cos(self.vehicle_yaw), self.current_position.y],
                    [0.0,                    0.0,                   1.0]
                ])
                det_T = np.linalg.inv(T)

                forward_point = None
                for pose_stamped in self.path.poses:
                    gx = pose_stamped.pose.position.x
                    gy = pose_stamped.pose.position.y
                    gp = np.array([gx, gy, 1.0])
                    lp = det_T.dot(gp)
                    if lp[0] > 0:
                        dist = sqrt(lp[0]**2 + lp[1]**2)
                        if dist >= self.lfd:
                            forward_point = lp
                            break

                if forward_point is not None:
                    theta = atan2(forward_point[1], forward_point[0])
                    steering = -atan2(2.0 * self.vehicle_length * sin(theta), self.lfd)
                else:
                    steering = 0.0

                # create and publish Ackermann command
                ack_msg = AckermannDriveStamped()
                ack_msg.header.stamp = rospy.Time.now()
                ack_msg.header.frame_id = 'base_link'
                ack_msg.drive.steering_angle = steering
                ack_msg.drive.speed = self.target_v

                # Debug print
                os.system('clear')
                print('-------------------------------------')
                print(f' steering (deg) = {steering * 180/pi: .2f}')
                print(f' target_v  (m/s)= {self.target_v: .2f}')
                print(f' current_v (m/s)= {self.current_speed: .2f}')
                print('-------------------------------------')

                self.ack_pub.publish(ack_msg)

            rate.sleep()

    def path_callback(self, msg: Path):
        self.path = msg
        self.is_path = True

    def target_v_callback(self, msg: Float32):
        self.target_v = msg.data

    def ego_callback(self, msg: EgoVehicleStatus):
        self.current_speed = msg.velocity.x

if __name__ == '__main__':
    try:
        PurePursuit()
    except rospy.ROSInterruptException:
        pass
