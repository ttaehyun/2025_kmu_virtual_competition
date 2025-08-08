#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16
from ackermann_msgs.msg import AckermannDriveStamped

class TrafficStopNode:
    def __init__(self):
        rospy.init_node('traffic_stop_node')
        
        self.stopline_sub = rospy.Subscriber('/stop_line', Int16, self.stopline_callback)
        self.traffic_light_sub = rospy.Subscriber('/trafficLight', Int16, self.traffic_light_callback)
        self.ack_nav2_sub = rospy.Subscriber('/ackermann_cmd_mux/input/nav_2', AckermannDriveStamped, self.ack_nav2_callback)
        self.ack_pub = rospy.Publisher('/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=1)
        self.stopline_detected = False
        self.trafficlight_detected = 0
        
        self.ack_nav2_msg = AckermannDriveStamped()
    def stopline_callback(self, msg):
        self.stopline_detected = msg.data
        print(f"Stop line detected: {self.stopline_detected}")
    def traffic_light_callback(self, msg):
        self.trafficlight_detected = msg.data
        print(f"Traffic light status: {self.trafficlight_detected}")
    
    def ack_nav2_callback(self, msg):
        if self.trafficlight_detected <99:
            self.ack_nav2_msg = msg
    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            if self.trafficlight_detected <99:
                ack_msg = AckermannDriveStamped()
                
                if self.trafficlight_detected <=5:
                    ack_msg.drive.speed = 0.2
                    ack_msg.drive.steering_angle = self.ack_nav2_msg.drive.steering_angle
                    if self.stopline_detected:
                        ack_msg.drive.speed = 0.0
                        print("Stop line detected, stopping vehicle.")
                    else:
                        pass
                else:
                    ack_msg.drive.speed = 0.5
                    ack_msg.drive.steering_angle = self.ack_nav2_msg.drive.steering_angle
                    print("Traffic light is green, proceeding.")
                self.ack_pub.publish(ack_msg)
            else:
                print("No traffic light detected")
            rate.sleep()

if __name__ == '__main__':
    node = TrafficStopNode()
    node.run()