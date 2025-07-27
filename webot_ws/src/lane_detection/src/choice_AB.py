#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Int32, Bool

class AR:
    def __init__(self):
        rospy.init_node('Choice_AB')

        self.lane_switch = rospy.Publisher('/direction_flag', Int32, queue_size=1)
        self.switch_data = Int32()

        self.ar_markers = []

        rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.ar_callback, queue_size=1)
        self.parking_pub = rospy.Publisher("/parking_flag", Bool, queue_size=1)
        self.is_ar_markers = False
        self.ar_markers = []
        self.parking_msg = Bool()

        ############## 변경 가능한 파라미터 ###############
        self.target_ar_id1 = 0 # A(왼쪽)
        self.target_ar_id2 = 4 # B(오른쪽)
        self.counting_to_need = 1
        self.distance_from_ar = 2.5
        ##############################################

        self.ar_id = None
        self.ar_detected_count = 0

        rate = rospy.Rate(8) # 10hz
        ################ main ###################
        while not rospy.is_shutdown():
            if self.is_ar_markers:
                if self.check_AR():
                    if self.ar_id == self.target_ar_id1:
                        self.switch_data.data = 0
                        self.lane_switch.publish(self.switch_data)
                        rospy.loginfo(f"check_AR: left lane gogo")

                        self.parking_msg.data = True
                        self.parking_pub.publish(self.parking_msg)

                    elif self.ar_id == self.target_ar_id2:
                        self.switch_data.data = 1
                        self.lane_switch.publish(self.switch_data)
                        rospy.loginfo(f"check_AR: right lane gogo")
                        
                        self.parking_msg.data = True
                        self.parking_pub.publish(self.parking_msg)

            self.is_ar_markers = False
            rate.sleep()

    def ar_callback(self, data):
        self.is_ar_markers = True
        self.ar_markers = data.markers

    def check_AR(self):
        if len(self.ar_markers) == 1:
            ar_marker = self.ar_markers[0]
            if self.ar_id == None:
                self.ar_id = ar_marker.id
            else:
                if self.ar_id != ar_marker.id:
                    self.ar_id = ar_marker.id
                    self.ar_detected_count = 0
                    return False
            if self.ar_detected(ar_marker.pose.pose.position.z):
                return True
            else:
                return False
        else:
            return False

    def ar_detected(self,distance):
        if distance <= self.distance_from_ar:
            self.ar_detected_count += 1
            if self.ar_detected_count >= self.counting_to_need:
                return True
            else:
                return False
        else:
            self.ar_detected_count = 0
            return False

if __name__ == '__main__':
    try:
        Choice_AB = AR()
    except rospy.ROSInterruptException:
        rospy.loginfo("Choice_AB node terminated.")