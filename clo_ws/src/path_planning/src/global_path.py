#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
from math import cos,sin,pi,sqrt,pow
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class read_path_pub :

    def __init__(self):
        rospy.init_node('global_path_pub', anonymous=True)
        self.kmu_out_path_pub = rospy.Publisher('/kmu_out_path',Path, queue_size=1)
        self.kmu_in_path_pub = rospy.Publisher('/kmu_in_path',Path, queue_size=1)

        rospack=rospkg.RosPack()
        path_planning_pkg_path=rospack.get_path('path_planning')

        kmu_out_path_full_path=path_planning_pkg_path+'/kmu_out_path.txt'
        self.kmu_out_path_f=open(kmu_out_path_full_path,'r')
        kmu_out_path_lines=self.kmu_out_path_f.readlines()

        self.kmu_out_path_msg=Path()
        self.kmu_out_path_msg.header.frame_id='/map'

        for line in kmu_out_path_lines :

            tmp=line.split()
            read_pose=PoseStamped()
            read_pose.pose.position.x=float(tmp[0])
            read_pose.pose.position.y=float(tmp[1])
            read_pose.pose.orientation.w=1
            self.kmu_out_path_msg.poses.append(read_pose)

        self.kmu_out_path_f.close()

        kmu_in_path_full_path=path_planning_pkg_path+'/kmu_in_path.txt'
        self.kmu_in_path_f=open(kmu_in_path_full_path,'r')
        kmu_in_path_lines=self.kmu_in_path_f.readlines()

        self.kmu_in_path_msg=Path()
        self.kmu_in_path_msg.header.frame_id='/map'

        for line in kmu_in_path_lines :

            tmp=line.split()
            read_pose=PoseStamped()
            read_pose.pose.position.x=float(tmp[0])
            read_pose.pose.position.y=float(tmp[1])
            read_pose.pose.orientation.w=1
            self.kmu_in_path_msg.poses.append(read_pose)

        self.kmu_in_path_f.close()

        rate = rospy.Rate(20) # 20hz
        while not rospy.is_shutdown():
            self.kmu_out_path_pub.publish(self.kmu_out_path_msg)
            self.kmu_in_path_pub.publish(self.kmu_in_path_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        test_track=read_path_pub()
    except rospy.ROSInterruptException:
        pass

