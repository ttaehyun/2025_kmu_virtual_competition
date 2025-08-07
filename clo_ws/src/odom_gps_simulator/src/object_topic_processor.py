#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Int16MultiArray
from morai_msgs.msg import ObjectStatusList, ObjectStatus, EgoVehicleStatus

class ObjectTopicProcessor:
    def __init__(self):
        rospy.init_node('object_topic_processor', anonymous=True)

        # Publishers
        self.filtered_obj_pub = rospy.Publisher('/delivery_object', ObjectStatusList, queue_size=10)
        self.delivery_check_pub = rospy.Publisher('/delivery_check', Int16MultiArray, queue_size=10)

        # Data placeholders
        self.ped_data_ = ObjectStatus()
        self.obj_1_data_ = ObjectStatus()
        self.obj_2_data_ = ObjectStatus()
        self.max_distance_ = 1.2
        self.obj_reach_ = Int16MultiArray(data=[0, 0, 0])
        self.reach_flag_ = False
        self.goal_free_time_ = rospy.Time.now()

        # Subscribers
        rospy.Subscriber('/Object_topic', ObjectStatusList, self.object_callback)
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.ego_callback)

    def object_callback(self, msg):
        # Filter specific objects by unique_id
        for ped in msg.pedestrian_list:
            if ped.unique_id == 50:
                self.ped_data_ = ped
        for obs in msg.obstacle_list:
            if obs.unique_id == 51:
                self.obj_1_data_ = obs
            elif obs.unique_id == 52:
                self.obj_2_data_ = obs

        filtered_msg = ObjectStatusList()
        filtered_msg.header = msg.header
        filtered_msg.num_of_pedestrian = 1
        filtered_msg.num_of_obstacle = 2
        filtered_msg.pedestrian_list.append(self.ped_data_)
        filtered_msg.obstacle_list.extend([self.obj_1_data_, self.obj_2_data_])

        self.filtered_obj_pub.publish(filtered_msg)

    def ego_callback(self, msg):
        ego_pos = np.array([msg.position.x, msg.position.y])
        ped_pos = np.array([self.ped_data_.position.x, self.ped_data_.position.y])
        obj1_pos = np.array([self.obj_1_data_.position.x, self.obj_1_data_.position.y])
        obj2_pos = np.array([self.obj_2_data_.position.x, self.obj_2_data_.position.y])

        d_ped = np.linalg.norm(ego_pos - ped_pos)
        d1 = np.linalg.norm(ego_pos - obj1_pos)
        d2 = np.linalg.norm(ego_pos - obj2_pos)

        now = rospy.Time.now()
        if d1 < self.max_distance_:
            if (now - self.goal_free_time_).to_sec() > 2.0:
                self.obj_reach_.data[1] = 1
        elif d2 < self.max_distance_:
            if (now - self.goal_free_time_).to_sec() > 2.0:
                self.obj_reach_.data[2] = 1
        elif d_ped < self.max_distance_:
            if (now - self.goal_free_time_).to_sec() > 2.0 and not self.reach_flag_:
                self.obj_reach_.data[0] += 1
                self.reach_flag_ = True
        else:
            self.goal_free_time_ = now
            self.reach_flag_ = False

        self.delivery_check_pub.publish(self.obj_reach_)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    processor = ObjectTopicProcessor()
    processor.run()
