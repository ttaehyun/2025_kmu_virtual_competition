#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16MultiArray
from morai_msgs.msg import ObjectStatusList
from tf.transformations import quaternion_from_euler
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def in_area(point, bounds):
    x, y = point
    min_x, max_x, min_y, max_y = bounds
    return min_x <= x <= max_x and min_y <= y <= max_y

class DeliveryGoalActionClient:
    RETRY_INTERVAL = 5.0  # seconds

    def __init__(self):
        rospy.init_node('delivery_goal_action_client', anonymous=True)

        # Action client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Waiting for move_base action server...')
        self.client.wait_for_server()
        rospy.loginfo('Connected to move_base action server')

        # Subscribers
        self.obj_sub = rospy.Subscriber('/delivery_object', ObjectStatusList, self.object_callback)
        self.chk_sub = rospy.Subscriber('/delivery_check', Int16MultiArray, self.check_callback)

        # Retry timer
        self.retry_timer = rospy.Timer(rospy.Duration(self.RETRY_INTERVAL), self.retry_callback)

        # Area boundaries
        self.area_A = (-17.0, -7.0, -6.0, -2.0)
        self.area_B = (-9.0, -2.0, 3.0, 6.0)
        self.area_C = (-5.0, -2.0, -3.0, 3.0)

        # State variables
        self.mode = 'idle'  # 'idle', 'delivery', 'return', 'done'
        self.delivery_done = False
        self.goals = []
        self.goal_indices = []
        self.current_idx = 0
        self.last_check = [0, 0, 0]

    def make_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        return goal

    def send_goal(self, goal):
        self.client.cancel_goal()
        rospy.loginfo(f"Sending goal: x={goal.target_pose.pose.position.x}, y={goal.target_pose.pose.position.y}")
        self.client.send_goal(goal)

    def object_callback(self, msg: ObjectStatusList):
        # Trigger new delivery if idle
        if self.mode != 'idle' or self.delivery_done:
            return

        # Identify pedestrian area
        ped = msg.pedestrian_list[0]
        ped_pt = (ped.position.x, ped.position.y)
        if in_area(ped_pt, self.area_A): seq = ['C','B']; ped_area = 'A'
        elif in_area(ped_pt, self.area_B): seq = ['A','C']; ped_area = 'B'
        elif in_area(ped_pt, self.area_C): seq = ['A','B']; ped_area = 'C'
        else:
            rospy.logwarn('Pedestrian not in A/B/C area; ignoring.')
            return

        # Map obstacle positions and indices
        obs_map = {}
        obs_idx_map = {}
        for idx, obs in enumerate(msg.obstacle_list):
            pt = (obs.position.x, obs.position.y)
            if in_area(pt, self.area_A):
                obs_map['A'] = [obs.position.x, obs.position.y]
                obs_idx_map['A'] = idx + 1  # delivery_check index
            elif in_area(pt, self.area_B):
                obs_map['B'] = [obs.position.x, obs.position.y]
                obs_idx_map['B'] = idx + 1
            elif in_area(pt, self.area_C):
                obs_map['C'] = [obs.position.x, obs.position.y]
                obs_idx_map['C'] = idx + 1

        # Build delivery goals with correct goal_indices
        self.goals = []
        self.goal_indices = []
        for area in seq:
            if area not in obs_map:
                rospy.logwarn(f"Obstacle {area} not found; skipping.")
                continue
            x, y = obs_map[area]
            if area in ['A', 'B']:
                x -= 0.3
            else:
                y += 0.3
            yaw = -1.5708 if area == 'C' else 0.0
            self.goals.append(self.make_goal(x, y, yaw))
            self.goal_indices.append(obs_idx_map[area])

        # Pedestrian goal
        x_p, y_p = ped.position.x, ped.position.y
        if ped_area in ['A','B']:
            x_p -= 0.3
        else:
            y_p += 0.3
        yaw_p = -1.5708 if ped_area == 'C' else 0.0
        self.goals.append(self.make_goal(x_p, y_p, yaw_p))
        self.goal_indices.append(0)

        # Start sequence
        self.current_idx = 0
        self.mode = 'delivery'
        self.last_check = [0, 0, 0]
        self.send_goal(self.goals[0])

    def check_callback(self, msg: Int16MultiArray):
        data = list(msg.data)
        if self.mode != 'delivery':
            self.last_check = data
            return
        idx = self.goal_indices[self.current_idx]
        if data[idx] > self.last_check[idx]:
            self.current_idx += 1
            if self.current_idx < len(self.goals):
                rospy.loginfo(f"Reached goal index {idx}, sending goal {self.current_idx}")
                self.send_goal(self.goals[self.current_idx])
            else:
                rospy.loginfo('All delivery goals reached; initiating return.')
                self.mode = 'return'
                self.send_return_goal()
        self.last_check = data

    def retry_callback(self, event):
        state = self.client.get_state()
        if self.mode == 'delivery' and state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
            rospy.logwarn(f"Delivery goal {self.current_idx} aborted/rejected ({state}), retrying...")
            self.send_goal(self.goals[self.current_idx])
        elif self.mode == 'return':
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                rospy.logwarn('Return goal aborted/rejected, retrying...')
                self.send_return_goal()
            elif state == GoalStatus.SUCCEEDED:
                rospy.loginfo('Return goal succeeded; no further action.')
                self.mode = 'done'
                self.delivery_done = True

    def send_return_goal(self):
        rg = self.make_goal(0.5, -5.43, 0.0)
        rospy.loginfo('Sending return goal: x=0.5, y=-5.43, yaw=0.0')
        self.send_goal(rg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    DeliveryGoalActionClient().run()
