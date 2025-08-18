#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeliveryGoalActionClient (안전가드 강화판 + 동영역 우선 + 보행자 도착자세 고정)

변경 핵심
- 같은 영역( A/B/C )에 보행자와 장애물이 동시에 있을 때:
  ▶ 순서 = [같은 영역 장애물 → 다른 영역(1곳) 장애물 → 보행자]
- 보행자 도착 자세 고정: y축 -0.8m 오프셋, yaw = -π(-180°)
- 안전장치: 빈 리스트/인덱스 범위/길이 불일치 방어, 불필요한 cancel 최소화 등 유지

가정(확인 필요)
- /delivery_check data 배열 매핑:
    index 0: 보행자
    index 1..N: 장애물 (ObstacleStatusList.obstacle_list 순서와 1-based 매핑)
  다를 경우 self.goal_indices 매핑을 조정할 것.
"""

import rospy
from math import pi
from std_msgs.msg import Int16MultiArray
from morai_msgs.msg import ObjectStatusList
from tf.transformations import quaternion_from_euler
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def in_area(point, bounds):
    """
    직사각형 구역 포함 여부 판단(경계값 포함).
    point: (x, y)
    bounds: (min_x, max_x, min_y, max_y)
    ※ <= 비교라 경계에 걸린 점이 여러 영역에 동시에 속할 수 있음(elif 순서가 우선순위).
    """
    x, y = point
    min_x, max_x, min_y, max_y = bounds
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)


class DeliveryGoalActionClient:
    RETRY_INTERVAL = 5.0  # seconds

    def __init__(self):
        rospy.init_node('delivery_goal_action_client', anonymous=True)

        # --- move_base 액션 클라이언트 ---
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Waiting for move_base action server...')
        self.client.wait_for_server()
        rospy.loginfo('Connected to move_base action server')

        # --- 구독자 ---
        self.obj_sub = rospy.Subscriber('/delivery_object', ObjectStatusList, self.object_callback)
        self.chk_sub = rospy.Subscriber('/delivery_check', Int16MultiArray, self.check_callback)

        # --- 재시도 타이머 ---
        self.retry_timer = rospy.Timer(rospy.Duration(self.RETRY_INTERVAL), self.retry_callback)

        # --- 영역 경계 (경계 포함) ---
        self.area_A = (-17.0, -7.0, -6.0, -2.0)
        self.area_B = (-9.0,  -2.0,  3.0,   6.0)
        self.area_C = (-5.0,  -2.0, -3.0,   3.0)

        # --- 상태 ---
        self.mode = 'idle'          # 'idle' | 'delivery' | 'return' | 'done'
        self.delivery_done = False
        self.goals = []
        self.goal_indices = []
        self.current_idx = 0
        self.last_check = []

        # --- 복귀 목표 ---
        self.return_x = 0.5
        self.return_y = -5.43
        self.return_yaw = 0.0  # rad

        # --- 장애물 접근 오프셋 ---
        self.offset_AB_x = -0.8  # A/B영역: x 음의 방향으로 0.4m
        self.offset_C_y  =  0.8  # C영역:   y 양의 방향으로 0.4m

        # --- 보행자 도착 자세(요청사항) ---
        self.ped_goal_offset_y = -0.8  # 보행자 기준 y축으로 -0.8m
        self.ped_goal_yaw      = 0.0   # -180도

        # 안전 종료 시 전체 goal 취소
        rospy.on_shutdown(lambda: self.client.cancel_all_goals())

    # --------------------------
    # 유틸: goal 생성
    # --------------------------
    def make_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        return goal

    def send_goal(self, goal):
        """
        새 goal 전송.
        - 현재 goal이 ACTIVE/PENDING이면 cancel 후 전송(불필요 preempt 최소화).
        """
        try:
            state = self.client.get_state()
        except Exception:
            state = None

        if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            try:
                self.client.cancel_goal()
                rospy.sleep(0.05)  # cancel 전파 딜레이
            except Exception as e:
                rospy.logwarn(f"cancel_goal() failed: {e}")

        x = goal.target_pose.pose.position.x
        y = goal.target_pose.pose.position.y
        rospy.loginfo(f"Sending goal: x={x:.3f}, y={y:.3f}")
        self.client.send_goal(goal)

    # --------------------------
    # 콜백: /delivery_object
    # --------------------------
    def object_callback(self, msg: ObjectStatusList):
        """
        - idle 상태에서만 새 배송 시퀀스를 시작.
        - 보행자 위치 기준으로 ped_area 결정.
        - 장애물 2개 가정 하에, 같은 영역에 장애물이 있으면
          [같은 영역 장애물 → 다른 영역(1곳) 장애물 → 보행자] 순서,
          없으면 [다른 두 영역(존재하는 곳들) 장애물 → 보행자] 순서.
        - goal_indices: 장애물=1-based, 보행자=0
        """
        if self.mode != 'idle' or self.delivery_done:
            return

        if not msg.pedestrian_list:
            rospy.logwarn('No pedestrian in /delivery_object; ignoring.')
            return

        # --- 보행자 영역 판단 ---
        ped = msg.pedestrian_list[0]
        ped_pt = (ped.position.x, ped.position.y)

        if in_area(ped_pt, self.area_A):
            ped_area = 'A'
            seq = ['C', 'B']  # '보행자 구역 제외' 두 영역의 우선순위
        elif in_area(ped_pt, self.area_B):
            ped_area = 'B'
            seq = ['A', 'C']
        elif in_area(ped_pt, self.area_C):
            ped_area = 'C'
            seq = ['A', 'B']
        else:
            rospy.logwarn('Pedestrian not in A/B/C area; ignoring.')
            return

        # --- 장애물 매핑 (마지막 관측 우선 사용) ---
        obs_map = {}     # 'A'|'B'|'C' -> [x, y]
        obs_idx_map = {} # 'A'|'B'|'C' -> /delivery_check index (1-based)
        for idx, obs in enumerate(msg.obstacle_list):
            pt = (obs.position.x, obs.position.y)
            if in_area(pt, self.area_A):
                obs_map['A'] = [obs.position.x, obs.position.y]
                obs_idx_map['A'] = idx + 1
            elif in_area(pt, self.area_B):
                obs_map['B'] = [obs.position.x, obs.position.y]
                obs_idx_map['B'] = idx + 1
            elif in_area(pt, self.area_C):
                obs_map['C'] = [obs.position.x, obs.position.y]
                obs_idx_map['C'] = idx + 1

        # --- 목표 생성 헬퍼 ---
        def add_obstacle_goal(area_key: str):
            x, y = obs_map[area_key]
            if area_key in ('A', 'B'):
                x += self.offset_AB_x
                yaw = 0.0
            else:  # 'C'
                y += self.offset_C_y
                yaw = -pi * 0.5
            self.goals.append(self.make_goal(x, y, yaw))
            self.goal_indices.append(obs_idx_map[area_key])

        # --- 배송 목표 구성 ---
        self.goals = []
        self.goal_indices = []

        same_area_has_obstacle = (ped_area in obs_map)

        if same_area_has_obstacle:
            # 1) 같은 영역 장애물 먼저
            add_obstacle_goal(ped_area)

            # 2) 다른 두 영역 중 실제 존재하는 1곳(가정상 정확히 1곳)
            other_present = [a for a in seq if a in obs_map]
            if len(other_present) != 1:
                rospy.logwarn(
                    f"[Assumption Mismatch] ped in {ped_area}: expected exactly one other-area obstacle, "
                    f"found {len(other_present)}. Proceeding with available ones."
                )
            for a in other_present[:1]:
                add_obstacle_goal(a)
        else:
            # 같은 영역에 장애물이 없다면: 두 다른 영역(존재하는 곳들)
            for a in seq:
                if a in obs_map:
                    add_obstacle_goal(a)

        # 3) 보행자 목표(마지막): y -0.8, yaw -π 고정
        x_p = ped.position.x
        y_p = ped.position.y + self.ped_goal_offset_y
        yaw_p = self.ped_goal_yaw
        self.goals.append(self.make_goal(x_p, y_p, yaw_p))
        self.goal_indices.append(0)  # 보행자 인덱스 0

        if not self.goals:
            rospy.logwarn('No goals built; staying idle.')
            return

        # --- 상태 갱신 및 첫 goal 전송 ---
        self.current_idx = 0
        self.mode = 'delivery'
        self.last_check = []
        self.send_goal(self.goals[0])

    # --------------------------
    # 콜백: /delivery_check
    # --------------------------
    def check_callback(self, msg: Int16MultiArray):
        """
        /delivery_check 각 인덱스 카운터 수신.
        - 현재 목표에 해당하는 인덱스의 값 증가 시 다음 목표로 전환.
        - 길이/범위 불일치는 경고 후 대기.
        """
        data = list(msg.data)
        if not data:
            return

        if len(self.last_check) < len(data):
            self.last_check += [0] * (len(data) - len(self.last_check))

        if self.mode != 'delivery':
            self.last_check[:len(data)] = data
            return

        if self.current_idx >= len(self.goal_indices):
            rospy.logwarn('Current goal index out of range; ignoring check.')
            self.last_check[:len(data)] = data
            return

        idx = self.goal_indices[self.current_idx]

        if idx < 0 or idx >= len(data):
            rospy.logwarn(f"/delivery_check size {len(data)} <= needed idx {idx}; waiting for valid data.")
            self.last_check[:len(data)] = data
            return

        if data[idx] > self.last_check[idx]:
            self.current_idx += 1
            if self.current_idx < len(self.goals):
                rospy.loginfo(f"Reached goal index {idx}, sending goal {self.current_idx}")
                self.send_goal(self.goals[self.current_idx])
            else:
                rospy.loginfo('All delivery goals reached; initiating return.')
                self.mode = 'return'
                self.send_return_goal()

        self.last_check[:len(data)] = data

    # --------------------------
    # 타이머: 재시도 로직
    # --------------------------
    def retry_callback(self, event):
        """
        주기적으로 move_base 상태 확인.
        - ABORTED/REJECTED: 현재 목표 재전송
        - 복귀 목표 SUCCEEDED: 타이머 종료 및 완료 처리
        """
        try:
            state = self.client.get_state()
        except Exception:
            return

        if self.mode == 'delivery':
            if self.current_idx >= len(self.goals):
                return
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED):
                rospy.logwarn(f"Delivery goal {self.current_idx} aborted/rejected ({state}), retrying...")
                self.send_goal(self.goals[self.current_idx])

        elif self.mode == 'return':
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED):
                rospy.logwarn('Return goal aborted/rejected, retrying...')
                self.send_return_goal()
            elif state == GoalStatus.SUCCEEDED:
                rospy.loginfo('Return goal succeeded; marking done.')
                self.mode = 'done'
                self.delivery_done = True
                try:
                    self.retry_timer.shutdown()
                except Exception:
                    pass

    # --------------------------
    # 복귀 목표 전송
    # --------------------------
    def send_return_goal(self):
        rg = self.make_goal(self.return_x, self.return_y, self.return_yaw)
        rospy.loginfo(f"Sending return goal: x={self.return_x:.3f}, y={self.return_y:.3f}, yaw={self.return_yaw:.3f}")
        self.send_goal(rg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        DeliveryGoalActionClient().run()
    except rospy.ROSInterruptException:
        pass