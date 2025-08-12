#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeliveryGoalActionClient (안전가드 강화판)

주요 변경점(요약):
- 보행자 리스트가 비었을 때 인덱스 에러 방지
- /delivery_check 길이와 goal index의 불일치에 대한 범위 체크(크래시 → 대기)
- self.last_check를 동적으로 길이 동기화(고정 길이 [0,0,0] → 가변)
- retry 타이머 종료 처리 (복귀 성공 시)
- goal 전송 시 불필요한 cancel 최소화(현재 상태가 ACTIVE/PENDING일 때만 preempt)
- 목표 생성/인덱싱 규칙과 경계 포함 처리에 대한 상세 주석 보강

중요 가정(필수 확인):
- /delivery_check 의 data 배열에서
  index 0: 보행자
  index 1..N: 장애물(ObstacleStatusList의 obstacle_list 순서대로 1-based 매핑)
  위 규칙과 다르면 self.goal_indices 매핑을 조정하세요.
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
    ※ <= 비교라 경계에 걸린 점이 여러 영역에 동시에 속할 수 있음(아래 elif 순서로 우선순위 결정).
    """
    x, y = point
    min_x, max_x, min_y, max_y = bounds
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)


class DeliveryGoalActionClient:
    RETRY_INTERVAL = 5.0  # seconds

    def __init__(self):
        rospy.init_node('delivery_goal_action_client', anonymous=True)

        # --- move_base 액션 클라이언트 준비 ---
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Waiting for move_base action server...')
        self.client.wait_for_server()
        rospy.loginfo('Connected to move_base action server')

        # --- 구독자 등록 ---
        # /delivery_object: 보행자+장애물 감지 결과(ObjectStatusList)
        self.obj_sub = rospy.Subscriber('/delivery_object', ObjectStatusList, self.object_callback)
        # /delivery_check: 현재 인덱스별 전달 완료 카운터(Int16MultiArray)
        self.chk_sub = rospy.Subscriber('/delivery_check', Int16MultiArray, self.check_callback)

        # --- 재시도 타이머 ---
        self.retry_timer = rospy.Timer(rospy.Duration(self.RETRY_INTERVAL), self.retry_callback)

        # --- 영역 경계 정의 ---
        # 경계값 포함(<=)이므로 경계 중복이 있을 수 있음. 아래 object_callback의 if/elif 순서가 우선순위를 결정.
        self.area_A = (-17.0, -7.0, -6.0, -2.0)
        self.area_B = (-9.0, -2.0,  3.0,  6.0)
        self.area_C = (-5.0, -2.0, -3.0,  3.0)

        # --- 상태 변수 ---
        self.mode = 'idle'          # 'idle' | 'delivery' | 'return' | 'done'
        self.delivery_done = False  # 전체 시퀀스 1회 완료 플래그
        self.goals = []             # MoveBaseGoal 리스트(장애물 2개 + 보행자 1개 순서)
        self.goal_indices = []      # /delivery_check에서 읽을 인덱스(장애물은 1-based, 보행자는 0)
        self.current_idx = 0        # self.goals 진행 인덱스
        self.last_check = []        # 마지막 /delivery_check 데이터(길이 동기화 방식)

        # --- 복귀 목표 파라미터(필요 시 조정) ---
        self.return_x = 0.5
        self.return_y = -5.43
        self.return_yaw = 0.0  # rad

        # --- 접근 오프셋(각 영역에서 목표지점 보정값) ---
        # A/B 영역: x 방향으로 -0.3m, C 영역: y 방향으로 +0.3m
        self.offset_AB_x = -0.4
        self.offset_C_y  =  0.4

    # --------------------------
    # 유틸: goal 생성
    # --------------------------
    def make_goal(self, x, y, yaw):
        """
        map 좌표계 기준 목표 Pose를 MoveBaseGoal로 생성.
        yaw: rad
        """
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
        - 현재 goal이 ACTIVE/PENDING이면 cancel 후 새 goal 전송(불필요 preempt 최소화).
        """
        state = self.client.get_state()
        if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            try:
                self.client.cancel_goal()
                # cancel 신호 전파 약간의 여유
                rospy.sleep(0.05)
            except Exception as e:
                rospy.logwarn(f"cancel_goal() failed: {e}")

        x = goal.target_pose.pose.position.x
        y = goal.target_pose.pose.position.y
        rospy.loginfo(f"Sending goal: x={x:.3f}, y={y:.3f}")
        self.client.send_goal(goal)

    # --------------------------
    # 콜백: 객체 정보(/delivery_object)
    # --------------------------
    def object_callback(self, msg: ObjectStatusList):
        """
        - idle 상태에서만 새 배송 시퀀스를 시작.
        - 보행자 위치 기준으로 '현재 보행자가 속한 영역'을 ped_area로 파악하고,
          나머지 두 영역에 대해 장애물 목표(접근 포즈) → 마지막에 보행자 목표 순으로 goals 구성.
        - goal_indices에는 /delivery_check 상의 인덱스를 저장:
          장애물은 1-based(idx+1), 보행자는 0.
        """
        # 이미 동작 중이거나 1회 완료면 무시
        if self.mode != 'idle' or self.delivery_done:
            return

        # 보행자 유무 방어: 비어 있으면 무시
        if not msg.pedestrian_list:
            rospy.logwarn('No pedestrian in /delivery_object; ignoring.')
            return

        # --- 보행자 영역 판단 ---
        ped = msg.pedestrian_list[0]
        ped_pt = (ped.position.x, ped.position.y)

        # 순서 중요: 경계 중복 시 위에서 매칭된 영역이 선택됨
        if in_area(ped_pt, self.area_A):
            seq = ['C', 'B']  # 보행자가 A에 있으면 C→B 순으로 장애물 배송 후 보행자
            ped_area = 'A'
        elif in_area(ped_pt, self.area_B):
            seq = ['A', 'C']
            ped_area = 'B'
        elif in_area(ped_pt, self.area_C):
            seq = ['A', 'B']
            ped_area = 'C'
        else:
            rospy.logwarn('Pedestrian not in A/B/C area; ignoring.')
            return

        # --- 장애물 매핑: 같은 영역에 여러 개면 "마지막 것만" 저장(의도에 맞는지 확인 필요) ---
        obs_map = {}     # 'A'|'B'|'C' -> [x, y]
        obs_idx_map = {} # 'A'|'B'|'C' -> /delivery_check index (1-based 가정)
        for idx, obs in enumerate(msg.obstacle_list):
            pt = (obs.position.x, obs.position.y)
            if in_area(pt, self.area_A):
                obs_map['A'] = [obs.position.x, obs.position.y]
                obs_idx_map['A'] = idx + 1  # ⚠ 1-based 가정. /delivery_check[1]이 obstacle_list[0]에 대응.
            elif in_area(pt, self.area_B):
                obs_map['B'] = [obs.position.x, obs.position.y]
                obs_idx_map['B'] = idx + 1
            elif in_area(pt, self.area_C):
                obs_map['C'] = [obs.position.x, obs.position.y]
                obs_idx_map['C'] = idx + 1

        # --- 배송 목표 구성: 두 영역 장애물 → 보행자 순 ---
        self.goals = []
        self.goal_indices = []

        for area in seq:
            if area not in obs_map:
                rospy.logwarn(f"Obstacle {area} not found; skipping.")
                continue

            x, y = obs_map[area]
            # 접근 포즈 보정(차량 접근 방향/충돌 여유)
            if area in ('A', 'B'):
                x += self.offset_AB_x     # (-) 방향으로 0.3 m
                yaw = 0.0                 # 전진 방향(동/서) 가정
            else:  # 'C'
                y += self.offset_C_y      # (+y) 방향으로 0.3 m
                yaw = -pi * 0.5            # 90 deg(북쪽) 바라보도록

            self.goals.append(self.make_goal(x, y, yaw))
            self.goal_indices.append(obs_idx_map[area])  # ⚠ 1-based index 저장

        # --- 보행자 목표(마지막) ---
        x_p, y_p = ped.position.x, ped.position.y
        if ped_area in ('A', 'B'):
            x_p += self.offset_AB_x
            yaw_p = 0.0
        else:  # 'C'
            y_p += self.offset_C_y
            yaw_p = -pi * 0.5

        self.goals.append(self.make_goal(x_p, y_p, yaw_p))
        self.goal_indices.append(0)  # ⚠ 보행자 인덱스는 0이라는 가정

        # 생성된 목표가 하나도 없으면 중단
        if not self.goals:
            rospy.logwarn('No goals built; staying idle.')
            return

        # --- 상태 갱신 및 첫 목표 송신 ---
        self.current_idx = 0
        self.mode = 'delivery'
        self.last_check = []  # 길이 동기화를 위해 비워둠
        self.send_goal(self.goals[0])

    # --------------------------
    # 콜백: 전달 체크(/delivery_check)
    # --------------------------
    def check_callback(self, msg: Int16MultiArray):
        """
        /delivery_check 데이터(각 인덱스별 전달 진척 카운터)를 수신.
        - 현재 목표에 해당하는 인덱스의 값이 증가하면 다음 목표로 진행.
        - 길이 불일치, 인덱스 범위 초과 등은 크래시 없이 경고 로그 후 대기.
        """
        data = list(msg.data)

        # 빈 데이터 방어
        if not data:
            return

        # last_check 길이 동기화(초기/확장 시 부족분을 0으로 보충)
        if len(self.last_check) < len(data):
            self.last_check += [0] * (len(data) - len(self.last_check))

        # delivery 모드가 아니면 동기화만
        if self.mode != 'delivery':
            self.last_check[:len(data)] = data
            return

        # 현재 진행 인덱스 가드
        if self.current_idx >= len(self.goal_indices):
            rospy.logwarn('Current goal index out of range; ignoring check.')
            self.last_check[:len(data)] = data
            return

        idx = self.goal_indices[self.current_idx]

        # /delivery_check 인덱스 범위 확인
        if idx < 0 or idx >= len(data):
            rospy.logwarn(f"/delivery_check size {len(data)} <= needed idx {idx}; waiting for valid data.")
            self.last_check[:len(data)] = data
            return

        # 핵심: 목표 인덱스의 카운터 값이 증가했는지 확인
        if data[idx] > self.last_check[idx]:
            # 다음 목표로 전환
            self.current_idx += 1
            if self.current_idx < len(self.goals):
                rospy.loginfo(f"Reached goal index {idx}, sending goal {self.current_idx}")
                self.send_goal(self.goals[self.current_idx])
            else:
                rospy.loginfo('All delivery goals reached; initiating return.')
                self.mode = 'return'
                self.send_return_goal()

        # 마지막에 동기화
        self.last_check[:len(data)] = data

    # --------------------------
    # 타이머: 재시도 로직
    # --------------------------
    def retry_callback(self, event):
        """
        일정 주기로 move_base 상태를 확인해 ABORTED/REJECTED 시 재전송.
        복귀(goal) 성공 시 타이머 종료 및 전체 완료 플래그 설정.
        """
        state = self.client.get_state()

        if self.mode == 'delivery':
            # 진행 인덱스 가드(예외 상황)
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
                # 더 이상 재시도 불필요하므로 타이머 종료(선택)
                try:
                    self.retry_timer.shutdown()
                except Exception:
                    pass

    # --------------------------
    # 복귀 목표 전송
    # --------------------------
    def send_return_goal(self):
        """
        복귀 지점으로 goal을 전송.
        """
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
