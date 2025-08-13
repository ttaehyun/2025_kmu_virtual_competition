#!/usr/bin/env python3
# lane_flag_gui.py
import rospy
from std_msgs.msg import Int32
import tkinter as tk
from functools import partial
import time

TOPIC = "/lane_flag"  # 바꾸고 싶으면 ROS 파라미터로도 받습니다.

class LaneFlagGUI:
    def __init__(self, master):
        rospy.init_node("lane_flag_gui", anonymous=True)
        topic = rospy.get_param("~topic", TOPIC)
        init_val = int(rospy.get_param("~init", 1))  # 기본: 오른쪽(1)
        self.pub = rospy.Publisher(topic, Int32, queue_size=1, latch=True)

        self.master = master
        self.master.title("Lane Flag Controller")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.state = init_val
        self.last_pub_ts = 0.0
        self.debounce_s = float(rospy.get_param("~debounce_s", 0.15))

        # UI
        self.lbl = tk.Label(master, text="", font=("Arial", 16))
        self.lbl.pack(padx=10, pady=10)

        btn_frame = tk.Frame(master)
        btn_frame.pack(padx=10, pady=5)

        self.btn_left  = tk.Button(btn_frame, text="◀ Left lane (0)",  width=16,
                                   command=partial(self.set_and_pub, 0))
        self.btn_right = tk.Button(btn_frame, text="Right lane (1) ▶", width=16,
                                   command=partial(self.set_and_pub, 1))
        self.btn_toggle = tk.Button(master, text="Toggle (Space)", width=20,
                                    command=self.toggle)

        self.btn_left.grid(row=0, column=0, padx=5, pady=5)
        self.btn_right.grid(row=0, column=1, padx=5, pady=5)
        self.btn_toggle.pack(padx=10, pady=5)

        # 키보드 단축키
        master.bind("<Left>",  lambda e: self.set_and_pub(0))
        master.bind("<Right>", lambda e: self.set_and_pub(1))
        master.bind("<space>", lambda e: self.toggle())

        # 초기 발행
        self.publish(self.state)
        self.update_label()
        # Tk 주기적으로 ROS 스핀 처리
        self.spin()

    def publish(self, v: int):
        now = time.time()
        if now - self.last_pub_ts < self.debounce_s:
            return
        self.pub.publish(Int32(v))
        self.last_pub_ts = now

    def set_and_pub(self, v: int):
        self.state = int(v)
        self.publish(self.state)
        self.update_label()

    def toggle(self):
        self.set_and_pub(1 - self.state)

    def update_label(self):
        txt = f"Current lane_flag: {self.state}  (0=Left, 1=Right)"
        self.lbl.config(text=txt)
        # 버튼 강조
        self.btn_left.config( relief=tk.SUNKEN if self.state==0 else tk.RAISED )
        self.btn_right.config(relief=tk.SUNKEN if self.state==1 else tk.RAISED )

    def spin(self):
        # rospy.spin() 대신 Tk after 로 주기 처리
        if not rospy.is_shutdown():
            # 필요하면 여기서 상태 구독/업데이트도 가능
            self.master.after(50, self.spin)
        else:
            try:
                self.master.destroy()
            except tk.TclError:
                pass

    def on_close(self):
        rospy.signal_shutdown("GUI closed")
        try:
            self.master.destroy()
        except tk.TclError:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    LaneFlagGUI(root)
    root.mainloop()
