import rospy
from sensor_msgs.msg import Imu

class sub:
    def __init__(self):
        rospy.init_node("sub")
        rospy.Subscriber("/imu", Imu, self.callback)

    def callback(self,msg):
        if msg.header.seq == 25000:
            print(msg)

sub = sub()
rospy.spin()

#