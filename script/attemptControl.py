#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray, String

class ControlNode():

    def __init__(self):
        """
        Initializes the ROS node and sets up subscribers and publishers.
        """
        rospy.init_node('CtrlAttemptnod', anonymous=True)
        self.waypoint_sub = rospy.Subscriber("/lane/waypoints", Float32MultiArray, self.callback, queue_size=3)
        self.steer_pub = rospy.Publisher("/automobile/command", String, queue_size=1)
        self.prev_error = 0
        self.Proportional = 0.12
        self.Differential = 0.095
        print("Node started")
        rospy.spin()


    def callback(self, data):
        """
        Callback function for handling waypoint data.
        """
        center = data.data[0]
        # print(center)
        error = 320 - center  
        steer = -(self.Proportional * error + self.Differential * self.prev_error ** 2)
        print(steer)
        self.steer_pub.publish(String(data=f'{{"action": "2", "steerAngle": {steer}}}'))

if __name__ == '__main__':
    print("into main")
    try:
        nod = ControlNode()
    except rospy.ROSInterruptException:
        pass