#!/usr/bin/env python3

import rospy
import numpy as np
import networkx as nx
import cv2
import os
import math
# from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates
# from utils.msg import IMU

# Waypoint displayer class: subscribe to "/lane/waypoints" and displays them
class WaypointNode():
    def __init__(self):
        self.map = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/templates/map.png')
        print("init display waypoint node")
        rospy.init_node('waypoint_node', anonymous=True)
        self.model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback, queue_size=3)
        self.waypoint_sub = rospy.Subscriber("/lane/waypoints", Float32MultiArray, self.callback_w, queue_size=3)
        # self.imu_sub = rospy.Subscriber("/automobile/IMU", IMU, self.callback_imu, queue_size=3)
        self.rate = rospy.Rate(15)
        self.p = Float32MultiArray()
        self.x = 0
        self.y = 0
        self.yaw = 0

    def callback(self, model):
        try:
            car_idx = model.name.index("automobile")
        except ValueError:
            print("Can't find automobile in modelstates")
            return

        self.x = model.pose[car_idx].position.x
        self.y = -model.pose[car_idx].position.y

    # Draw the car's position and orientation on the map and the waypoints relative to it
    def callback_w(self, waypoints):
        img_map = np.copy(self.map)
        img_map = cv2.arrowedLine(img_map, (int(self.x/15*self.map.shape[0]),int(self.y/15*self.map.shape[1])),
                    ((int((self.x+0.3*math.cos(self.yaw))/15*self.map.shape[0]),int((self.y-0.3*math.sin(self.yaw))/15*self.map.shape[1]))), color=(255,0,255), thickness=10)
        for i in range(waypoints.layout.dim[0].size):   
            cv2.circle(img_map, (int((self.x+waypoints.data[2*i]*math.cos(self.yaw)-waypoints.data[2*i+1]*math.sin(self.yaw))/15*self.map.shape[0]),
                                int((self.y-waypoints.data[2*i]*math.sin(self.yaw)-waypoints.data[2*i+1]*math.cos(self.yaw))/15*self.map.shape[1])), radius=15, color=(0, 255, 0), thickness=-1)
        windowName = 'track'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,700,700)
        cv2.imshow(windowName, img_map)
        key = cv2.waitKey(1)
    
    def callback_imu(self, imu):
        self.yaw = imu.yaw

if __name__ == '__main__':
    while not rospy.is_shutdown():
        try:
            node = WaypointNode()
            node.rate.sleep()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass