#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from pynput import keyboard
import threading
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

# #-----Declare Global Variables ----- #
CAMERA_PARAMS = {'fx': 554.3826904296875, 'fy': 554.3826904296875, 'cx': 320, 'cy': 240}
initial = np.float32([[0,360],
                      [640,360],
                      [0,480],
                      [640,480]])

final = np.float32([[0,0],
                    [640,0],
                    [0,480],
                    [640,480]])

cameraMatrix = np.array([[CAMERA_PARAMS['fx'], 0, CAMERA_PARAMS['cx']],
                         [0, CAMERA_PARAMS['fy'], CAMERA_PARAMS['cy']],
                         [0, 0, 1]])

distCoeff = np.array([])
# dest_size = np.array([640,480])
# undistortImage = None
# inverseMap = None
houghThresh = 5
houghMin = 5
houghMax = 5
sobel_size = 3
edgeImage = []

def getIPM(inputImage):
    undistortImage = cv2.undistort(inputImage, cameraMatrix, distCoeff)
    transMatrix = cv2.getPerspectiveTransform(initial, final)
    # print(transMatrix)
    dest_size = (inputImage.shape[1],inputImage.shape[0])
    inverseMap = cv2.warpPerspective(undistortImage, transMatrix, dest_size, flags=cv2.INTER_LINEAR)
    return inverseMap

def getEdges(inputImage):
    imageHist = cv2.calcHist([inputImage], [0], None, [256], [0, 256])
    upperThresh = 128 + np.argmax(imageHist[128:256])
    lowerThresh = np.argmax(imageHist[0:127])
    # print(upperThresh,lowerThresh)
    edgeImage =cv2.Canny(inputImage,lowerThresh,upperThresh,sobel_size,L2gradient = False)
    return edgeImage

def plotPoints(pointArray):
    endpoints = pointArray[:, :, :2].reshape(-1, 2)

    # Plot the endpoints using Matplotlib
    plt.scatter(endpoints[:, 0], endpoints[:, 1], color='red', marker='o')
    plt.title('Endpoints of Probabilistic Hough Transform Output')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()
    plt.show()

def getLines(inputImage):
    lines = cv2.HoughLinesP(inputImage,1,3.14/180,houghThresh,houghMin,houghMax)
    return lines

def displayLines(inputImage, lines):
    # Draw lines on the image
    # bgr_image = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(inputImage, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a red line
    return inputImage

def on_key_press(key):
    global houghThresh, houghMin, houghMax
    try:
        if key.char == '1':
            houghThresh += 5
            print(houghThresh)
        elif key.char == '2':
            houghThresh -= 5
            print(houghThresh)
        elif key.char == '3':
            houghMin += 5
            print(houghMin)
        elif key.char == '4':
            houghMin -= 5
            print(houghMin)
        elif key.char == '5':
            houghMax += 5
            print(houghMax)
        elif key.char == '6':
            houghMax -= 5
            print(houghMax)   

    except AttributeError:
        print(f'Special key {key} pressed')

def listen_for_keys():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()

# Create a separate thread for the listener
listener_thread = threading.Thread(target=listen_for_keys)

# Start the listener thread
# listener_thread.start()

class laneDetectNode():
        
        def __init__(self):
            """
            Creates a bridge for converting the image from Gazebo image intro OpenCv image
            """
            self.bridge = CvBridge()
            self.cv_image = np.zeros((640, 480))
            rospy.init_node('LaneAttemptnod', anonymous=True)
            self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
            self.waypoint_pub = rospy.Publisher("/lane/waypoints", Float32MultiArray, queue_size=3)
            self.rate = rospy.Rate(15)
            rospy.spin()

        def callback(self,data):
            waypoints = Float32MultiArray()
            dimension = MultiArrayDimension()
            dimension.label = "#ofwaypoints"
            dimension.size = 5
            waypoints.layout.dim = [dimension]
            waypoints.data = [0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 1.2, 1.2, 1.5, 1.5]
            self.waypoint_pub.publish(waypoints)
            return
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            c_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            roadImage = getIPM(self.cv_image)
            roadImage = getEdges(roadImage)
            lines = getLines(roadImage)
            roadImage = displayLines(getIPM(c_image),lines)
            # print("show img")
            cv2.imshow('IPM- Edges',roadImage)
            key = cv2.waitKey(1)
        
        
if __name__ == '__main__':
    try:
        nod = laneDetectNode()
        nod.rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass