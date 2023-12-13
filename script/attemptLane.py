#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from pynput import keyboard
import threading
from Line import Line
from line_fit import line_fit, tune_fit, calc_curve, calc_vehicle_offset, final_viz, viz1, viz2, viz3

# #-----Declare Global Variables ----- #
CAMERA_PARAMS = {'fx': 554.3826904296875, 'fy': 554.3826904296875, 'cx': 320, 'cy': 240}
initial = np.float32([[0,300],
                      [640,300],
                      [0,480],
                      [640,480]])

final = np.float32([[0,0],
                    [640,0],
                    [0,480],
                    [640,480]])

transMatrix = cv2.getPerspectiveTransform(initial, final)
# print(transMatrix)

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
    dest_size = (inputImage.shape[1],inputImage.shape[0])
    inverseMap = cv2.warpPerspective(undistortImage, transMatrix, dest_size, flags=cv2.INTER_LINEAR)
    return inverseMap

def getEdges(inputImage):
    imageHist = cv2.calcHist([inputImage], [0], None, [256], [0, 256])
    upperThresh = 128 + np.argmax(imageHist[128:256])
    lowerThresh = np.argmax(imageHist[0:127])
    threshold_value = np.clip(np.max(inputImage) - 55, 30, 200)
    _, binary_thresholded = cv2.threshold(inputImage, threshold_value, 255, cv2.THRESH_BINARY)

    # print(upperThresh,lowerThresh)
    # edgeImage =cv2.Canny(inputImage,lowerThresh,upperThresh,sobel_size,L2gradient = False)
    return binary_thresholded

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
            self.depth_image = np.zeros((640, 480))
            self.normalized_depth_image = np.zeros((640, 480))
            rospy.init_node('LaneAttemptnod', anonymous=True)
            self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
            # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depthcallback)
            rospy.spin()

        def depthcallback(self,data):
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            min_val, max_val, _, _ = cv2.minMaxLoc(self.depth_image)
            self.normalized_depth_image = cv2.convertScaleAbs(self.depth_image, alpha=255.0 / (max_val - min_val), beta=-min_val * 255.0 / (max_val - min_val))
            # print(data.header)
            # print(self.depth_image[479, 639])
            # cv2.imshow("Depth", self.normalized_depth_image)
            # key = cv2.waitKey(1)

        def callback(self,data):
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            c_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            roadImage = getIPM(self.cv_image)
            binary_warped = getEdges(roadImage)
            cv2.imshow("Warped preview", binary_warped)
            key = cv2.waitKey(1)
            window_size = 2  # how many frames for line smoothing
            left_line = Line(n=window_size)
            right_line = Line(n=window_size)
            detected = False  # did the fast line fit detect the lines?
                # Perform polynomial fit
            if not detected:
                # Slow line fit
                ret = line_fit(binary_warped)
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                left_lane_inds = ret['left_lane_inds']
                right_lane_inds = ret['right_lane_inds']

                # Get moving average of line fit coefficients
                left_fit = left_line.add_fit(left_fit)
                right_fit = right_line.add_fit(right_fit)

                # Calculate curvature
                left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

                detected = True  # slow line fit always detects the line

            else:  # implies detected == True
                # Fast line fit
                left_fit = left_line.get_fit()
                right_fit = right_line.get_fit()
                ret = tune_fit(binary_warped, left_fit, right_fit)
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                left_lane_inds = ret['left_lane_inds']
                right_lane_inds = ret['right_lane_inds']

                # Only make updates if we detected lines in current frame
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = left_line.add_fit(left_fit)
                    right_fit = right_line.add_fit(right_fit)
                    left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
                else:
                    detected = False


            # gyu_img = final_viz(self.cv_image, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
            gyu_img = viz3(binary_warped, ret)
            cv2.imshow("final preview", gyu_img)
        
        
if __name__ == '__main__':
    try:
        nod = laneDetectNode()
    except rospy.ROSInterruptException:
        pass