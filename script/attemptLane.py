#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# import matplotlib.pyplot as plt
import math
from Line import Line
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from line_fit import line_fit, tune_fit, viz3
import timeit
from utils.msg import Lane

# #-----Declare Global Variables ----- #

CAMERA_PARAMS = {'fx': 554.3826904296875, 'fy': 554.3826904296875, 'cx': 320, 'cy': 240} # Camera parameters - need to calibrate

# Initial coordinates of input image
initial = np.float32([[0,300],
                      [640,300],
                      [0,480],
                      [640,480]])

# Where the initial coordinates will end up on the final image
final = np.float32([[0,0],
                    [640,0],
                    [0,480],
                    [640,480]])

# Compute the transformation matix
transMatrix = cv2.getPerspectiveTransform(initial, final)

# Camera matrix for accurate IPM transform
cameraMatrix = np.array([[CAMERA_PARAMS['fx'], 0, CAMERA_PARAMS['cx']],
                         [0, CAMERA_PARAMS['fy'], CAMERA_PARAMS['cy']],
                         [0, 0, 1]])

distCoeff = np.array([])

# # ----- Declare global functions ------ # # 

def getIPM(inputImage):
    """
    Calculate the inverse perspective transform of the input image

    Parameters:
    - input image
    - Matrix with parameters of the camera
    - Matrix with the desired perspective transform
    Returns:
     - Persepctive transformed image
    """
    undistortImage = cv2.undistort(inputImage, cameraMatrix, distCoeff)
    dest_size = (inputImage.shape[1],inputImage.shape[0])
    inverseMap = cv2.warpPerspective(undistortImage, transMatrix, dest_size, flags=cv2.INTER_LINEAR)
    return inverseMap

def getLanes(inputImage):
    """
    Compute the lane lines of a given input image

    Parameters:
    - Input image to compute edges for
    Returns:
    - Binary image of lane lines
    """ 
    imageHist = cv2.calcHist([inputImage], [0], None, [256], [0, 256])
    threshold_value = np.clip(np.max(inputImage) - 75, 30, 200)
    _, binary_thresholded = cv2.threshold(inputImage, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_thresholded


def getWaypoints(wayLines, y_Values):
    """
    Calculate the location of the waypoints

    Parameters:
    - Array containing polynomial fits of left an right curves
    - Array containing Y values to compute waypoints for
    Returns:
    - X coordinate location of waypoints
    """
    offset = 175
    wayPoint = np.zeros(len(y_Values))
    # print(wayLines['number_of_fits'])
    if(wayLines['number_of_fits'] == '2'):
        for i in range(len(y_Values)):
                x_right = wayLines['right_fit'][0] * y_Values[i]**2 + wayLines['right_fit'][1] * y_Values[i] + wayLines['right_fit'][2]
                x_left = wayLines['left_fit'][0] * y_Values[i]**2 + wayLines['left_fit'][1] * y_Values[i] + wayLines['left_fit'][2]
                wayPoint[i] = 0.5*(x_right + x_left)
                wayPoint[i] = np.clip(wayPoint[i], 0, 639)
    
    elif(wayLines['number_of_fits'] == 'left'):
            for i in range(len(y_Values)):
                wayPoint[i] = wayLines['left_fit'][0] * y_Values[i]**2 + wayLines['left_fit'][1] * y_Values[i] + wayLines['left_fit'][2] + offset
                wayPoint[i] = np.clip(wayPoint[i], 0, 639)
                # print(wayPoint)

    elif(wayLines['number_of_fits'] == 'right'):
            for i in range(len(y_Values)):
                wayPoint[i] = wayLines['right_fit'][0] * y_Values[i]**2 + wayLines['right_fit'][1] * y_Values[i] + wayLines['right_fit'][2] - offset
                wayPoint[i] = np.clip(wayPoint[i], 0, 639)
    elif(wayLines['stop_line']):
        for i in range(len(y_Values)):
                wayPoint[i] = 320
                
    else:
            for i in range(len(y_Values)):
                 wayPoint[i] = 320
    print(wayPoint)
    return wayPoint


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
            self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
            # self.waypoint_pub = rospy.Publisher("/lane/waypoints", Float32MultiArray, queue_size=3)
            self.lane_pub = rospy.Publisher("/lane", Lane, queue_size=3)
            # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depthcallback)
            # self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depthcallback)
            # self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depthcallback)
            self.detected = False  # did the fast line fit detect the lines?
            window_size = 2  # how many frames for line smoothing
            self.left_line = Line(n=window_size)
            self.right_line = Line(n=window_size)
            self.stop_line = False
            self.cross_walk = False
            self.lane_msg = Lane()
            # self.refresh = 0       # Counter to refresh the line detection
            # self.refresh = rospy.Timer(rospy.Duration(1), self.slow_detect)
            rospy.spin()

        def slow_detect(self, event):
            self.detected = False
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
            binary_warped = getLanes(roadImage)
            # cv2.imshow("Warped preview", roadImage)
            key = cv2.waitKey(1)
                # Perform polynomial fit
            if not self.detected:
                # print('SLOW')
                # t1 = timeit.default_timer()
                # Slow line fit
                ret = line_fit(binary_warped)
                left_fit = ret.get('left_fit', None)
                right_fit = ret.get('right_fit', None)
                self.stop_line = ret['stop_line']
                if(self.stop_line):
                     self.cross_walk = ret['cross_walk']
                nonzerox = ret.get('nonzerox', None)
                nonzeroy = ret.get('nonzeroy', None)
                left_lane_inds = ret.get('left_lane_inds', None)
                right_lane_inds = ret.get('right_lane_inds', None)

                # Update values only if they are not None
                if left_fit is not None:
                    left_fit = self.left_line.add_fit(left_fit)
                if right_fit is not None:
                    right_fit = self.right_line.add_fit(right_fit)

                # if(self.stop_line):
                    #  print(self.cross_walk)
                # # Get moving average of line fit coefficients
                # left_fit = left_line.add_fit(left_fit)
                # right_fit = right_line.add_fit(right_fit)

                # Calculate curvature
                # left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

                # self.detected = True  # slow line fit always detects the line

            else:  # implies detected == True
                # Fast line fit
                # print('FAST')
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(binary_warped, left_fit, right_fit,self.stop_line)

            if ret is not None:
                left_fit = ret.get('left_fit', None)
                right_fit = ret.get('right_fit', None)
                # nonzerox = ret.get('nonzerox', None)
                # nonzeroy = ret.get('nonzeroy', None)
                # left_lane_inds = ret.get('left_lane_inds', None)
                # right_lane_inds = ret.get('right_lane_inds', None)
                # number_of_fits = ret['number_of_fits']

                # Update values only if they are not None
                if left_fit is not None:
                    left_fit = self.left_line.add_fit(left_fit)
                # else: 
                #     self.refresh +=1
                if right_fit is not None:
                    right_fit = self.right_line.add_fit(right_fit)
                # else: 
                #     self.refresh +=1

            # left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
            else:
                self.detected = False
            # print(self.refresh)
            # if (self.refresh >= 20):
            #      self.reresh = 0
            #      self.detected = False
            # print(ret)
            y_Values = np.array([10,50,100,150,200,250])
            wayPoint = getWaypoints(ret,y_Values)
            gyu_img = viz3(getIPM(c_image),c_image, ret,wayPoint,y_Values, False)
            cv2.imshow("final preview", gyu_img)       # binary_warped = getLanes(roadImage)
            # cv2.imshow("Warped preview", binary_warped)
            # Publish waypoints corresponding to the IPM transformed image pixels
            waypoints = Float32MultiArray()
            dimension = MultiArrayDimension()
            dimension.label = "#ofwaypoints"
            dimension.size = 6
            waypoints.layout.dim = [dimension]
            # print(wayPoint)
            wp1 = self.pixel_to_world(wayPoint[0],10)
            wp2 = self.pixel_to_world(wayPoint[1],50)
            wp3 = self.pixel_to_world(wayPoint[2],100)
            wp4 = self.pixel_to_world(wayPoint[3],150)
            wp5 = self.pixel_to_world(wayPoint[4],200)
            wp6 = self.pixel_to_world(wayPoint[5],250)
            waypoints.data = [wp1[1], -wp1[0], wp2[1], -wp2[0], wp3[1], -wp3[0], wp4[1], -wp4[0], wp5[1], -wp5[0], wp6[1], -wp6[0]]
            self.waypoint_pub.publish(waypoints)
            # print(timeit.default_timer()-t1)

        # Convert IPM pixel coordinates to world coordinates (relative to camera)
        # Depends on IPM tranform matrix and height and orientation of the camera
        def pixel_to_world(self,x,y):
            height = 0.16
            roll = 0.15
            original_pixel_coord = cv2.perspectiveTransform(np.array([[[x, y]]], dtype='float32'), np.linalg.inv(transMatrix))[0][0].astype(int)
            # print(original_pixel_coord)
            depth_value = self.depth_image[original_pixel_coord[1], original_pixel_coord[0]]/1000
            # print(depth_value)
            if depth_value < 0.03:
                return np.array([0,0])
            map_y = math.sqrt(math.pow(depth_value,2)-math.pow(height,2))
            map_x = (original_pixel_coord[0] - CAMERA_PARAMS['cx']) * depth_value / CAMERA_PARAMS['fx'] + roll * depth_value
            return np.array([map_x,map_y])
        
if __name__ == '__main__':
    try:
        nod = laneDetectNode()
    except rospy.ROSInterruptException:
        pass