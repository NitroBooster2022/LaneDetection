#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import PointStamped as TF2PointStamped
import math
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

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
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depthcallback)
        self.waypoint_pub = rospy.Publisher("/lane/waypoints", Float32MultiArray, queue_size=3)
        rospy.spin()

    def depthcallback(self, data):
        # self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        # min_val, max_val, _, _ = cv2.minMaxLoc(self.depth_image)
        # self.normalized_depth_image = cv2.convertScaleAbs(self.depth_image, alpha=255.0 / (max_val - min_val), beta=-min_val * 255.0 / (max_val - min_val))
        # print(data.header)
        # cv2.imshow("Depth", self.normalized_depth_image)
        # key = cv2.waitKey(1)

        self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")

        # x = 320
        # y = 180
        # depth_value = self.depth_image[y, x]/1000

        # camera_point = PointStamped()
        # camera_point.header = data.header
        # camera_point.point.x = (x - CAMERA_PARAMS['cx']) * depth_value / CAMERA_PARAMS['fx']
        # camera_point.point.y = (y - CAMERA_PARAMS['cy']) * depth_value / CAMERA_PARAMS['fy']
        # camera_point.point.z = depth_value

        # print(f"Point in Camera Frame: ({camera_point.point.x}, {camera_point.point.y}, {camera_point.point.z})")

        # R = np.array([[1, 0, 0.15],
        #             [0, 1, 0],
        #             [0.15, 0, 1]])
        
        # map_x = R[0, 0] * camera_point.point.x + R[0, 1] * camera_point.point.y + R[0, 2] * camera_point.point.z
        # map_y = R[1, 0] * camera_point.point.x + R[1, 1] * camera_point.point.y + R[1, 2] * camera_point.point.z
        # map_z = R[2, 0] * camera_point.point.x + R[2, 1] * camera_point.point.y + R[2, 2] * camera_point.point.z
        
        # map_y = math.sqrt(math.pow(depth_value,2)-math.pow(0.16,2))

        # print(f"Point in Map Frame: ({map_x}, {map_y}, {map_z})")

    def callback(self,data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        c_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        roadImage = getIPM(self.cv_image)
        # print(self.pixel_to_world(639,0))
        waypoints = Float32MultiArray()
        dimension = MultiArrayDimension()
        dimension.label = "#ofwaypoints"
        dimension.size = 4
        waypoints.layout.dim = [dimension]
        wp1 = self.pixel_to_world(0,0)
        wp2 = self.pixel_to_world(0,479)
        wp3 = self.pixel_to_world(639,0)
        wp4 = self.pixel_to_world(639,479)
        waypoints.data = [wp1[1], -wp1[0], wp2[1], -wp2[0], wp3[1], -wp3[0], wp4[1], -wp4[0]]
        self.waypoint_pub.publish(waypoints)

    def pixel_to_world(self,x,y):
        original_pixel_coord = cv2.perspectiveTransform(np.array([[[x, y]]], dtype='float32'), np.linalg.inv(transMatrix))[0][0].astype(int)
        # print(original_pixel_coord)
        depth_value = self.depth_image[original_pixel_coord[1], original_pixel_coord[0]]/1000
        map_y = math.sqrt(math.pow(depth_value,2)-math.pow(0.16,2))
        map_x = (original_pixel_coord[0] - CAMERA_PARAMS['cx']) * depth_value / CAMERA_PARAMS['fx'] + 0.15 * depth_value
        return np.array([map_x,map_y])

if __name__ == '__main__':
    try:
        nod = laneDetectNode()
    except rospy.ROSInterruptException:
        pass