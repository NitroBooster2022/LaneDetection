#!/usr/bin/env python3

import rospy
import json
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from combined_thresh import combined_thresh
import Line
from line_fit import line_fit, tune_fit, calc_curve, calc_vehicle_offset, final_viz, viz1, viz2


class lane_attempt():
    
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('offset_calc', anonymous = False)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        self.mega_control = rospy.Publisher("/automobile/command", String, queue_size=1 )
        rospy.spin()

    def callback(self, data):
        # data is an gazebo image -> convert to cv2
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #gray_im = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        # perform binary and warp
        img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(self.cv_image)
        binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)
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
        # gyu_img = viz1(binary_warped, ret)
        # cv2.imshow("final preview", gyu_img)

if __name__ == '__main__':
    try:
        nod = lane_attempt()
    except rospy.ROSInterruptException:
        pass