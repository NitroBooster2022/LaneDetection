import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

lane_width =350

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
transMatrix = cv2.getPerspectiveTransform(final, initial)

# Camera matrix for accurate IPM transform
cameraMatrix = np.array([[CAMERA_PARAMS['fx'], 0, CAMERA_PARAMS['cx']],
                         [0, CAMERA_PARAMS['fy'], CAMERA_PARAMS['cy']],
                         [0, 0, 1]])

def find_center_indices(hist, threshold):
	"""
	Function to find the center of the lanes detected in an image

	Parameters:
	- Array containing histogram of said image
	- Threshold value for considered lane lines
	Returns:
	- array of center indices
	"""
	valid_groups = []

	# Find indices where histogram values are above the threshold
	above_threshold = np.where(hist > threshold)[0]

	# Find consecutive groups of five or more indices
	consecutive_groups = np.split(above_threshold, np.where(np.diff(above_threshold) != 1)[0] + 1)

	# Filter groups with five or more consecutive indices
	# valid_groups = [group for group in consecutive_groups if len(group) >= 5]

	# Iterate over consecutive_groups, check for stop line
	for group in consecutive_groups:
		if len(group) >= 5:
			valid_groups.append(group)

	# Find the center index for each valid group
	center_indices = [(group[0] + group[-1]) // 2 for group in valid_groups]

	return center_indices

def find_stop_line(image, threshold):
	"""
	Function to detect if there is a stop line on the image
	Parameters:
	- Array of image we want to detect on
	- Threshold for the histogram that will be applied on the image
	"""
	# Find indices where histogram values are above the threshold
	histogram = np.sum(image[0:480,:]//2, axis = 0)
	above_threshold = np.where(histogram > threshold)[0]
	stop_line = False
	# Find consecutive groups of five or more indices
	consecutive_groups = np.split(above_threshold, np.where(np.diff(above_threshold) != 1)[0] + 1)

	horistogram = np.sum(image[:,0:640]//2, axis = 1)
	max_index = np.argmax(horistogram)
	# print(max_index)
	# plt.plot(horistogram)
	# plt.show()

	width = 0
	# check to see if there is a sequence of pixels long enough for stop line
	for group in consecutive_groups:
		if len(group) >= 370:
			stop_line = True
			above_threshold2 = np.where(horistogram > 50000)[0]
			if len(above_threshold2) > 0:
				width = abs(above_threshold2[-1] - above_threshold2[0])
			else:
				width = 0  

	return stop_line, max_index, width

def check_cross_walk(image, stop_index):
	"""
	Function to detect if there is a crosswalk, density approach
	Should only be called if there is a stop line that is detected

	Parameters:
	- Array of image we want to detect on
	- Index of detected stop line
	"""
	# Ratio of the number of non white pixels over the number of total pixels
	density = np.count_nonzero(image[0:stop_index]) / (image[0:stop_index].shape[0] * image[0:stop_index].shape[1])
	if(density > 0.3):
		return True
	else:
		return False

def find_closest_pair(arr, lane_width):
	"""
	Function to find the two lane markings which distance between them is most likely to be that of an actual lane marking

	Parameters:
	- Array containing values of different lane marking center
	- Expected value of the distance between lane markings
	Returns:
	- Center indices of the lane markings
	"""

	n = len(arr)

	if n < 2:
		raise ValueError("Array must have at least two elements")

	min_diff = np.inf
	result_pair = (arr[0], arr[1])

	for i in range(n - 1):
		for j in range(i + 1, n):
			current_diff = abs(abs(arr[i] - arr[j]) - lane_width)
			if current_diff < min_diff:
				min_diff = current_diff
				result_pair = (arr[i], arr[j])

	# print(result_pair)
	return result_pair

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the image
	# histogram = np.sum(binary_warped[:,:]//2, axis=0)
	histogram = np.sum(binary_warped[200:480,:]//2, axis = 0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	# midpoint = np.int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[0:midpoint])
	# rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# print(midpoint,leftx_base, rightx_base)
	# for i in range(640):
	# 	if histogram[i]>=1500:
	# 		leftx_base = i
	# 		break
	# for i in range(640):
	# 	if histogram[639-i]>=1500:
	# 		rightx_base = 639-i
	# 		break
	# print(midpoint,leftx_base, rightx_base)
	# plt.plot(histogram)
	# plt.show()
	# Choose the number of sliding windows
	threshold = 5000 # Magic number for constant discovered by Antoine in 2023 AD
	stop_line, stop_index, width = find_stop_line(binary_warped,threshold)		# check for stop line and its location
	indices= find_center_indices(histogram,threshold+width*125)	# find the center of possible lane markings
	# print(threshold+width*255)
	
	if(stop_line):
		cross_walk = check_cross_walk(binary_warped, stop_index)		# if there is a stop line, check for crosswalk

	# print(indices)

	if(len(indices) == 0): # If no lane markings are found
		ret = {}
		ret['out_img'] = out_img
		ret['number_of_fits'] = '0'
		ret['stop_line'] = False
		ret['stop_index'] = stop_index
		if(stop_line):
			ret['cross_walk'] = cross_walk
		return ret
	
	if(len(indices) == 1): # if only one lane marking is found
		ret = {}
		if(indices[0] < 320):	 # if lane marking is on the left side of the image
			ret['number_of_fits'] = 'left'
		else:					 # if lane marking is on the right side of the image
			ret['number_of_fits'] = 'right'
		leftx_base = indices[0]
		rightx_base = indices[0]

	else:						 # two or more lane markings found
		ret = {}
		(leftx_base, rightx_base) = find_closest_pair(indices,lane_width)
		# print(leftx_base, rightx_base)
		ret['number_of_fits'] = '2'

	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 50
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# print('current left - ' + str(leftx_current))
		# Draw the windows on the visualization image
		# cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		# cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			# print('mean of left indices - ' + str(np.mean(nonzerox[good_left_inds])))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			#print('current right - '+str(rightx_current))
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	(nonzerox[left_lane_inds])
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Return a dict of relevant variables
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds
	ret['stop_line'] = stop_line
	ret['stop_index'] = stop_index
	if(stop_line):
		ret['cross_walk'] = cross_walk

	return ret


def tune_fit(binary_warped, left_fit, right_fit, stop_line):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 50
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	ret = {}
	if lefty.shape[0] > min_inds:
		# left_fit = np.polyfit(lefty, leftx, 2)
		ret['number_of_fits'] = 'left'

	if righty.shape[0] > min_inds:
		# right_fit = np.polyfit(righty, rightx, 2)
		ret['number_of_fits'] = 'right'

	if lefty.shape[0] > min_inds and righty.shape[0] > min_inds:
		ret['number_of_fits'] = '2'

	if lefty.shape[0] < min_inds and righty.shape[0] < min_inds:
		ret['number_of_fits'] = '0'
		return ret
	
	# Fit a second order polynomial to each
	if ret['number of fits'] == '2':
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

	elif ret['number of fits'] == 'left':
		left_fit = np.polyfit(lefty, leftx, 2)
	
	elif ret['number of fits'] == 'right':
		left_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds
	ret['stop_line'] = stop_line
	return ret


def viz3(binary_warped, non_warped, ret, waypoints, y_Values, IPM = True):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	"""
	# Grab variables from ret dictionary
	if ret is not None:
		left_fit = ret.get('left_fit', None)
		right_fit = ret.get('right_fit', None)
		nonzerox = ret.get('nonzerox', None)
		nonzeroy = ret.get('nonzeroy', None)
		left_lane_inds = ret.get('left_lane_inds', None)
		right_lane_inds = ret.get('right_lane_inds', None)
		stop_line = ret.get('stop_line', None)
		stop_index = ret.get('stop_index', None)

		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

		# Create an empty image
		# img_shape = (binary_warped.shape[0], binary_warped.shape[1], 3)
		# result = np.zeros(img_shape, dtype=np.uint8)

	# 	# Create an image to draw the lines on
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		result = np.dstack((warp_zero, warp_zero, warp_zero))
		result = np.zeros((480, 640, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

		if (IPM) :result = cv2.addWeighted(binary_warped, 1, result, 0.3, 0)

		# Update values only if they are not None
		if left_fit is not None:
			left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
		else:
			left_fitx = None
		if right_fit is not None:
			right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
		else:
			right_fitx = None

	else:
		return None;

	if left_fitx is not None:
		cv2.polylines(result, np.int32([np.column_stack((left_fitx, ploty))]), isClosed=False, color=(255, 255, 0), thickness=15)
	if right_fitx is not None:
		cv2.polylines(result, np.int32([np.column_stack((right_fitx, ploty))]), isClosed=False, color=(255, 255, 0), thickness=15)

	for i in range(len(y_Values)):
		# print(waypoints[i])
		x = int(waypoints[i])
		y = int(y_Values[i])
		cv2.circle(result, (x, y), 5, (0, 255, 0), -1)  # Draw a filled green circle

	# Draw stop line
	if(stop_line):
		# print("stopline")
		cv2.line(result, (int(0),int(stop_index)), (int(639),int(stop_index)), color=(0, 0, 255), thickness = 2)
	
	if (not IPM):
		dest_size = (binary_warped.shape[1],binary_warped.shape[0])
		result = cv2.warpPerspective(result, transMatrix, dest_size, flags=cv2.INTER_LINEAR)
		result = cv2.addWeighted(result, 1, non_warped, 0.5, 0)


	if ret.get('stop_line', None)==True:
		cv2.putText(result, 'Stopline detected!', (int(64),int(48)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
	
	if ret.get('cross_walk', None)==True:
		cv2.putText(result, 'Crosswalk detected!', (int(128),int(96)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
	
	return result

# def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
# 	"""
# 	Final lane line prediction visualized and overlayed on top of original image
# 	"""
# 	# Generate x and y values for plotting
# 	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
# 	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# 	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# 	# Create an image to draw the lines on
# 	#warp_zero = np.zeros_like(warped).astype(np.uint8)
# 	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
# 	color_warp = np.zeros((640, 480, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

# 	# Recast the x and y points into usable format for cv2.fillPoly()
# 	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# 	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# 	pts = np.hstack((pts_left, pts_right))

# 	# Draw the lane onto the warped blank image
# 	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# 	# Warp the blank back to original image space using inverse perspective matrix (Minv)
# 	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
# 	# Combine the result with the original image
# 	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

# 	# Annotate lane curvature values and vehicle offset from center
# 	avg_curve = (left_curve + right_curve)/2
# 	label_str = 'Radius of curvature: %.1f m' % avg_curve
# 	result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

# 	label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
# 	result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

# 	return result

