import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from pynput import keyboard
import threading

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
kernel_size = (1,5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

def getIPM(inputImage):
    undistortImage = cv2.undistort(inputImage, cameraMatrix, distCoeff)
    transMatrix = cv2.getPerspectiveTransform(initial, final)
    # print(transMatrix)
    dest_size = (inputImage.shape[1],inputImage.shape[0])
    inverseMap = cv2.warpPerspective(undistortImage, transMatrix, dest_size, flags=cv2.INTER_LINEAR)
    return inverseMap

def plotPoints(pointArray):
    endpoints = pointArray[:, :, :2].reshape(-1, 2)
    sorted_indices = np.argsort(endpoints[:, 0])
    endpoints_sorted = endpoints[sorted_indices]
    for i in range(0,640,64):
        ranged_endpoints = endpoints_sorted[(endpoints_sorted[:,0] >= i) & (endpoints_sorted[:,0] <= (i+64))]
        coeff = np.polyfit(ranged_endpoints[:,0], ranged_endpoints[:,1],6,1)
        poly_f = np.poly1d(coeff)
        y_fit = poly_f(ranged_endpoints[:,0])
        # spl = UnivariateSpline(ranged_endpoints[:,0], ranged_endpoints[:,1], s = None)
        # print(ranged_endpoints,i)
        # Plot the endpoints using Matplotlib
        plt.scatter(ranged_endpoints[:, 0], ranged_endpoints[:, 1], color='red', marker='o')
        plt.plot(ranged_endpoints[:,0], y_fit(ranged_endpoints[:,0]),color='green')
        plt.title('Endpoints of Probabilistic Hough Transform Output')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.gca().invert_yaxis()
        plt.show()

# class laneDetectNode():
        
def __init__(self):
    """
    Creates a bridge for converting the image from Gazebo image intro OpenCv image
    """
    self.bridge = CvBridge()
    self.cv_image = np.zeros((640, 480))
    rospy.init_node('LaneAttemptnod', anonymous=True)
    self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
    rospy.spin()

def callback(self,data):
    self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
    #empty for now
    return


def getEdges(inputImage):
    imageHist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Figure out how to dynamically compute upper and lower thresh
    upperThresh = 128 + np.argmax(imageHist[128:256])
    lowerThresh = np.argmax(imageHist[0:127])-5
    # print(upperThresh,lowerThresh)
    edgeImage =cv2.Canny(inputImage,lowerThresh,upperThresh,sobel_size,L2gradient = False)
    return edgeImage
        
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

def getPoly(inputArray, windowSize):
    endpoints = inputArray[:, :, :2].reshape(-1, 2)
    sorted_indices = np.argsort(endpoints[:, 0])
    endpoints_sorted = endpoints[sorted_indices]
    print(endpoints_sorted)
    return

def on_key_press(key):
    global houghThresh, houghMin, houghMax
    try:
        if key.char == '1':
            print("Key '1' pressed")
            houghThresh += 5
        elif key.char == '2':
            print("Key '2' pressed")
            houghThresh -= 5
        elif key.char == '3':
            print("Key '3' pressed")
            houghMin += 5
        elif key.char == '4':
            print("Key '4' pressed")
            houghMin -= 5
        elif key.char == '5':
            print("Key '5' pressed")
            houghMax += 5
        elif key.char == '6':
            print("Key '6' pressed")
            houghMax -= 5   

    except AttributeError:
        print(f'Special key {key} pressed')

# Specify the path to your image
image_path = '/home/nash/Desktop/Simulator/saved_frames/test.png'

# Read the BGR image
bgr_image = cv2.imread(image_path)


def listen_for_keys():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()

# Create a separate thread for the listener
listener_thread = threading.Thread(target=listen_for_keys)

# Start the listener thread
# listener_thread.start()

# Check if the image is successfully loaded
if bgr_image is not None:
    # Convert BGR to grayscale
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    c_image = getIPM(bgr_image)
    image = getIPM(gray_image)
    edgeImage = getEdges(image)
    closed = cv2.morphologyEx(edgeImage,cv2.MORPH_DILATE,kernel)
    lines = getLines(closed)
    # plotPoints(lines)
    roadImage = displayLines(c_image, lines)
    cv2.imshow('EdgeImage', roadImage)
    cv2.imshow('Closed Image', closed)

    while True:
        key = cv2.waitKey(1)
        # Check for the 'Esc' key (key code 27)
        if key == 27:
            break

    cv2.destroyAllWindows()

else:
    print(f"Failed to load image from path: {image_path}")

