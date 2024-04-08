#include <ros/ros.h>
#include <opencv2/opencv.hpp>
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
// #include <chrono>
// using namespace std::chrono;


class LaneDetectNode_2{
    public:
        LaneDetectNode_2(){
            // image_sub = it.subscribe("/camera/color/image_raw", 1, &LaneDetector::imageCallback, this);
            // // image_pub = it.advertise("/automobile/image_modified", 1);
            // lane_pub = nh.advertise<utils::Lane>("/lane", 1);
            // image = cv::Mat::zeros(480, 640, CV_8UC1);
            bool stopline = false;
            bool dotted = false;
            // ros::Rate rate(40); 
            // while (ros::ok()) {
                // ros::spinOnce();
                // rate.sleep();
            // }
        }
        // ros::NodeHandle nh;
        // image_transport::ImageTransport it;
        // image_transport::Subscriber image_sub;
        // image_transport::Publisher image_pub;
        // ros::Publisher lane_pub;

            // Declare CAMERA_PARAMS as a constant global variable
        const std::map<std::string, double> CAMERA_PARAMS = {
            {"fx", 554.3826904296875},
            {"fy", 554.3826904296875},
            {"cx", 320},
            {"cy", 240}
        };

        // Define initial coordinates of input image as a constant global variable
        const cv::Mat initial = (cv::Mat_<float>(4, 2) <<
            0, 300,
            640, 300,
            0, 480,
            640, 480
        );

        // Define where the initial coordinates will end up on the final image as a constant global variable
        const cv::Mat final = (cv::Mat_<float>(4, 2) <<
            0, 0,
            640, 0,
            0, 480,
            640, 480
        );

        // Compute the transformation matrix
        cv::Mat transMatrix = cv::getPerspectiveTransform(initial, final);

        // Define camera matrix for accurate IPM transform as a constant global variable
        const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            CAMERA_PARAMS.at("fx"), 0, CAMERA_PARAMS.at("cx"),
            0, CAMERA_PARAMS.at("fy"), CAMERA_PARAMS.at("cy"),
            0, 0, 1
        );

        // Define distortion coefficients as an empty constant global variable
        const cv::Mat distCoeff = cv::Mat();

        //------------------Declare Global Functions ---------------------------------//

        cv::Mat getIPM(cv::Mat inputImage) {
            cv::Mat undistortImage;
            cv::undistort(inputImage, undistortImage, cameraMatrix, distCoeff);
            cv::Mat inverseMap;
            cv::Size dest_size(inputImage.cols, inputImage.rows);
            cv::warpPerspective(undistortImage, inverseMap, transMatrix, dest_size, cv::INTER_LINEAR);
            return inverseMap;
        }

        cv::Mat getLanes(cv::Mat inputImage) {
            cv::Mat imageHist;
            cv::calcHist(&inputImage, 1, 0, cv::Mat(), imageHist, 1, &inputImage.rows, 0);
            double minVal, maxVal;
            cv::minMaxLoc(imageHist, &minVal, &maxVal);
            int threshold_value = std::min(std::max(static_cast<int>(maxVal) - 75, 30), 200);
            cv::Mat binary_thresholded;
            cv::threshold(inputImage, binary_thresholded, threshold_value, 255, cv::THRESH_BINARY);
            return binary_thresholded;
        }

        std::vector<double> getWaypoints(std::map<std::string, std::string> wayLines, std::vector<double> y_Values) {
            int offset = 175;
            std::vector<double> wayPoint(y_Values.size(), 0);
            if (wayLines["number_of_fits"] == "2") {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    double x_right = wayLines["right_fit"][0] * pow(y_Values[i], 2) + wayLines["right_fit"][1] * y_Values[i] + wayLines["right_fit"][2];
                    double x_left = wayLines["left_fit"][0] * pow(y_Values[i], 2) + wayLines["left_fit"][1] * y_Values[i] + wayLines["left_fit"][2];
                    wayPoint[i] = 0.5 * (x_right + x_left);
                    wayPoint[i] = std::min(std::max(wayPoint[i], 0.0), 639.0);
                }
            } else if (wayLines["number_of_fits"] == "left") {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    wayPoint[i] = wayLines["left_fit"][0] * pow(y_Values[i], 2) + wayLines["left_fit"][1] * y_Values[i] + wayLines["left_fit"][2] + offset;
                    wayPoint[i] = std::min(std::max(wayPoint[i], 0.0), 639.0);
                }
            } else if (wayLines["number_of_fits"] == "right") {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    wayPoint[i] = wayLines["right_fit"][0] * pow(y_Values[i], 2) + wayLines["right_fit"][1] * y_Values[i] + wayLines["right_fit"][2] - offset;
                    wayPoint[i] = std::min(std::max(wayPoint[i], 0.0), 639.0);
                }
            } else if (wayLines["stop_line"] == "0") {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    wayPoint[i] = 320;
                }
            } else {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    wayPoint[i] = 320;
                }
            }
            for (size_t i = 0; i < wayPoint.size(); ++i)
                std::cout << wayPoint[i] << " ";
            std::cout << std::endl;
            return wayPoint;
        }
};

int main(int argc, char *argv[])
{
    /* code */
    ros::init(argc, argv, "malosnode");
    ros::NodeHandle nh;
    LaneDetectNode_2 detector;
    cv::Mat color_image = cv::imread("/home/nash/Desktop/Simulator/src/LaneDetection/src/lane_image.png", cv::IMREAD_GRAYSCALE); // Remember to update image path
    cv::Mat ipm_image = detector.getIPM(color_image);
    cv::Mat binary_image = detector.getLanes(ipm_image);
    cv::imshow("Binary Image", binary_image);
    cv::waitKey(0);
    return 0;
}
