#include <ros/ros.h>
#include "attemptLane_2.hpp"

int main(int argc, char *argv[])
{
    std::vector<int> y_Values = {10,50,100,150,200,250};
    std::cout << "Starting the attemptLane_2 NODE..."<< std::endl;
    ros::init(argc, argv, "malosnode");
    std::cout << "node created";
    ros::NodeHandle nh;
    LaneDetectNode_2 detector;
    cv::Mat color_image = cv::imread("/home/nash/Desktop/Simulator/src/LaneDetection/src/lane_image.png", cv::IMREAD_GRAYSCALE); // Remember to update image path
    cv::Mat ipm_image = detector.getIPM(color_image);
    cv::Mat binary_image = detector.getLanes(ipm_image);
    std::cout << "Starting the line fit..."<< std::endl;
    std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret = line_fit_2(binary_image);
    detector.printTuple(ret);
    std::vector<double> waypoints = detector.getWaypoints(ret,y_Values);
    std::cout << "Waypoints are:"<< std::endl;
    for (int value : waypoints) {
        std::cout << value << " ";
    }
    std::cout << ""<< std::endl;
    std::vector<double> right_1  = std::get<2>(ret);
    std::vector<double> left_1  = std::get<1>(ret);
    plot_polynomial(right_1, left_1, waypoints, y_Values);
    cv::imshow("Binary Image", binary_image);
    cv::waitKey(0);
    return 0;
}