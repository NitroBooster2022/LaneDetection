#include <ros/ros.h>
#include <opencv2/opencv.hpp>
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <chrono>
// using namespace std::chrono;
// #include "line_fit_2.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include "interpolation.h"
#include "stdafx.h"
#include <iostream>
#include "matplotlibcpp.h"
#include "cv_bridge/cv_bridge.h"
#include "utils/Lane.h"
#include <image_transport/image_transport.h>

namespace plt = matplotlibcpp;


class LaneDetectNode_2{
    public:
        LaneDetectNode_2(ros::NodeHandle& nh) :
         it(nh){
            image_sub = it.subscribe("/camera/color/image_raw", 1, &LaneDetectNode_2::imageCallback, this);
            depth_sub = it.subscribe("/camera/depth/image_raw", 1, &LaneDetectNode_2::depthCallback, this);
            // // image_pub = it.advertise("/automobile/image_modified", 1);
            lane_pub = nh.advertise<std_msgs::Float32MultiArray>("/lane/waypoints", 1);
            // image = cv::Mat::zeros(480, 640, CV_8UC1);
            bool stopline = false;
            bool dotted = false;
            ros::Rate rate(40); 
            while (ros::ok()) {
                ros::spinOnce();
                rate.sleep();
            }
        }

        // ROS OBJECTS
        ros::NodeHandle nh;
        image_transport::ImageTransport it;
        image_transport::Subscriber image_sub;
        image_transport::Subscriber depth_sub;
        image_transport::Publisher image_pub;
        ros::Publisher lane_pub;
        cv_bridge::CvImagePtr cv_ptr;
        cv_bridge::CvImagePtr cv_ptr_depth = nullptr;
        cv::Mat normalizedDepthImage;
        std::vector<int> y_Values = {475,450,420,400,350,300};

        // IMAGE CONTAINERS
        cv::Mat depth_image;
        cv::Mat image_cont_1;
        cv::Mat image_cont_2;
        cv::Mat image_cont_3;
        cv::Mat grayscale_image;
        cv::Mat color_image;
        cv::Mat horistogram;

        // VECTOR CONTAINERS
        std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret;
        std::vector<double> waypoints;
        std::vector<double> right_1;
        std::vector<double> left_1;
        std_msgs::Float32MultiArray waypoints_msg;
        std_msgs::MultiArrayDimension dimension;

        // STATIC VARIABLES
        double maxVal;
        double minVal;
        double height = 0.16;
        double roll = 0.15;
        double depthy;
        double laneMaxVal;
        double laneMinVal;


        // LINE FIT
        int lane_width = 350;        // HARD CODED LANE WIDTH
        int n_windows = 9;           // HARD CODED WINDOW NUMBER FOR LANE PARSING
        int threshold = 2000;       // HARD CODED THRESHOLD
        int leftx_base = 0;
        int rightx_base = 640;
        int number_of_fits = 0;
        bool stop_line = false;
        int stop_index = 0;
        bool cross_walk = false;
        int width = 370;
        int stop_loc = -1;


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
        cv::Mat invMatrix = cv::getPerspectiveTransform(final, initial);

        // Define camera matrix for accurate IPM transform as a constant global variable
        const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            CAMERA_PARAMS.at("fx"), 0, CAMERA_PARAMS.at("cx"),
            0, CAMERA_PARAMS.at("fy"), CAMERA_PARAMS.at("cy"),
            0, 0, 1
        );

        // Define distortion coefficients as an empty constant global variable
        const cv::Mat distCoeff = cv::Mat();

        //------------------Declare Global Functions ---------------------------------//
        void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
            ros::Time start_time = ros::Time::now(); 

            // ROS_INFO("Image received");
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // ROS_INFO("Image converted");
            color_image = cv_ptr->image;
            cv::cvtColor(color_image, grayscale_image, cv::COLOR_BGR2GRAY);
            // ROS_INFO("Image converted to cv::Mat");
            // image_cont_1 = getIPM(color_image);
            image_cont_2 = getIPM(grayscale_image);
            
            // ROS_INFO("IPM image created");
            image_cont_3 = getLanes(image_cont_2);
            
            // ROS_INFO("Binary image created");
            
            ret = line_fit_2(image_cont_3);
            
            // ROS_INFO("Line fit done");
            // printTuple(ret);
            waypoints = getWaypoints(ret,y_Values);

            right_1  = std::get<2>(ret);
            left_1  = std::get<1>(ret);
            // plot_polynomial(right_1, left_1, waypoints, y_Values);
            // Set dimension label and size
            dimension.label = "#ofwaypoints";
            dimension.size = 1;
            waypoints_msg.layout.dim.push_back(dimension);

            // Populate data array with waypoints
            waypoints_msg.data.push_back(waypoints[5]);

            // Publish the message
            lane_pub.publish(waypoints_msg);

            // End time
            
            ros::Time end_time = ros::Time::now();
            ros::Duration elapsed_time = end_time - start_time;
                        
            // Calculate elapsed time
            
            // Print elapsed time in seconds
            ROS_INFO("Elapsed time: %.3f seconds", elapsed_time.toSec());

        }

        void depthCallback(const sensor_msgs::ImageConstPtr &msg) {
            cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv_ptr_depth->image.convertTo(normalizedDepthImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            if(cv_ptr_depth == nullptr || cv_ptr_depth->image.empty()) {
                ROS_ERROR("cv_bridge failed to convert image");
                return;
            }
            // std::cout << "-------------Depth done-----------------" << std::endl;
            // try {
            //     cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            // } catch (cv_bridge::Exception &e) {
            //     ROS_ERROR("cv_bridge exception: %s", e.what());
            //     return;
            // }
        }
        
        // void depthCallback(const sensor_msgs::Image::ConstPtr& msg) {
        //     try {
        //         cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        //     } catch (cv_bridge::Exception& e) {
        //         ROS_ERROR("cv_bridge exception: %s", e.what());
        //         return;
        //     }
        //     depth_image = cv_ptr_depth->image;
        //     double min_val, max_val;
        //     cv::minMaxLoc(depth_image, &min_val, &max_val);
        //     cv::Mat normalized_depth_image = (depth_image - min_val) * 255.0 / (max_val - min_val);
        //     std::cout << "Depth done" << std::endl;
        // }


        std::vector<float> pixel_to_world(double x,  double y, const cv::Mat& depth_image) {
            std::cout << "Started Pixel" << std::endl;
            // Convert pixel coordinates using inverse perspective transform
            image_cont_1 = (cv::Mat_<float>(3,1) << x, y, 1);
            cv::warpPerspective(image_cont_1, image_cont_2, invMatrix, image_cont_1.size(), cv::INTER_LINEAR);
            // cv::Mat original_pixel_coord = transMatrix.inv() * pixel_coord;
            std::cout << "no here" << std::endl;
            std::cout << image_cont_2<< std::endl;
            // std::vector<double> original_pixel(pixel_2.at<float>(0, 0), pixel_2.at<float>(1, 0));
            depthy = image_cont_2.at<float>(0);
            // Access depth value from depth image
            std::cout << depthy << std::endl;
            // double depth_value = depth_image.at<double>(original_pixel[0], original_pixel[1]);
            std::cout << "4 here" << std::endl;
            // if (depth_value < 0.03) {
            //     return std::vector<float>{0.0f, 0.0f}; // Return an empty array
            // }
            std::cout << "4 here" << std::endl;
            // Calculate world coordinates
            // double map_y = sqrt(pow(depth_value, 2) - pow(height, 2));
            // double map_x = (original_pixel[0] - CAMERA_PARAMS.at("cx")) * depth_value / CAMERA_PARAMS.at("fx") + roll * depth_value;
            // std::cout << "5 here" << std::endl;
            // // Create vector and populate it with the world coordinates
            std::vector<float> world_coords(2);
            // world_coords[0] = static_cast<float>(map_x);
            // world_coords[1] = static_cast<float>(map_y);
            // std::cout << "Ended Pixel" << std::endl;
            return world_coords;
        }
 
        cv::Mat getIPM(cv::Mat inputImage, bool rev = false) {
            // cv::Mat undistortImage;
            cv::undistort(inputImage, image_cont_1, cameraMatrix, distCoeff);
            // cv::Mat inverseMap;
            cv::Size dest_size(inputImage.cols, inputImage.rows);
            cv::warpPerspective(image_cont_1, image_cont_2, transMatrix, dest_size, cv::INTER_LINEAR);
            return image_cont_2;
        }


        cv::Mat getLanes(cv::Mat inputImage) {
            // cv::Mat imageHist;
            cv::calcHist(&inputImage, 1, 0, cv::Mat(), image_cont_1, 1, &inputImage.rows, 0);
            cv::minMaxLoc(image_cont_1, &laneMinVal, &laneMaxVal);
            int threshold_value = std::min(std::max(static_cast<int>(maxVal) - 75, 30), 200);
            // cv::Mat binary_thresholded;
            cv::threshold(inputImage, image_cont_3, threshold_value, 255, cv::THRESH_BINARY);
            return image_cont_3;
        }

        std::vector<double> getWaypoints(std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret, std::vector<int> &y_Values) {
            int offset = 175;
            std::vector<double> waypoints(y_Values.size()); // Resized waypoints to match the size of y_Values
            std::vector<double> L_x(y_Values.size());     // Resized L_x
            std::vector<double> R_x(y_Values.size());     // Resized R_x
            int number_of_fits = std::get<0>(ret);
            left_1 = std::get<1>(ret);
            right_1 = std::get<2>(ret);
            std::cout << "NUMBER OF FITS --- : " << number_of_fits << std::endl;
            if (number_of_fits == 2) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    L_x[i] = left_1[0] + y_Values[i]*left_1[1] + left_1[2]*(y_Values[i])*(y_Values[i]);
                    R_x[i] = right_1[0] + y_Values[i]*right_1[1] + right_1[2]*(y_Values[i])*(y_Values[i]);
                    waypoints[i] = 0.5*(L_x[i] + R_x[i]);
                    waypoints[i] = static_cast<int>(std::max(0.0, std::min(static_cast<double>(waypoints[i]), 639.0)));
                }
            } else if (number_of_fits == 1) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    L_x[i] = (offset - (480 - y_Values[i])*0.05) + left_1[0] + y_Values[i]*left_1[1] + left_1[2]*(y_Values[i])*(y_Values[i]);
                    // std::cout << "Way: " << L_x[i] << std::endl;
                    waypoints[i] = std::max(0.0, std::min(L_x[i], 639.0));
                    // std::cout << "After: " << waypoints[i] << std::endl;
                }
            } else if (number_of_fits == 3) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    R_x[i] = - (offset - (480 - y_Values[i])*0.08) + right_1[0] + y_Values[i]*right_1[1] + right_1[2]*(y_Values[i])*(y_Values[i]);
                    waypoints[i] = std::max(0.0, std::min(R_x[i], 639.0));
                }
            // } else if (wayLines["stop_line"] == "0") {
            //     for (size_t i = 0; i < y_Values.size(); ++i) {
            //         waypoints[i] = 320;
            //     }
            } else {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    waypoints[i] = 320;
                }
            }
            std::cout << "Before return:" << std::endl;
            return waypoints;
        }

        std::vector<int> find_center_indices(const cv::Mat & histogram, int threshold) { // Find the center indices of potential lane lines
            
            std::vector<int> hist;
            
            for (int i = 0; i < histogram.cols; ++i) {
                hist.push_back(histogram.at<int>(0, i));
            }

            std::vector<std::vector<int>> valid_groups;
            
            std::vector<int> above_threshold; // Container for histogram values that are above the threshold

            for (int i = 0; i < hist.size(); ++i) {     // Parse through to check for values above threshold
                if (hist[i] > threshold) {
                    above_threshold.push_back(i);       // Append values that are above
                }
            }

            // std::cout << std::endl;
            std::vector<std::vector<int>> consecutive_groups;   // Container for consecutive groups 

            for (int i = 0; i < above_threshold.size();) {      // Parse through indices of values above threshold
                int j = i;
                while (j < above_threshold.size() && above_threshold[j] - above_threshold[i] == j - i) {
                    ++j;
                }
                if (j - i >= 5) {
                    consecutive_groups.push_back(std::vector<int>(above_threshold.begin() + i, above_threshold.begin() + j));   // Retain consectuive groups
                }
                i = j;
            }

            for (const std::vector<int>& group : consecutive_groups) {
                if (group.size() >= 5) {
                    valid_groups.push_back(group);
                }
            }

            std::vector<int> center_indices;
            for (const auto& group : valid_groups) {
                    int front = group.front();
                    int back = group.back();
                    int midpoint_index = group.front() + 0.5*(group.back() - group.front());
                    std::cout << "Vector front : " << front << std::endl;
                    std::cout << "Vector back : " << back << std::endl;
                    center_indices.push_back(midpoint_index);
                    std::cout << "Midpoint index: " << midpoint_index <<std::endl;
            }

            return center_indices;
        }

        int find_stop_line(const cv::Mat& image, int threshold) { // Function to identify presence of stop line

            cv::Mat roi = image(cv::Range::all(), cv::Range(0, 639));
            cv::reduce(roi/ 2, horistogram, 1, cv::REDUCE_SUM, CV_32S);
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(horistogram, &min_val, &max_val, &min_loc, &max_loc);
            std::cout << "Max loc done :  "<< max_loc << std::endl;

            std::vector<double> hist;
            std::cout << horistogram.rows << std::endl;
            horistogram = image.row(max_loc.y);
            for (int i = 0; i < horistogram.cols; ++i) {
                hist.push_back(static_cast<double>(horistogram.at<uchar>(i)));
            }
            std::cout << "Hist size :  "<< hist.size() << std::endl;
            int sum = 0;
            for(int i =0; i < hist.size(); i++){
                // std::cout << "Value is : " << hist[i] << std::endl;
            }
            
            for(int i =0; i < hist.size(); i++){        // ADD up nonzero values to see if there are consecutive ones
                sum+= 1;                                // Consecutive means stop line
                if(hist[i] < 1){
                    sum = 0;
                }
                else if( sum >= 370){
                    stop_line = true;
                    break;
                }
            }

            if(stop_line == true){
                stop_loc = max_loc.y;
            }
            else{
                stop_loc =-1;
            }

            std::cout << "Stop line check done "<< std::endl;

            return stop_loc;
        }

        bool check_cross_walk(const cv::Mat& image, int stop_index) {
            // Compute the density of non-white pixels
            cv::Mat roi = image(cv::Range(0, stop_index), cv::Range::all());
            double density = static_cast<double>(cv::countNonZero(roi)) / (roi.rows * roi.cols);

            // Check if the density exceeds the threshold (0.3)
            if (density > 0.3) {
                return true;
            } else {
                return false;
            }
        }

        std::vector<double> convertToArray(const alglib::real_1d_array& arr) {  // Convert between alglib 1d array and std::vector
            std::vector<double> vec;        // Declare vector
            int size = arr.length();        
            vec.reserve(size);  
            for (int i = 0; i < size; ++i) {    // Iterate over to to transform
                vec.push_back(arr[i]);
            }
            return vec;
        }

        std::vector<int> find_closest_pair(const std::vector<int>& indices, int lane_width) {

            int n = indices.size();     // size of input array

            if (n < 2) {        // check to see if at least two lane lines
                throw std::invalid_argument("Array must have at least two elements");
            }

            int min_diff = std::numeric_limits<int>::max();
            std::vector<int> result_pair(2);
            result_pair[0] = indices[0];
            result_pair[1] = indices[1];

            for (int i = 0; i < n - 1; ++i) {       // iterate over different pairs
                for (int j = i + 1; j < n; ++j) {
                    int current_diff = std::abs(std::abs(indices[i] - indices[j]) - lane_width);        // check for how close to optimal distance the current distance is
                    if (current_diff < min_diff) {
                        min_diff = current_diff;        // compare current pair difference with optimal difference
                        result_pair[0] = indices[i];
                        result_pair[1] = indices[j];
                        std::cout << "Min DIff: " << min_diff << std::endl;
                    }
                }
            }

            return result_pair;
        }

        std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> line_fit_2(cv::Mat binary_warped){
            // Declare variables to be used
            
            // Tuple variables  --- Initialize and declare
            left_1 = {0.0};
            right_1 = {0.0};
            cv::reduce(binary_warped(cv::Range(200, 480), cv::Range::all()) / 2, image_cont_1, 0, cv::REDUCE_SUM, CV_32S);
            // displayHistogram(histogram);
            std::cout << "Reduce done "<< std::endl;
            std::cout << "Histogram Data: " << image_cont_1.size() << std::endl;
            std::cout << "Stop line done "<< std::endl;
            std::vector<int> indices = find_center_indices(image_cont_1, threshold);               // Get the center indices
            std::cout << "Center indices are:"<< std::endl;
            for (int value : indices) {
                std::cout << value << " ";
            }
            std::cout << ""<< std::endl;
            std::cout << "Center indices done"<< std::endl;
            
            int size_indices = indices.size();      // Number of lanes detected

            if(size_indices == 0){                  // Check to see if lanes detected, if not return
                number_of_fits = 0;
                ret = std::make_tuple(number_of_fits,left_1,right_1,stop_line,stop_index,cross_walk);
                return ret;
            }

            if(size_indices == 1){                  // If only one lane line is detected
                if(indices[0] < 320){               // Check on which side of the car it is
                    number_of_fits = 1;      // NOTE : 1-LEFT FIT, 2- BOTH FITS, 3 - RIGHT FIT
                    leftx_base = indices[0];
                    rightx_base = 0;
                }
                else {                  
                    number_of_fits = 3; // NOTE : 1-LEFT FIT, 2- BOTH FITS, 3 - RIGHT FIT
                    leftx_base = 0;
                    rightx_base = indices[0];
                }
            }

            else {                      
                if(size_indices > 2){   // If more than one lane line, check for closest pair of lane lines 
                    std::vector<int> closest_pair = find_closest_pair(indices,lane_width);
                    indices[0] = closest_pair[0];       // Initialize the start of the lane line at bottom of the screen
                    indices[1] = closest_pair[1];
                }
                int delta = std::abs(indices[0]-indices[1]);        // Check to see if the two lane lines are close enough to be the same
                if(delta < 160){
                    indices[0] = 0.5*(indices[0]+indices[1]);
                    std::cout << "Delta : " << delta << std::endl;
                    std::cout << "Base[0] : " << indices[0] << std::endl;
                    if(indices[0] < 320){
                        number_of_fits = 1;      // NOTE : 1-LEFT FIT, 2- BOTH FITS, 3 - RIGHT FIT
                        leftx_base = indices[0];
                        rightx_base = 0;
                    }
                    else{
                        number_of_fits = 3; // NOTE : 1-LEFT FIT, 2- BOTH FITS, 3 - RIGHT FIT
                        leftx_base = 0;
                        rightx_base = indices[0];  
                    }
                }
                else{
                leftx_base = indices[0];       // Initialize the start of the lane line at bottom of the screen
                rightx_base = indices[1];
                number_of_fits = 2;                 // Set number of fits as a reference
                }

            }

            std::cout << "Left Base : " << leftx_base << std::endl;
            std::cout << "Right Base : " << rightx_base << std::endl;
            std::cout << "NUMBER OF FITS : " << number_of_fits << std::endl;
            std::cout << "Chosen the right number of fits done"<< std::endl;

            int window_height = static_cast<int>(binary_warped.rows / n_windows);        // Caclulate height of parsing windows
            // std::cout << "Window hieght: " << window_height << std::endl;
            // Find nonzero pixel locations
            std::vector<cv::Point> nonzero;

            cv::findNonZero(binary_warped, nonzero);    // Find nonzero values in OpenCV point format

            std::vector<int> nonzeroy, nonzerox;
            int bigsize;
            for (size_t i = 0; i < nonzero.size(); i += 2) { // Increment index by 2 -- THIS IS THE BOTTLENECK (REDUCE TO INCREASE SPEED)
                nonzeroy.push_back(nonzero[i].y);
                nonzerox.push_back(nonzero[i].x);
                bigsize += 1;
            }
            std::cout << "NonzeroX size: " << bigsize << std::endl;


            // Current positions to be updated for each window
            int leftx_current = leftx_base; // Assuming leftx_base is already defined
            int rightx_current = rightx_base; // Assuming rightx_base is already defined

            // std::cout << "New right base location : " << rightx_current << std::endl;
            // std::cout << "New left base location : " << leftx_current << std::endl; 

            // Set the width of the windows +/- margin
            int margin = 50;

            // Set minimum number of pixels found to recenter window
            int minpix = 50;

            // Create empty vectors to receive left and right lane pixel indices
            std::vector<int> left_lane_inds;
            std::vector<int> right_lane_inds;

            // std::cout << "Loop over windows begin"<< std::endl;
            for (int window = 0; window < n_windows; ++window) {
                // Identify window boundaries in y
                int win_y_low = binary_warped.rows - (window + 1) * window_height;
                int win_y_high = binary_warped.rows - window * window_height;
                // std::cout << "High boundary win : " << win_y_high << std::endl;
                // std::cout << "Low boundary win : " << win_y_low << std::endl; 

                // LEFT LANE
                if (number_of_fits == 1 || number_of_fits == 2) {
                    int win_xleft_low = leftx_current - margin;     // Bounding boxes around the lane lines
                    int win_xleft_high = leftx_current + margin;
                    int sum_left = 0;
                    std::vector<int> good_left_inds;
                    // std::cout << "Adding good  LEFT LANE pixels"<< std::endl;
                    for (size_t i = 0; i < nonzerox.size(); ++i) {  // Parse through and only select pixels within the bounding boxes
                        if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high &&
                            nonzerox[i] >= win_xleft_low && nonzerox[i] < win_xleft_high) {
                            good_left_inds.push_back(i);            // Keep pixels within the boxes
                            sum_left += nonzerox[i];
                            // std::cout << " x : " << nonzerox[i] << " y : " << nonzeroy[i] << std::endl;
                        }
                    }

                    // std::cout << "Size of good pixels LEFT : " << good_left_inds.size() << std::endl; 
                    left_lane_inds.insert(left_lane_inds.end(), good_left_inds.begin(), good_left_inds.end());      // Append all good indices together

                    if (good_left_inds.size() > minpix) {       // Recenter mean for the next bounding box
                        int mean_left = sum_left/good_left_inds.size();
                        leftx_current =  mean_left;
                        // std::cout << "New left base location : " << leftx_current << std::endl;  
                    }
                }

                // RIGHT LANE
                if (number_of_fits == 3 || number_of_fits == 2) {
                    int win_xright_low = rightx_current - margin;   // Bounding boxes around the lane lines
                    int win_xright_high = rightx_current + margin;
                    int sum_right = 0;
                    std::vector<int> good_right_inds;
                    // std::cout << "Adding good  RIGHT LANE pixels"<< std::endl;
                    for (size_t i = 0; i < nonzerox.size(); ++i) {  // Parse through and only select pixels within the bounding boxes
                        if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high &&
                            nonzerox[i] >= win_xright_low && nonzerox[i] < win_xright_high) {
                            good_right_inds.push_back(i);           // Keep pixels within the boxes
                            sum_right += nonzerox[i];
                            // std::cout << " x : " << nonzerox[i] << " y : " << nonzeroy[i] << std::endl;
                        }
                    }

                    right_lane_inds.insert(right_lane_inds.end(), good_right_inds.begin(), good_right_inds.end());  // Append all good indices together

                    if (good_right_inds.size() > minpix) {          // Keep pixels within the boxes
                        int mean_right = sum_right / good_right_inds.size();
                        rightx_current = mean_right;
                        // std::cout << "New right base location : " << rightx_current << std::endl;
                    }
                }

            }

            // std::cout << "Loop over windows done"<< std::endl;
            // Declare vectors to contain the pixel coordinates to fit
            std::vector<double> leftx;
            std::vector<double> lefty;
            std::vector<double> rightx;
            std::vector<double> righty;
            // Define the degree of the polynomial
            int m = 3;


            if (number_of_fits == 1 || number_of_fits == 2) {
                // Concatenate left_lane_inds if needed
                // left_lane_inds = concatenate(left_lane_inds);

                // Populate leftx and lefty vectors
                for (int idx : left_lane_inds) {
                    leftx.push_back(nonzerox[idx]);
                    lefty.push_back(nonzeroy[idx]);            
                }
                   
                // Perform polynomial fitting
                alglib::real_1d_array x_left, y_left;           // Declare alglib array type
                x_left.setcontent(leftx.size(), leftx.data());  // Populate X array
                y_left.setcontent(lefty.size(), lefty.data());  // Populate Y array
                alglib::polynomialfitreport rep_left;
                alglib::barycentricinterpolant p_left;
                std::cout << "Before fit"<< std::endl;
                alglib::polynomialfit(y_left, x_left, m, p_left, rep_left);     // Perform polynomial fit
                std::cout << "After fit"<< std::endl;
                // Convert polynomial coefficients to standard form
                alglib::real_1d_array a1;
                alglib::polynomialbar2pow(p_left, a1);
                left_1 = convertToArray(a1);      // Convert back to std::vector 
                std::cout << "Alglib done"<< std::endl;
            }

            // std::cout << "left polynomial fit done "<< std::endl;

            if (number_of_fits == 3 || number_of_fits == 2) {
                // Concatenate right_lane_inds if needed
                // left_lane_inds = concatenate(left_lane_inds);

                // Populate rightx and righty vectors
                for (int idx : right_lane_inds) {
                    rightx.push_back(nonzerox[idx]);
                    righty.push_back(nonzeroy[idx]);
                }
 

                // Perform polynomial fitting
                alglib::real_1d_array x_right, y_right;             // Declare alglib array type
                x_right.setcontent(rightx.size(), rightx.data());   // Populate X array
                y_right.setcontent(righty.size(), righty.data());   // Populate Y array
                alglib::polynomialfitreport rep_right; 
                alglib::barycentricinterpolant p_right;
                std::cout << "Before fit"<< std::endl;
                alglib::polynomialfit(y_right, x_right, m, p_right, rep_right);     // Perform polynomial fit
                std::cout << "After fit"<< std::endl;
                // Convert polynomial coefficients to standard form
                alglib::real_1d_array a3;
                alglib::polynomialbar2pow(p_right, a3);
                right_1 = convertToArray(a3);     // Convert back to std::Vector 
                std::cout << "Alglib done"<< std::endl;

            }

            // std::cout << "right polynomial fit done"<< std::endl;
            // Make and return tuple of required values
            ret = std::make_tuple(number_of_fits,left_1,right_1,stop_line,stop_index,cross_walk);

            return ret;
        }

};

int main(int argc, char *argv[])
{
    /* code */
    std::cout << "Starting the attemptLane_2 NODE..."<< std::endl;
    ros::init(argc, argv, "malosnode");
    std::cout << "node created";
    ros::NodeHandle nh;
    LaneDetectNode_2 detector(nh);
    return 0;
}