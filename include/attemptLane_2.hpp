#include <ros/ros.h>
#include <opencv2/opencv.hpp>
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
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

namespace plt = matplotlibcpp;


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


        void printTuple(const std::tuple<int, std::vector<double>, std::vector<double>, bool, int, bool>& ret) {
            std::cout << "Contents of the tuple ret:" << std::endl;
            std::cout << "number_of_fits: " << std::get<0>(ret) << std::endl;

            std::cout << "left_fit: ";
            for (const auto& val : std::get<1>(ret)) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            std::cout << "right_fit: ";
            for (const auto& val : std::get<2>(ret)) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            std::cout << "stop_line: " << std::boolalpha << std::get<3>(ret) << std::endl;
            std::cout << "stop_index: " << std::get<4>(ret) << std::endl;
            std::cout << "cross_walk: " << std::boolalpha << std::get<5>(ret) << std::endl;
        }


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

        std::vector<double> getWaypoints(std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret, std::vector<int> &y_Values) {
            int offset = 175;
            std::vector<double> wayPoint(y_Values.size()); // Resized wayPoint to match the size of y_Values
             std::vector<double> L_x(y_Values.size());     // Resized L_x
            std::vector<double> R_x(y_Values.size());     // Resized R_x
            int number_of_fits = std::get<0>(ret);
            std::vector<double> fit_L = std::get<1>(ret);
            std::vector<double> fit_R = std::get<2>(ret);
            std::cout << "Variables init:" << std::endl;
            if (number_of_fits == 2) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    L_x[i] = fit_L[0] + y_Values[i]*fit_L[1] + fit_L[2]*(y_Values[i])*(y_Values[i]);
                    R_x[i] = fit_R[0] + y_Values[i]*fit_R[1] + fit_R[2]*(y_Values[i])*(y_Values[i]);
                    wayPoint[i] = 0.5*(L_x[i] + R_x[i]);
                    wayPoint[i] = static_cast<int>(std::max(0.0, std::min(static_cast<double>(wayPoint[i]), 639.0)));
                }
            } else if (number_of_fits == 1) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    L_x[i] = fit_L[0] + y_Values[i]*fit_L[1] + fit_L[2]*(y_Values[i])*(y_Values[i]);
                    wayPoint[i] = static_cast<int>(std::max(0.0, std::min(static_cast<double>(wayPoint[i]), 639.0)));
                }
            } else if (number_of_fits == 3) {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    R_x[i] = fit_R[0] + y_Values[i]*fit_R[1] + fit_R[2]*(y_Values[i])*(y_Values[i]);
                    wayPoint[i] = static_cast<int>(std::max(0.0, std::min(static_cast<double>(wayPoint[i]), 639.0)));
                }
            // } else if (wayLines["stop_line"] == "0") {
            //     for (size_t i = 0; i < y_Values.size(); ++i) {
            //         wayPoint[i] = 320;
            //     }
            } else {
                for (size_t i = 0; i < y_Values.size(); ++i) {
                    wayPoint[i] = 320;
                }
            }
            std::cout << "Before return:" << std::endl;
            return wayPoint;
        }
};

//------------------------------------------------ LINE FIT PART BEGIN ------------------------------------------------------------------//
//------------------------------------------------ LINE FIT PART BEGIN ------------------------------------------------------------------//
//------------------------------------------------ LINE FIT PART BEGIN ------------------------------------------------------------------//

void plot_polynomial(const std::vector<double> &a3, const std::vector<double> &a2, const std::vector<double> &a4, const std::vector<int> &yY) {
    // Create a range of x-values
    std::vector<int> x_values;
    std::vector<double> x_2;
    for (int x = 0; x <= 640; x += 1) {
        x_values.push_back(x);
    }

    // Evaluate the polynomial for each x-value
    std::vector<double> yR_values(x_values.size());
    std::vector<double> yL_values(x_values.size());
    for (size_t i = 0; i < x_values.size(); ++i) {
        yL_values[i] = a3[0] + x_values[i]*a3[1] + a3[2]*(x_values[i])*(x_values[i]);
        yR_values[i] = a2[0] + x_values[i]*a2[1] + a2[2]*(x_values[i])*(x_values[i]);
    }

    // Plot the polynomial curve
        plt::plot(yR_values, x_values,".");
        plt::plot(yL_values, x_values,".");
        plt::plot(a4,yY,".");
        plt::xlabel("x");
        plt::ylabel("y");
        plt::ylim(479,0);
        plt::xlim(0,639);
        plt::title("Plot of Polynomial");
        plt::grid(true);
        plt::show();
}


void displayHistogram(const cv::Mat& histogram) {
    // Convert histogram data to vector for plotting
    std::vector<int> histData;
    
// Copy the histogram data to a vector
    for (int i = 0; i < histogram.cols; ++i) {
        histData.push_back(histogram.at<int>(0, i));
    }
    // Print histogram data
    // std::cout << "Histogram Data: ";
    // for (int value : histData) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    // Plot the histogram
    plt::plot(histData);
    plt::show();

    // Close the plot window after 5 seconds
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // plt::close(); // Close the plot window
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

    std::cout << std::endl;
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

    // for (const std::vector<int>& group : consecutive_groups) {
    //     std::cout << "Consecutive Group:";
    //     for (int value : group) {
    //         std::cout << " " << value;
    //     }
    //     std::cout << std::endl;
    // }

    // Iterate over consecutive_groups and find ones that are five continuous pixels or more
    for (const std::vector<int>& group : consecutive_groups) {
        if (group.size() >= 5) {
            valid_groups.push_back(group);
        }
    }


    // for (const std::vector<int>& group : valid_groups) {
    //     std::cout << "Valid Group:";
    //     for (int value : group) {
    //         std::cout << " " << value;
    //     }
    //     std::cout << std::endl;
    // }

    // Find the center index for each valid group
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

std::tuple<bool, int, int> find_stop_line(const cv::Mat& image, int threshold) { // Function to identify presence of stop line

    // NOTE : CAN BE OPTIMIZED BY COMBINING VERTICAL HISTOGRAM WITH find_center_indices, reduce one histogram computation
    // Maybe modify the return values to conform with what is needed

    // Find indices where histogram values are above the threshold
    std::cout << "Starting reduce "<< std::endl;
    cv::Mat histogram;
    cv::reduce(image(cv::Range(0, 480), cv::Range::all()) / 2, histogram, 0, cv::REDUCE_SUM, CV_32S);
    std::cout << "Histogram done "<< std::endl;
    
    std::vector<int> above_threshold;       // Container for values above threshold

    for (int i = 0; i < histogram.cols; ++i) {      // Iterate over histogram values
        if (histogram.at<int>(0, i) > threshold) {
            above_threshold.push_back(i);           // Retain values above threshold
        }
    }

    // Find consecutive groups of five or more indices
    std::vector<std::vector<int>> consecutive_groups;       // Container for continuous pixel groups

    for (int i = 0; i < above_threshold.size();) {          // Iterate over pixels above the threshold
        int j = i;
        while (j < above_threshold.size() && above_threshold[j] - above_threshold[i] == j - i) {
            ++j;
        }
        if (j - i >= 5) {
            consecutive_groups.push_back(std::vector<int>(above_threshold.begin() + i, above_threshold.begin() + j));       // Retain groups of continuous pixels
        }
        i = j;
    }

    std::cout << "Consecutive done "<< std::endl;

    // Find the maximum index of the horizontal histogram
    // This is because the stop line will be a section of a lot of white pixels, i.e. the max of the histogram horizontally

    cv::Mat horistogram;
    cv::reduce(image(cv::Range::all(), cv::Range(0, 640)) / 2, horistogram, 1, cv::REDUCE_SUM, CV_32S);
    cv::Point max_loc;
    cv::minMaxLoc(horistogram, nullptr, nullptr, nullptr, &max_loc);

    // Check to see if there is a sequence of pixels long enough for a stop line
    bool stop_line = false;
    int width = 0;
    for (const auto& group : consecutive_groups) {  
        if (group.size() >= 370) {                  // NOTE: HARD CODED VALUE for the stop line width
            stop_line = true;
            cv::Mat above_threshold2;
            cv::threshold(horistogram, above_threshold2, 50000, 255, cv::THRESH_BINARY);
            std::vector<cv::Point> non_zero_indices;
            cv::findNonZero(above_threshold2, non_zero_indices);
            if (!non_zero_indices.empty()) {
                width = abs(non_zero_indices.back().y - non_zero_indices.front().y);
            }
            break;
        }
    }

    std::cout << "Stop line check done "<< std::endl;

    return std::make_tuple(stop_line, max_loc.y, width);
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

std::vector<int> concatenate(const std::vector<std::vector<int>>& arrays) {
    std::vector<int> result;
    for (const auto& array : arrays) {
        result.insert(result.end(), array.begin(), array.end());
    }
    return result;
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
            }
        }
    }

    return result_pair;
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

std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> line_fit_2(cv::Mat binary_warped){
    // Declare variables to be used
    int lane_width = 350;        // HARD CODED LANE WIDTH
    int n_windows = 9;           // HARD CODED WINDOW NUMBER FOR LANE PARSING
    cv::Mat histogram;
    int threshold = 2000;       // HARD CODED THRESHOLD
    int leftx_base = 0;
    int rightx_base = 640;
    std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret;
    // Tuple variables  --- Initialize and declare
    int number_of_fits = 0;
    std::vector<double> left_fit = {0.0};
    std::vector<double> right_fit = {0.0};
    bool stop_line = false;
    int stop_index = 0;
    bool cross_walk = false;
    cv::reduce(binary_warped(cv::Range(200, 480), cv::Range::all()) / 2, histogram, 0, cv::REDUCE_SUM, CV_32S);
    // displayHistogram(histogram);
    std::cout << "Reduce done "<< std::endl;
    std::cout << "Histogram Data: " << histogram.size() << std::endl;
    std::tuple<bool, int, int> stop_data = find_stop_line(binary_warped, threshold);    // Get the stop line data
    std::cout << "Stop line done "<< std::endl;
    std::vector<int> indices = find_center_indices(histogram, threshold);               // Get the center indices
    // std::cout << "Center indices are:"<< std::endl;
    // for (int value : indices) {
    //     std::cout << value << " ";
    // }
    // std::cout << ""<< std::endl;
    std::cout << "Center indices done"<< std::endl;
    stop_line = std::get<0>(stop_data);

    if(stop_line){      // Check crosswalk only if there is a stop line 
        stop_index = std::get<1>(stop_data);
        cross_walk = check_cross_walk(binary_warped,stop_index);
    }
    std::cout << "Cross walk done"<< std::endl;
    int size_indices = indices.size();      // Number of lanes detected

    if(size_indices == 0){                  // Check to see if lanes detected, if not return
        number_of_fits = 0;
        ret = std::make_tuple(number_of_fits,left_fit,right_fit,stop_line,stop_index,cross_walk);
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
            leftx_base = closest_pair[0];       // Initialize the start of the lane line at bottom of the screen
            rightx_base = closest_pair[1];
        }
        else{
            leftx_base = indices[0];       // Initialize the start of the lane line at bottom of the screen
            rightx_base = indices[1];
        }
        number_of_fits = 2;                 // Set number of fits as a reference
    }

    std::cout << "Left Base : " << leftx_base << std::endl;
    std::cout << "Right Base : " << rightx_base << std::endl;
    std::cout << "Chosen the right number of fits done"<< std::endl;

    int window_height = static_cast<int>(binary_warped.rows / n_windows);        // Caclulate height of parsing windows
    std::cout << "Window hieght: " << window_height << std::endl;
    // Find nonzero pixel locations
    std::vector<cv::Point> nonzero;

    cv::findNonZero(binary_warped, nonzero);    // Find nonzero values in OpenCV point format
    // std::cout << "Nonzero pixels are:"<< std::endl;
    // for (cv::Point value_i : nonzero) {
    //     std::cout << value_i << " ";
    // }
    std::cout << ""<< std::endl;
    std::cout << "Found nonzero pixels done"<< std::endl;

    // Separate x and y coordinates of nonzero pixels
    std::vector<int> nonzeroy, nonzerox;
    for (const auto& point : nonzero) {
        nonzeroy.push_back(point.y);
        nonzerox.push_back(point.x);
    }

    // Current positions to be updated for each window
    int leftx_current = leftx_base; // Assuming leftx_base is already defined
    int rightx_current = rightx_base; // Assuming rightx_base is already defined

    std::cout << "New right base location : " << rightx_current << std::endl;
    std::cout << "New left base location : " << leftx_current << std::endl; 

    // Set the width of the windows +/- margin
    int margin = 50;

    // Set minimum number of pixels found to recenter window
    int minpix = 50;

    // Create empty vectors to receive left and right lane pixel indices
    std::vector<int> left_lane_inds;
    std::vector<int> right_lane_inds;

    std::cout << "Loop over windows begin"<< std::endl;

    for (int window = 0; window < n_windows; ++window) {
        // Identify window boundaries in y
        int win_y_low = binary_warped.rows - (window + 1) * window_height;
        int win_y_high = binary_warped.rows - window * window_height;
        std::cout << "High boundary win : " << win_y_high << std::endl;
        std::cout << "Low boundary win : " << win_y_low << std::endl; 

        // LEFT LANE
        if (number_of_fits == 1 || number_of_fits == 2) {
            int win_xleft_low = leftx_current - margin;     // Bounding boxes around the lane lines
            int win_xleft_high = leftx_current + margin;
            int sum_left = 0;
            std::vector<int> good_left_inds;
            std::cout << "Adding good  LEFT LANE pixels"<< std::endl;
            for (size_t i = 0; i < nonzerox.size(); ++i) {  // Parse through and only select pixels within the bounding boxes
                if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high &&
                    nonzerox[i] >= win_xleft_low && nonzerox[i] < win_xleft_high) {
                    good_left_inds.push_back(i);            // Keep pixels within the boxes
                    sum_left += nonzerox[i];
                    // std::cout << " x : " << nonzerox[i] << " y : " << nonzeroy[i] << std::endl;
                }
            }

            std::cout << "Size of good pixels LEFT : " << good_left_inds.size() << std::endl; 
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
            std::cout << "Adding good  RIGHT LANE pixels"<< std::endl;
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

    std::cout << "Loop over windows done"<< std::endl;

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
        // plt::plot(leftx, lefty,".");
        // plt::xlabel("x");
        // plt::ylabel("y");
        // plt::ylim(479,0);
        // plt::xlim(0,639);
        // plt::title("Plot of Polynomial");
        // plt::grid(true);
        

        // Perform polynomial fitting
        alglib::real_1d_array x_left, y_left;           // Declare alglib array type
        x_left.setcontent(leftx.size(), leftx.data());  // Populate X array
        y_left.setcontent(lefty.size(), lefty.data());  // Populate Y array
        alglib::polynomialfitreport rep_left;
        alglib::barycentricinterpolant p_left;  
        alglib::polynomialfit(y_left, x_left, m, p_left, rep_left);     // Perform polynomial fit

        // Convert polynomial coefficients to standard form
        alglib::real_1d_array a1;
        alglib::polynomialbar2pow(p_left, a1);
        left_fit = convertToArray(a1);      // Convert back to std::vector 

    }

    std::cout << "left polynomial fit done "<< std::endl;

    if (number_of_fits == 3 || number_of_fits == 2) {
        // Concatenate right_lane_inds if needed
        // left_lane_inds = concatenate(left_lane_inds);

        // Populate rightx and righty vectors
        for (int idx : right_lane_inds) {
            rightx.push_back(nonzerox[idx]);
            righty.push_back(nonzeroy[idx]);
        }

        // plt::plot(rightx, righty);

        // Perform polynomial fitting
        alglib::real_1d_array x_right, y_right;             // Declare alglib array type
        x_right.setcontent(rightx.size(), rightx.data());   // Populate X array
        y_right.setcontent(righty.size(), righty.data());   // Populate Y array
        alglib::polynomialfitreport rep_right; 
        alglib::barycentricinterpolant p_right;
        alglib::polynomialfit(y_right, x_right, m, p_right, rep_right);     // Perform polynomial fit

        // Convert polynomial coefficients to standard form
        alglib::real_1d_array a3;
        alglib::polynomialbar2pow(p_right, a3);
        right_fit = convertToArray(a3);     // Convert back to std::Vector 
    }

    std::cout << "right polynomial fit done"<< std::endl;
    // Make and return tuple of required values
    ret = std::make_tuple(number_of_fits,left_fit,right_fit,stop_line,stop_index,cross_walk);
    return ret;
}

//------------------------------------------------ LINE FIT PART END------------------------------------------------------------------//
//------------------------------------------------ LINE FIT PART END------------------------------------------------------------------//
//------------------------------------------------ LINE FIT PART END------------------------------------------------------------------//


int main(int argc, char *argv[])
{
    /* code */
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