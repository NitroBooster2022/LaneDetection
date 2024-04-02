#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "interpolation.h"
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>


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

std::vector<int> find_center_indices(const std::vector<int> & hist, int threshold) { // Find the center indices of potential lane lines
    std::vector<std::vector<int>> valid_groups;
    
    std::vector<int> above_threshold; // Container for histogram values that are above the threshold

    for (int i = 0; i < hist.size(); ++i) {     // Parse through to check for values above threshold
        if (hist[i] > threshold) {
            above_threshold.push_back(i);       // Append values that are above
        }
    }

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

    // Iterate over consecutive_groups and find ones that are five continuous pixels or more
    for (const std::vector<int>& group : consecutive_groups) {
        if (group.size() >= 5) {
            valid_groups.push_back(group);
        }
    }

    // Find the center index for each valid group
    std::vector<int> center_indices;
    for (const auto& group : valid_groups) {
            int distance = std::distance(group.begin(), group.end());
            int midpoint_index = (distance / 2);
            center_indices.push_back(midpoint_index);
    }

    return center_indices;
}

std::tuple<bool, int, int> find_stop_line(const cv::Mat& image, int threshold) { // Function to identify presence of stop line

    // NOTE : CAN BE OPTIMIZED BY COMBINING VERTICAL HISTOGRAM WITH find_center_indices, reduce one histogram computation
    // Maybe modify the return values to conform with what is needed

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // Convert to grayscale

    // Find indices where histogram values are above the threshold
    cv::Mat histogram;
    cv::reduce(grayImage(cv::Range(0, 480), cv::Range::all()) / 2, histogram, 0, cv::REDUCE_SUM, CV_32S);

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

    // Find the maximum index of the horizontal histogram
    // This is because the stop line will be a section of a lot of white pixels, i.e. the max of the histogram horizontally

    cv::Mat horistogram;
    cv::reduce(grayImage(cv::Range::all(), cv::Range(0, 640)) / 2, horistogram, 1, cv::REDUCE_SUM, CV_32S);
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

std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> line_fit_2(cv::Mat binary_warped){
    // Declare variables to be used
    int lane_width = 350;        // HARD CODED LANE WIDTH
    int n_windows = 9;           // HARD CODED WINDOW NUMBER FOR LANE PARSING
    cv::Mat histogram;
    int threshold = 5000;
    int leftx_base = 0;
    int rightx_base = 640;
    std::tuple<int,std::vector<double>, std::vector<double>, bool, int, bool> ret;
    // Tuple variables
    int number_of_fits = 0;
    std::vector<double> left_fit = {0.0};
    std::vector<double> right_fit = {0.0};
    bool stop_line = false;
    int stop_index = 0;
    bool cross_walk = false;

    cv::reduce(binary_warped(cv::Range(200, 480), cv::Range::all()) / 2, histogram, 0, cv::REDUCE_SUM, CV_32S);

    std::tuple<bool, int, int> stop_data = find_stop_line(binary_warped, threshold);    // Get the stop line data

    std::vector<int> indices = find_center_indices(histogram, threshold);               // Get the center indices

    stop_line = std::get<0>(stop_data);

    if(stop_line){
        stop_index = std::get<1>(stop_data);
        cross_walk = check_cross_walk(binary_warped,stop_index);
    }

    int size_indices = indices.size();

    if(size_indices == 0){
        number_of_fits = 0;
        ret = std::make_tuple(number_of_fits,left_fit,right_fit,stop_line,stop_index,cross_walk);
        return ret;
    }

    if(size_indices == 1){
        if(indices[0] < 320){
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
        std::vector<int> closest_pair = find_closest_pair(indices,lane_width);
        leftx_base = closest_pair[0];
        rightx_base = closest_pair[1];
        number_of_fits = 2;
    }

    int window_height = static_cast<int>(binary_warped.rows / n_windows);        // Caclulate height of parsing windows

    // Find nonzero pixel locations
    std::vector<cv::Point> nonzero;

    cv::findNonZero(binary_warped, nonzero);
    
    // Separate x and y coordinates of nonzero pixels
    std::vector<int> nonzeroy, nonzerox;
    for (const auto& point : nonzero) {
        nonzeroy.push_back(point.y);
        nonzerox.push_back(point.x);
    }

    // Current positions to be updated for each window
    int leftx_current = leftx_base; // Assuming leftx_base is already defined
    int rightx_current = rightx_base; // Assuming rightx_base is already defined

    // Set the width of the windows +/- margin
    int margin = 50;

    // Set minimum number of pixels found to recenter window
    int minpix = 50;

    // Create empty vectors to receive left and right lane pixel indices
    std::vector<int> left_lane_inds, right_lane_inds;

    for (int window = 0; window < n_windows; ++window) {
    // Identify window boundaries in y
    int win_y_low = binary_warped.rows - (window + 1) * window_height;
    int win_y_high = binary_warped.rows - window * window_height;

    // LEFT LANE
    if (number_of_fits == 1 || number_of_fits == 2) {
        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;

        std::vector<int> good_left_inds;
        for (size_t i = 0; i < nonzerox.size(); ++i) {
            if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high &&
                nonzerox[i] >= win_xleft_low && nonzerox[i] < win_xleft_high) {
                good_left_inds.push_back(i);
            }
        }

        left_lane_inds.insert(left_lane_inds.end(), good_left_inds.begin(), good_left_inds.end());

        if (good_left_inds.size() > minpix) {
            leftx_current = cvRound(cv::mean(cv::Mat(good_left_inds))[0]);
        }
    }

    // RIGHT LANE
    if (number_of_fits == 3 || number_of_fits == 2) {
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        std::vector<int> good_right_inds;
        for (size_t i = 0; i < nonzerox.size(); ++i) {
            if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high &&
                nonzerox[i] >= win_xright_low && nonzerox[i] < win_xright_high) {
                good_right_inds.push_back(i);
            }
        }

        right_lane_inds.insert(right_lane_inds.end(), good_right_inds.begin(), good_right_inds.end());

        if (good_right_inds.size() > minpix) {
            rightx_current = cvRound(cv::mean(cv::Mat(good_right_inds))[0]);
        }
    }

    }

    // Polynomial fitting time
    std::vector<int> leftx;
    std::vector<int> lefty;

    if (number_of_fits == 1 || number_of_fits == 2){
        left_lane_inds = concatenate(left_lane_inds);
        for (int idx : left_lane_inds) {
            leftx.push_back(nonzerox[idx]);
            lefty.push_back(nonzeroy[idx]);
        }
        double t = 2;
        ae_int_t m = 2
        barycentricinterpolant poly_nom;
        polynomialfitreport rep_ort;
        double v;
        polynomialfit(x, y, m, p, rep);
        polynomialbar2pow(p, left_fit);
    }

    if (number_of_fits == 3 || number_of_fits == 2){
            right_lane_inds = concatenate(right_lane_inds);
            for (int idx : right_lane_inds) {
                rightx.push_back(nonzerox[idx]);
                righty.push_back(nonzeroy[idx]);
            }
            double t = 2;
            ae_int_t m = 2
            alglib::barycentricinterpolant poly_nom;
            alglib::polynomialfitreport rep_ort;
            double v;
            alglib.polynomialfit(x, y, m, p, rep);
            polynomialbar2pow(p, right_fit); 
        }
    


}


