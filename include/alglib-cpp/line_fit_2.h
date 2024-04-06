#ifndef LINE_FIT_2_H  // Header guard to prevent double inclusion
#define LINE_FIT_2_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

// Function to find center indices of potential lane lines
std::vector<int> find_center_indices(const std::vector<int> & hist, int threshold);

// Function to identify presence of stop line
std::tuple<bool, int, int> find_stop_line(const cv::Mat& image, int threshold);

// Function to check for crosswalk
bool check_cross_walk(const cv::Mat& image, int stop_index);

// Function to concatenate arrays
std::vector<int> concatenate(const std::vector<std::vector<int>>& arrays);

// Function to find the closest pair of indices
std::vector<int> find_closest_pair(const std::vector<int>& indices, int lane_width);

// Function to fit lines
std::tuple<int, std::vector<double>, std::vector<double>, bool, int, bool> line_fit_2(cv::Mat binary_warped);

#endif  // LINE_FIT_2_H