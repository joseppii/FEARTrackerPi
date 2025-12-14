#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct InferenceResult {
    std::vector<float> classification_map;  // [1, 1, H, W]
    std::vector<float> regression_map;      // [1, 4, H, W]
    cv::Size output_size;                   // H, W of the output maps

    InferenceResult() : output_size(0, 0) {}
};
