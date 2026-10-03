#pragma once

#include <opencv2/opencv.hpp>

namespace mono_vio {

struct CameraIntrinsics {
    double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
    double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0};
    int    width{0}, height{0};

    cv::Mat toCvMat() const {
        return (cv::Mat_<double>(3,3)
            << fx, 0, cx,
               0, fy, cy,
               0,  0,  1);
    }

    cv::Mat distCoeffs() const {
        return (cv::Mat_<double>(1,4) << k1, k2, p1, p2);
    }
};

} // namespace mono_vio
