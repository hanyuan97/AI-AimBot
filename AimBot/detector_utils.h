#pragma once
#ifndef _DETECTOR_UTILS
#define _DETECTOR_UTILS
#include <codecvt>
#include <fstream>
// #include "detector.h"
#ifndef _CUDAARITHM
#define _CUDAARITHM
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/cudaimgproc.hpp> 
 
 #endif
#include "defines.h"

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

struct Target
{
    bool isFind;
    cv::Rect box;
    cv::Point pos;
};


namespace utils
{
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::wstring stringToWstring(const std::string str);
    std::vector<std::string> loadNames(const std::string& path);
    float distance(float x, float y);
    void findClosest(std::vector<Detection>& detections, int classId, Target &target);
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
        const std::vector<std::string>& classNames, Target& target);

    void letterbox(const cv::Mat& image, cv::Mat& outImage,
        const cv::Size& newShape,
        const cv::Scalar& color,
        bool auto_,
        bool scaleFill,
        bool scaleUp,
        int stride);

    /*void letterbox(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& outImage,
        const cv::Size& newShape,
        const cv::Scalar& color,
        bool auto_,
        bool scaleFill,
        bool scaleUp,
        int stride);*/

    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

    template <typename T>
    T clip(const T& n, const T& lower, const T& upper);
}
#endif