#pragma once
#ifndef _CUDAARITHM
#define _CUDAARITHM
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudaarithm.hpp>
#endif
#include <onnxruntime_cxx_api.h>
#include <utility>
#include "detector_utils.h"


class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath,
        const bool& isGPU,
        const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat& image);
    std::vector<Detection> detect(cv::cuda::GpuMat& image);

private:
    Ort::Env env{ nullptr };
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };
    float confThreshold = 0.3f;
    float iouThreshold = 0.4f;

    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    void preprocessing(cv::cuda::GpuMat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
        const cv::Size& originalImageShape,
        std::vector<Ort::Value>& outputTensors,
        const float& confThreshold, const float& iouThreshold);

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
        float& bestConf, int& bestClassId);

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

};