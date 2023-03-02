#pragma once
#ifndef _CUDAARITHM
#define _CUDAARITHM
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp> 
//#include <opencv2/cudaimgproc.hpp> 
//#include <opencv2/cudaarithm.hpp>
#endif
#include <utility>
#include "detector_utils.h"

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>


#include<opencv2/core/core.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;


class YOLOv5TRTDetector {
public:
    explicit YOLOv5TRTDetector(std::nullptr_t) {};
    YOLOv5TRTDetector(const std::string& modelPath);
    // ~YOLOv5TRTDetector();
    void doInference(float* input, float* output, int batchSize);
    std::vector<Detection> detect(cv::Mat& image);
    IExecutionContext* context = nullptr;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;

private:
    // void preprocessingTexture(ID3D11Texture2D* texture);
    void preprocessingMat(cv::Mat& image);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, const float& confThreshold, const float& iouThreshold);
    cv::Size2f inputImageShape;
    float confThreshold = 0.3f;
    float iouThreshold = 0.4f;
    float data[3 * DETECTION_SIZE * DETECTION_SIZE];
    float prob[OUTPUT_SIZE];
};
