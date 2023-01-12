#pragma once
#include "detector_utils.h"
#define NOMINMAX
#include <windows.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>



size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

std::wstring utils::stringToWstring(const std::string str)
{
    /*
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
    */
    // std::string strs = str;
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

std::vector<std::string> utils::loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline(infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

float utils::distance(float x, float y) {
    const float centerX = CENTER;
    const float centerY = CENTER;
    return std::sqrt(std::pow(centerX - x, 2) + std::pow(centerX - y, 2));
}


void utils::findClosest(std::vector<Detection>& detections, int classId, Target& target) {
    float max_dis = AIM_RANGE / 2;
    target.isFind = false;
    for (const Detection& detection : detections) {
        if (detection.classId != classId) continue;
        float targetX = (float)detection.box.x + (float)detection.box.width * 0.5;
        float targetY = (float)detection.box.y + (float)detection.box.height * 0.24;
        float dis = distance(targetX, targetY);
        if (dis <= max_dis) {
            target.pos.x = targetX;
            target.pos.y = targetY;
            target.box = detection.box;
            max_dis = dis;
            target.isFind = true;
        }
    }
}

cv::Rect AimRect = cv::Rect(CENTER - AIM_RANGE / 2, CENTER - AIM_RANGE / 2, AIM_RANGE, AIM_RANGE);

void utils::visualizeDetection(cv::Mat& frame, std::vector<Detection>& detections,
    const std::vector<std::string>& classNames, Target& target)
{
    cv::rectangle(frame, AimRect, cv::Scalar(255, 0, 0), 3);
    if (target.isFind) {
        cv::rectangle(frame, target.box, cv::Scalar(0, 0, 255), 4);
        cv::circle(frame, target.pos, 5, cv::Scalar(0, 0, 255), 4);
        cv::line(frame, cv::Point(CENTER, CENTER), target.pos, cv::Scalar(255, 255, 0), 3);
    }
    for (const Detection& detection : detections)
    {
        cv::rectangle(frame, detection.box, cv::Scalar(229, 160, 21), 3);
        /*
        int x = detection.box.x;
        int y = detection.box.y;
        
        int conf = (int)std::round(detection.conf * 100);
        int classId = detection.classId;
        std::string label = classNames[classId] + " 0." + std::to_string(conf);

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 4, &baseline);
        cv::rectangle(frame,
            cv::Point(x, y - 25), cv::Point(x + size.width, y),
            cv::Scalar(229, 160, 21), -1);
        
        cv::putText(frame, label,
            cv::Point(x, y - 3), cv::FONT_ITALIC,
            0.8, cv::Scalar(255, 255, 255), 3);
        */
    }
}

void utils::letterbox(const cv::Mat& image, cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ (int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::letterbox(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ (int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::cuda::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::cuda::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
        (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.height = (int)std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}

template <typename T>
T utils::clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}