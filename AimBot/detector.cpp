#pragma once
#include "detector.h"

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;


YOLOv5TRTDetector::YOLOv5TRTDetector(const std::string& modelPath) {
    char* trtModelStream{ nullptr };
    size_t size{0};
    std::ifstream file(modelPath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cout << "file is not good" << std::endl;
    }
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    cv::Size& inputSize = cv::Size(DETECTION_SIZE, DETECTION_SIZE);
    this->inputImageShape = cv::Size2f(inputSize);
}

//YOLOv5TRTDetector::~YOLOv5TRTDetector() {
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//}

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";

void YOLOv5TRTDetector::doInference(float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.

    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    //const int inputIndex = 0;
    //const int outputIndex = 1;
    // Create GPU buffers on device
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * DETECTION_SIZE * DETECTION_SIZE * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // Set input dimensions
    Dims inputDims = context->getEngine().getBindingDimensions(inputIndex);
    inputDims.d[0] = batchSize;
    context->setBindingDimensions(inputIndex, inputDims);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * DETECTION_SIZE * DETECTION_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


//void YOLOv5TRTDetector::preprocessingTexture(ID3D11Texture2D* inputTexture) {
//    // Create CUDA resources
//    cudaResourceDesc resDesc = {};
//    resDesc.resType = cudaResourceTypeD3D11Texture2D;
//    resDesc.res.d3d11Texture2D = inputTex;
//    resDesc.flags = cudaResourceUsageGather;
//    cudaTextureDesc texDesc = {};
//    texDesc.readMode = cudaReadModeNormalizedFloat;
//    cudaTextureObject_t texObj = 0;
//    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
//}

void YOLOv5TRTDetector::preprocessingMat(cv::Mat& image) {
    cv::Mat resizedImage, floatImage;
    // cv::cuda::GpuMat image_gpu, resizedImage_gpu, floatImage_gpu;
    //cv::cuda::cvtColor(image_gpu, resizedImage_gpu, cv::COLOR_BGR2RGB);
    
    cv::cvtColor(image, resizedImage, cv::COLOR_BGRA2BGR);
    /*utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
        cv::Scalar(114, 114, 114), false,
        false, true, 32);*/
    cv::resize(resizedImage, resizedImage, cv::Size(DETECTION_SIZE, DETECTION_SIZE), 0, 0, cv::INTER_LINEAR);
    std::vector<cv::Mat> InputImage;
    InputImage.push_back(resizedImage);
    int ImgCount = InputImage.size();
    auto blobFromImage_start = std::chrono::high_resolution_clock::now();
    //float input_data[BatchSize * 3 * INPUT_H * INPUT_W];
    for (int b = 0; b < ImgCount; b++) {
        cv::Mat img = InputImage.at(b);
        int w = img.cols;
        int h = img.rows;
        int i = 0;
        for (int row = 0; row < h; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < DETECTION_SIZE; ++col) {
                data[b * 3 * DETECTION_SIZE * DETECTION_SIZE + i] = (float)uc_pixel[2] / 255.0;
                data[b * 3 * DETECTION_SIZE * DETECTION_SIZE + i + DETECTION_SIZE * DETECTION_SIZE] = (float)uc_pixel[1] / 255.0;
                data[b * 3 * DETECTION_SIZE * DETECTION_SIZE + i + 2 * DETECTION_SIZE * DETECTION_SIZE] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }
    auto blobFromImage_end = std::chrono::high_resolution_clock::now();
    auto blobFromImage_time = std::chrono::duration_cast<std::chrono::milliseconds>(blobFromImage_end - blobFromImage_start).count();
    std::cout << "blobFromImage_time: " << blobFromImage_time << "ms" << std::endl;
}

std::vector<Detection>YOLOv5TRTDetector::postprocessing(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, const float& confThreshold, const float& iouThreshold) {
    // std::vector<Detection> results;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    int tmp_idx;
    float tmp_cls_score;
    for (int i = 0; i < anchor_output_num; i++) {
        tmp_idx = i * (cls_num + 5);
        int centerX = (int)(prob[tmp_idx + 0]);
        int centerY = (int)(prob[tmp_idx + 1]);
        int width = (int)(prob[tmp_idx + 2]);
        int height = (int)(prob[tmp_idx + 3]);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        float clsConf = prob[tmp_idx + 4];
        float confidence = tmp_cls_score = prob[tmp_idx + 5] * clsConf;
        float objConf;
        int classId=0;
        for (int j = 1; j < cls_num; j++) {
            tmp_idx = i * (cls_num + 5) + 5 + j;
            if (tmp_cls_score < prob[tmp_idx] * clsConf)
            {
                tmp_cls_score = prob[tmp_idx] * clsConf;
                classId = j;
                confidence = tmp_cls_score;
            }
        }
        boxes.emplace_back(left, top, width, height);
        confs.emplace_back(confidence);
        classIds.emplace_back(classId);

    }


    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;
    // nms_keep_index = nms(pre_results, nms_thr);

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        // std::cout << "before: " << det.box << std::endl;
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);
        // std::cout << "after: " << det.box << std::endl;
        det.conf = confs[idx];
        det.classId = classIds[idx];
        // std::cout << "after: " << det.classId << std::endl;
        detections.emplace_back(det);
    }
    
    return detections;

}

std::vector<Detection> YOLOv5TRTDetector::detect(cv::Mat& image) {
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    this->preprocessingMat(image);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
    std::cout << "preprocess_time: " << preprocess_time << "ms" << std::endl;
    auto doInference_start = std::chrono::high_resolution_clock::now();
    this->doInference(data, prob, 1);
    auto doInference_end = std::chrono::high_resolution_clock::now();
    auto doInference_time = std::chrono::duration_cast<std::chrono::milliseconds>(doInference_end - doInference_start).count();
    std::cout << "doInference_time: " << doInference_time << "ms" << std::endl;

    cv::Size resizedShape = cv::Size((int)DETECTION_SIZE, (int)DETECTION_SIZE);
    auto postprocessing_start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> result = this->postprocessing(resizedShape, image.size(), confThreshold, iouThreshold);
    auto postprocessing_end = std::chrono::high_resolution_clock::now();
    auto postprocessing_time = std::chrono::duration_cast<std::chrono::milliseconds>(postprocessing_end - postprocessing_start).count();
    std::cout << "postprocessing_time: " << postprocessing_time << "ms" << std::endl;
    return result;
}
