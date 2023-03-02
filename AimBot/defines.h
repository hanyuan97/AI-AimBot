#pragma once

constexpr auto DETECTION_RANGE = 560;  // 560 640
constexpr auto CENTER = (DETECTION_RANGE / 2);
constexpr auto DETECTION_SIZE = 640;
constexpr auto AIM_RANGE = 240;
constexpr auto TARGET_CLASSID = 0;

constexpr auto cls_num = 1;

// yolov5
constexpr auto anchor_output_num = 25200;  //anchor:416-->10647 | 640-->25200 | 960-->56700
constexpr auto OUTPUT_SIZE = 1 * anchor_output_num * (cls_num + 5); //1000 * sizeof(Detection) / sizeof(float) + 1;