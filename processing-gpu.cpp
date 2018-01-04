#include <add-gpu.h>
#include "include.hpp"
#include <iostream>
using namespace std;

void local_maxima_gpu(const cv::Mat& src, cv::Mat& dst) {
    const float* srcData = (const float*) src.data;
    uint8_t* dstData = dst.data;
    int w = src.size().width;
    int h = src.size().height;

    gpuLocalMaxima(srcData, dstData, w, h);
}

void dt_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm, int mask) {
    if(mask == DIST_MASK_3) {
        cout << "Using 3x3 mask" << endl;
        dt_3x3_lm_gpu(diff, dt, lm);
    } else if(mask == DIST_MASK_5) {
        cout << "Using 5x5 mask" << endl;
        dt_5x5_lm_gpu(diff, dt, lm);
    } else if(mask == DIST_MASK_PRECISE) {
        cout << "Using precise mask" << endl;
        dt_fast_gpu(diff, dt);
        local_maxima_gpu(dt, lm);
    } else {
        cerr << "Error: Unknown mask!";
    }
}

void dt_3x3_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    uint8_t* lmData = lm.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTLM(diffData, dtData, lmData, w, h);
}

void dt_5x5_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    uint8_t* lmData = lm.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTLM_5x5(diffData, dtData, lmData, w, h);
}

void dt_fast_gpu(const cv::Mat& diff, cv::Mat& dt) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTfast(diffData, dtData, w, h);
}


void distance_transform(const cv::Mat& srcMat, cv::Mat& dstMat, int mask) {
    cv::distanceTransform(srcMat, dstMat, DIST_TYPE, mask);
}

void local_maxima(const cv::Mat& srcMat, cv::Mat& dstMat) {
    // non maxima suppression
    cv::Mat temp;
    cv::dilate(srcMat, temp, cv::Mat());
    dstMat = srcMat == temp;
    cv::bitwise_and(dstMat, srcMat >= 1.5, dstMat);
}