#include <add-gpu.h>
#include <include.h>
#include <iostream>
using namespace std;

/// Compute local maxima on GPU.
/**
  Compute local maxima on GPU with OpenCV arrays. 

  \param[in] src   Source OpenCV matrix with float distance transform.
  \param[out] dst  Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \return void
*/
void local_maxima_gpu(const cv::Mat& src, cv::Mat& dst) {
    const float* srcData = (const float*) src.data;
    uint8_t* dstData = dst.data;
    int w = src.size().width;
    int h = src.size().height;

    gpuLocalMaxima(srcData, dstData, w, h);
}


/// Wrapper to compute distance transform and its local maxima on GPU with different masks.
/**
  Wraps function for all types of masks.

  \param[in] diff  Source OpenCV matrix with 8bit binary image.
  \param[out] dt   Output OpenCV float matrix with Euclidean distance transform values.
  \param[out] lm   Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \param[in] mask  Used mask (0 - precise, 3 - 3x3, 5 - 5x5).
  \return void

  ERRORS:
   - DT_UNKNOWN_MASK ... the mask is not known (allowed 0, 3, 5)
*/
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
        cerr << "Error: DT_UNKNOWN_MASK";
    }
}


/// Compute distance transform and its local maxima on GPU with 3x3 mask.
/**
  The mask is applied in parallel to all pixels until no one changes its value.

  \param[in] diff  Source OpenCV matrix with 8bit binary image.
  \param[out] dt   Output OpenCV float matrix with Euclidean distance transform values.
  \param[out] lm   Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \return void
*/
void dt_3x3_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    uint8_t* lmData = lm.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTLM(diffData, dtData, lmData, w, h);
}


/// Compute distance transform and its local maxima on GPU with 5x5 mask.
/**
  The mask is applied in parallel to all pixels until no one changes its value.

  \param[in] diff  Source OpenCV matrix with 8bit binary image.
  \param[out] dt   Output OpenCV float matrix with Euclidean distance transform values.
  \param[out] lm   Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \return void
*/
void dt_5x5_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    uint8_t* lmData = lm.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTLM_5x5(diffData, dtData, lmData, w, h);
}


/// Compute exact Euclidean distance transform and its local maxima on GPU.
/**
  The computation is split into two phases -- for rows and columns.
  In each phase, two scans are performed to find squared distance to the closest pixel
  in the row / column. Then, the squared root is taken,

  \param[in] diff  Source OpenCV matrix with 8bit binary image.
  \param[out] dt   Output OpenCV float matrix with Euclidean distance transform values.
  \param[out] lm   Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \return void
*/
void dt_fast_gpu(const cv::Mat& diff, cv::Mat& dt) {
    const uint8_t* diffData = diff.data;
    float* dtData = (float*) dt.data;
    int w = diff.size().width;
    int h = diff.size().height;

    gpuDTfast(diffData, dtData, w, h);
}

/// Wrapper to compute Euclidean distance transform CPU with different masks.
/**
  Implementation is done in OpenCV library.

  \param[in] srcMat  Source OpenCV matrix with 8bit binary image.
  \param[out] dt     Output OpenCV float matrix with Euclidean distance transform values.
  \param[in] mask    Used mask (0 - precise, 3 - 3x3, 5 - 5x5).
  \return void
*/
void distance_transform(const cv::Mat& srcMat, cv::Mat& dstMat, int mask) {
    cv::distanceTransform(srcMat, dstMat, DIST_TYPE, mask);
}


/// Compute local maxima of given distance transform on CPU.
/**
  Input image is dilated using [1 1 1; 1 1 1; 1 1 1] kernel,
  which computes maximal value of all pixels in the neighbourhood.
  Then, local maxima are the ones which are equal to the maximal value.
  Local maxima of small regions are excluded.

  \param[in] srcMat   Source OpenCV float matrix with Euclidean distance transform values.
  \param[out] dstMat  Output OpenCV 8bit matrix with positive numbers (here 255) indicating local maxima.
  \return void
*/
void local_maxima(const cv::Mat& srcMat, cv::Mat& dstMat) {
    // non maxima suppression
    cv::Mat temp;
    cv::dilate(srcMat, temp, cv::Mat());
    dstMat = srcMat == temp;
    cv::bitwise_and(dstMat, srcMat >= 1.5, dstMat);
}