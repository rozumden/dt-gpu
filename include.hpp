#ifndef FMO_OPENCV_HPP
#define FMO_OPENCV_HPP

#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define DIST_TYPE CV_DIST_L2
#define DIST_MASK_3 CV_DIST_MASK_3
#define DIST_MASK_5 CV_DIST_MASK_5
#define DIST_MASK_PRECISE CV_DIST_MASK_PRECISE

#elif CV_MAJOR_VERSION == 3
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DIST_TYPE cv::DIST_L2 
#define DIST_MASK_3 cv::DIST_MASK_3
#define DIST_MASK_5 cv::DIST_MASK_5
#define DIST_MASK_PRECISE cv::DIST_MASK_PRECISE
#endif

void local_maxima_gpu(const cv::Mat& src, cv::Mat& dst);
void dt_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm, int mask);
void dt_3x3_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm);
void dt_5x5_lm_gpu(const cv::Mat& diff, cv::Mat& dt, cv::Mat &lm);
void dt_fast_gpu(const cv::Mat& diff, cv::Mat& dt);
void distance_transform(const cv::Mat& srcMat, cv::Mat& dstMat, int mask);
void local_maxima(const cv::Mat& srcMat, cv::Mat& dstMat);

#endif // FMO_OPENCV_HPP
