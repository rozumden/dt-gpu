#include <add-gpu.h>

namespace fmo {
    void test_gpu() {
        hello();
    }

    void local_maxima_gpu(const Mat& src, Mat& dst) {
        dst.resize(Format::GRAY, src.dims());
        const float* srcData = (const float*) src.data();
        uint8_t* dstData = dst.data();
        int w = src.dims().width;
        int h = src.dims().height;

        gpuLocalMaxima(srcData, dstData, w, h);
    }

    void dt_lm_gpu(const Mat& diff, Mat& dt, Mat &lm) {
        dt.resize(Format::FLOAT, diff.dims());
        lm.resize(Format::GRAY, diff.dims());
        const uint8_t* diffData = diff.data();
        float* dtData = (float*) dt.data();
        uint8_t* lmData = lm.data();
        int w = diff.dims().width;
        int h = diff.dims().height;

        gpuDTLM(diffData, dtData, lmData, w, h);
    }

}
