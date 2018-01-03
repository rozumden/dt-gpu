#include <stdio.h>

extern void hello(int dev);
extern void gpuLocalMaxima(const float *src, __uint8_t *dst, int w, int h);
extern void gpuDTLM(const __uint8_t *diffData, float *dtData, __uint8_t *lmData, int w, int h);
extern void gpuDTLM_5x5(const __uint8_t *diffData, float *dtData, __uint8_t *lmData, int w, int h);
extern void gpuDTfast(const __uint8_t *diffData, float *dtData, int w, int h);