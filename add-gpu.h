#include <stdio.h>

extern void hello();
extern void gpuLocalMaxima(const float *src, __uint8_t *dst, int w, int h);
extern void gpuDTLM(const __uint8_t *diffData, float *dtData, __uint8_t *lmData, int w, int h);