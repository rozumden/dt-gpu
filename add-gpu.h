//----------------------------------------------------------------------------------------
/**
 * \file       add-gpu.h
 * \author     Denys Rozumnyi
 * \date       2018/01/04
 * \brief      Description of communication with GPU
 *
 *  Each function here provides a description of calling GPU-based algorithms.
 *	These functions will be compiled using nvcc (CUDA) and linked.
*/
//----------------------------------------------------------------------------------------
#ifndef _ADD_GPU_H_
#define _ADD_GPU_H_

#include <stdio.h>

extern void hello(int dev);
extern void gpuLocalMaxima(const float *src, __uint8_t *dst, int w, int h);
extern void gpuDTLM(const __uint8_t *diffData, float *dtData, __uint8_t *lmData, int w, int h);
extern void gpuDTLM_5x5(const __uint8_t *diffData, float *dtData, __uint8_t *lmData, int w, int h);
extern void gpuDTfast(const __uint8_t *diffData, float *dtData, int w, int h);

#endif // _ADD_GPU_H_