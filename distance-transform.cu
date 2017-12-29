#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "init.h"

surface<void, 2> surfRef;
cudaArray* cuInArray;

__constant__ float mask[3][3];

float a = 0.955;
float b = 1.3693;
float tmpMask[][3] = { {b,a,b}, {a,0,a}, {b,a,b}};

__global__ void initDT(BYTE *src, int w, int h){
    float col = blockIdx.x*blockDim.x + threadIdx.x;
    float row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;

    if(ind < w*h) {
        float data = 0.f;
        if(src[ind] != 0)
            data = w*h;

        surf2Dwrite(data, surfRef, col* 4, row);
    }
}

__global__ void calcDT(int w, int h){
    __shared__ bool found;
    float col = blockIdx.x*blockDim.x + threadIdx.x;
    float row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;

    if(ind < w*h) {
        found = true;
        int inf = w*h;
        int i,j;
        float data;
        surf2Dread(&data, surfRef, col* 4, row);
        if(data > 0) {
            float newData, oldData;
            newData = data;
            oldData = data;
            while(newData >= inf || found) {
                found = false;
//                __syncthreads();

                oldData = newData;
                newData = inf;
                for(i = -1; i <= 1; i++)
                    for(j = -1; j <= 1; j++) {
                        if(i == 0 && j == 0) continue;
                        surf2Dread(&data, surfRef, (col+i) * 4, row+j, cudaBoundaryModeClamp);
                        data += mask[i+1][j+1];
                        if(data < newData) newData = data;
                    }
                if(newData != oldData) {
                    surf2Dwrite(newData, surfRef, col * 4, row);
                    found = true;
                }
//                if(threadIdx.x + threadIdx.y == 0)
//                    printf("[i=%f,j=%f], old=%.2f, new = %.2f\n",col,row,oldData, newData);

                __syncthreads();
            }
        }


    }
}

void gpuDTLM(const BYTE *diffData, float *dtData, BYTE *lmData, int w, int h) {
    int TH = 32;
    dim3 dimBlock(TH,TH);
    int DW = (int) ceil(w/(float)TH);
    int DH = (int) ceil(h/(float)TH);
    dim3 dimGrid(DW,DH);

    int ARRAY_SIZE = w*h;

    BYTE *devSrc;
    cudaMalloc((void **) &devSrc, ARRAY_SIZE * sizeof(BYTE));
    cudaMemcpy(devSrc, diffData, ARRAY_SIZE*sizeof(BYTE), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( mask, tmpMask, sizeof(float)*3*3) ;

    // surface
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuInArray, &channelDesc, w, h, cudaArraySurfaceLoadStore);

    cudaBindSurfaceToArray(surfRef, cuInArray, channelDesc);

    initDT<<<dimGrid, dimBlock>>>(devSrc, w, h);
    calcDT<<<dimGrid, dimBlock>>>(w,h);
    calcDT<<<dimGrid, dimBlock>>>(w,h);
    cudaDeviceSynchronize();
    CHECK_ERROR(cudaGetLastError());
    cudaMemcpyFromArray(dtData, cuInArray, 0, 0, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost );

    cudaFreeArray(cuInArray);
    cudaFree(devSrc);
}
