#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "init.h"

surface<void, 2> surfRef5;
cudaArray* cuInArray5;

__constant__ float mask5[5][5];

float a5 = 1;
float b5 = 1.4;
float c5 = 2.1969;
float tmpMask5[][5] = { {-1,c5,-1,c5,-1}, {c5,b5,a5,b5,c5}, {-1,a5,0,a5,-1},{c5,b5,a5,b5,c5}, {-1,c5,-1,c5,-1}};

__global__ void calcLM_5x5(BYTE *dst, int w, int h){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;
    if(row < h && col < w) {
        float eps = 0;
        float data;
        surf2Dread(&data, surfRef5, col* 4, row);
        
        bool islm = false;

        if(data >= 1.5) {
            float data1;
            surf2Dread(&data1, surfRef5, (col-1) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col-1) * 4, row+0, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col-1) * 4, row+1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col+0) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col+0) * 4, row+1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col+1) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col+1) * 4, row+0, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef5, (col+1) * 4, row+1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) islm = true;
            }}}}}}}
        } 
        if(islm) {
            dst[ind] = 255;
        } else {
            dst[ind] = 0;
        }
    }
}

__global__ void initDT_5x5(BYTE *src, int w, int h){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;

    if(row < h && col < w) {
        float data = 0.f;
        if(src[ind] != 0)
            data = (float)w*h;

        surf2Dwrite(data, surfRef5, col*4, row);
    }
}

__global__ void calcDT_5x5(int w, int h, int *done){
    __shared__ int found;
    bool written = true;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    bool mainthread = (threadIdx.x + threadIdx.y == 0);

    if(row < h && col < w) {
        if(mainthread) {
            written = false;
            atomicExch(&found, 1);
        }
        __syncthreads();

        int inf = 2147483647;
        float data;
        float eps = 0;
        surf2Dread(&data, surfRef5, col* 4, row);
        if(data > 0 || mainthread) {
            float newData, oldData;
            newData = data;
            oldData = data;
            while(found > 0) {
                if(mainthread) {
                    atomicExch(&found, 0);
                }
                __syncthreads();

                oldData = newData;
                newData = inf;
                
                surf2Dread(&data, surfRef5, (col-1) * 4, row-1, cudaBoundaryModeClamp);
                data += mask5[-1+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-1) * 4, row+0, cudaBoundaryModeClamp);
                data += mask5[-1+2][0+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-1) * 4, row+1, cudaBoundaryModeClamp);
                data += mask5[-1+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+0) * 4, row-1, cudaBoundaryModeClamp);
                data += mask5[0+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+0) * 4, row+1, cudaBoundaryModeClamp);
                data += mask5[0+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+1) * 4, row-1, cudaBoundaryModeClamp);
                data += mask5[1+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+1) * 4, row+0, cudaBoundaryModeClamp);
                data += mask5[1+2][0+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+1) * 4, row+1, cudaBoundaryModeClamp);
                data += mask5[1+2][1+2];
                if(newData - data > eps) newData = data;

                /// for c's
                surf2Dread(&data, surfRef5, (col+2) * 4, row-1, cudaBoundaryModeClamp);
                data += mask5[2+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+2) * 4, row+1, cudaBoundaryModeClamp);
                data += mask5[2+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-2) * 4, row-1, cudaBoundaryModeClamp);
                data += mask5[-2+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-2) * 4, row+1, cudaBoundaryModeClamp);
                data += mask5[-2+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-1) * 4, row-2, cudaBoundaryModeClamp);
                data += mask5[-1+2][-2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+1) * 4, row-2, cudaBoundaryModeClamp);
                data += mask5[1+2][-2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col-1) * 4, row+2, cudaBoundaryModeClamp);
                data += mask5[-1+2][2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef5, (col+1) * 4, row+2, cudaBoundaryModeClamp);
                data += mask5[1+2][2+2];
                if(newData - data > eps) newData = data;

                if(newData < oldData) {
                    surf2Dwrite(newData, surfRef5, col * 4, row);
                    atomicExch(&found, 1);
                }

                __syncthreads();
                if(mainthread && found > 0 && !written) {
                    atomicExch(done, 0);
                    written = true;
                }
            }
        }


    }
}

void gpuDTLM_5x5(const BYTE *diffData, float *dtData, BYTE *lmData, int w, int h) {
    int TH = 32;
    dim3 dimBlock(TH,TH);
    int DW = (int) ceil(w/(float)TH);
    int DH = (int) ceil(h/(float)TH);
    dim3 dimGrid(DW,DH);

    int ARRAY_SIZE = w*h;

    BYTE *devSrc, *devLM;
    int *done;
    int doneCpu = 1;
    cudaMalloc((void **) &done, sizeof(int));
    cudaMalloc((void **) &devSrc, ARRAY_SIZE * sizeof(BYTE));
    cudaMalloc((void **) &devLM, ARRAY_SIZE * sizeof(BYTE));
    cudaMemcpy(devSrc, diffData, ARRAY_SIZE*sizeof(BYTE), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( mask5, tmpMask5, sizeof(float)*5*5) ;

    // surface
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuInArray5, &channelDesc, w, h, cudaArraySurfaceLoadStore);

    cudaBindSurfaceToArray(surfRef5, cuInArray5, channelDesc);
    initDT_5x5<<<dimGrid, dimBlock>>>(devSrc, w, h);
    cudaDeviceSynchronize();
    int i = 0;
    while(true) {
        i++;
        cudaMemset(done,1,sizeof(int));
        calcDT_5x5<<<dimGrid, dimBlock, sizeof(int) >>>(w,h,done);
        cudaDeviceSynchronize();
        cudaMemcpy(&doneCpu, done, sizeof(int), cudaMemcpyDeviceToHost);
        if(doneCpu > 0) break;
    }
    cudaDeviceSynchronize();
    calcLM_5x5<<<dimGrid, dimBlock>>>(devLM, w, h);
    cudaMemcpyFromArray(dtData, cuInArray5, 0, 0, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(lmData, devLM, ARRAY_SIZE * sizeof(BYTE), cudaMemcpyDeviceToHost );
    printf("Finished after %d iterations\n", i);
    CHECK_ERROR(cudaGetLastError());
    
    cudaFreeArray(cuInArray5);
    cudaFree(devSrc);
    cudaFree(devLM);
    cudaFree(done);
}
