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

__global__ void calcLM(BYTE *dst, int w, int h){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;
    if(row < h && col < w) {
        float eps = 0;
        float data;
        surf2Dread(&data, surfRef, col* 4, row);
        
        bool islm = false;

        if(data >= 1.5) {
            float data1;
            surf2Dread(&data1, surfRef, (col-1) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col-1) * 4, row+0, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col-1) * 4, row+1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col+0) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col+0) * 4, row+1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col+1) * 4, row-1, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col+1) * 4, row+0, cudaBoundaryModeClamp);
            if(data - data1 >= eps) {
            surf2Dread(&data1, surfRef, (col+1) * 4, row+1, cudaBoundaryModeClamp);
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

__global__ void initDT(BYTE *src, int w, int h){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;

    if(row < h && col < w) {
        float data = 0.f;
        if(src[ind] != 0)
            data = (float)w*h;

        surf2Dwrite(data, surfRef, col*4, row);
    }
}

__global__ void calcDT(int w, int h, int *done){
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
        surf2Dread(&data, surfRef, col* 4, row);
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
                
                surf2Dread(&data, surfRef, (col-1) * 4, row-1, cudaBoundaryModeClamp);
                data += mask[-1+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+0, cudaBoundaryModeClamp);
                data += mask[-1+1][0+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+1, cudaBoundaryModeClamp);
                data += mask[-1+1][1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row-1, cudaBoundaryModeClamp);
                data += mask[0+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row+1, cudaBoundaryModeClamp);
                data += mask[0+1][1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row-1, cudaBoundaryModeClamp);
                data += mask[1+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+0, cudaBoundaryModeClamp);
                data += mask[1+1][0+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+1, cudaBoundaryModeClamp);
                data += mask[1+1][1+1];
                if(newData - data > eps) newData = data;

                if(newData < oldData) {
                    surf2Dwrite(newData, surfRef, col * 4, row);
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

void gpuDTLM(const BYTE *diffData, float *dtData, BYTE *lmData, int w, int h) {
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
    cudaMemcpyToSymbol( mask, tmpMask, sizeof(float)*3*3) ;

    // surface
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuInArray, &channelDesc, w, h, cudaArraySurfaceLoadStore);
    CHECK_ERROR(cudaGetLastError());

    cudaBindSurfaceToArray(surfRef, cuInArray, channelDesc);
    initDT<<<dimGrid, dimBlock>>>(devSrc, w, h);
    cudaDeviceSynchronize();
    int i = 0;
    while(true) {
        ++i;
        cudaMemset(done,1,sizeof(int));
        calcDT<<<dimGrid, dimBlock, sizeof(int) >>>(w,h,done);
        cudaDeviceSynchronize();
        cudaMemcpy(&doneCpu, done, sizeof(int), cudaMemcpyDeviceToHost);
        if(doneCpu > 0) break;
    }
    cudaDeviceSynchronize();
    calcLM<<<dimGrid, dimBlock>>>(devLM, w, h);
    cudaMemcpyFromArray(dtData, cuInArray, 0, 0, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(lmData, devLM, ARRAY_SIZE * sizeof(BYTE), cudaMemcpyDeviceToHost );
    printf("Finished after %d iterations\n", i);
    CHECK_ERROR(cudaGetLastError());
    
    cudaFreeArray(cuInArray);
    cudaFree(devSrc);
    cudaFree(devLM);
    cudaFree(done);

}
