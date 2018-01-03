#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdlib.h"
#include <cmath>
#include "init.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
cudaArray *cuArray;

__global__ void findLM(BYTE *dst, int w, int h){
    double col = blockIdx.x*blockDim.x + threadIdx.x;
    double row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * w + col;
    float eps = 0;
    if(row < h && col < w) {
        row += 0.5;
        col += 0.5;
        row /= h;
        col /= w;
        double row1 = 1.f/h;
        double col1 = 1.f/w;

        float inp = tex2D(texRef, col, row);
        if(inp >= 1.5 &&
           inp - tex2D(texRef, col-col1, row) >= eps &&
           inp - tex2D(texRef, col-col1, row-row1) >= eps  &&
           inp - tex2D(texRef, col-col1, row+row1) >= eps &&
           inp - tex2D(texRef, col+col1, row) >= eps &&
           inp - tex2D(texRef, col+col1, row-row1) >= eps &&
           inp - tex2D(texRef, col+col1, row+row1) >= eps &&
           inp - tex2D(texRef, col, row-row1) >= eps &&
           inp - tex2D(texRef, col, row+row1) >= eps)
        {
            dst[ind] = 255;
        } else {
            dst[ind] = 0;
        }
              
    }
}

void gpuLocalMaxima(const float *src, BYTE *dst, int w, int h){
    int TH = 32;
    dim3 dimBlock(TH,TH);
    int DW = (int) ceil(w/(float)TH);
    int DH = (int) ceil(h/(float)TH);
    dim3 dimGrid(DW,DH);

    int ARRAY_SIZE = w*h;
    BYTE* devDst;

    /// textures
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuArray, &channelDesc, w, h);
    cudaMemcpyToArray(cuArray, 0, 0, src, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = true;
    cudaBindTextureToArray(texRef, cuArray, channelDesc);
    //////

    cudaMalloc((void **) &devDst, ARRAY_SIZE * sizeof(BYTE));

    findLM<<<dimGrid, dimBlock>>>(devDst,w,h);
    CHECK_ERROR(cudaGetLastError());

    cudaDeviceSynchronize();

    cudaMemcpy(dst, devDst, ARRAY_SIZE*sizeof(BYTE), cudaMemcpyDeviceToHost);

    cudaFree(devDst);
    cudaUnbindTexture(texRef);
    cudaFreeArray(cuArray);
}

