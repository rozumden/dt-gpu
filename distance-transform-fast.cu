#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "init.h"


/// Compute squared Euclidean distance transform for each column.
/**
  Two scans are performed to find squared distance to the closest pixel
  in the column, which is stored in shared memoty. 

  \param[in] src       Source array with 8bit binary image.
  \param[out] out      Output int array with squared Euclidean distance transform in columns.
  \param[in] sizeRow   Image widht
  \param[in] sizeCol   Image height 
  \return void
*/
__global__ void computeCol(BYTE* src, int* out, int sizeRow, int sizeCol) {
	extern __shared__ int imgCol []; // allocates shared memory
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	int untilPixel = min(x + sizeRow/blockDim.x, sizeCol);
	int row, rowi;
	int d;
	int value;
	for (row = threadIdx.x; row < sizeCol; row += blockDim.x) {
		imgCol[row] = src[row*sizeRow+y]; // copy column to shared memory
	}
	__syncthreads();
	for(row = x; row < untilPixel; row += blockDim.x) {
		value = imgCol[row];
		if(value != 0) {
			value = sizeRow*sizeRow + sizeCol*sizeCol;
			d = 1;
			for(rowi = 1; rowi < sizeCol - row; rowi++) { // scan 1
				if(imgCol[row + rowi] == 0)
					value = min(value, d);
				d += 1 + 2 * rowi;
				if(d > value) break;
			}
			d = 1;
			for(rowi = 1; rowi <= row; rowi++) { // scan 2
				if(imgCol[row - rowi] == 0)
					value = min(value, d);
				d += 1 + 2 * rowi;
				if(d > value) break;
			}
		}
		out[row * sizeRow + y] = value; 
	}
}


/// Compute squared Euclidean distance transform for each row.
/**
  Two scans are performed to find squared distance to the closest pixel
  in the row, which is stored in shared memoty. 

  \param[in] out       Output int array with squared Euclidean distance transform in columns.
  \param[out] res      Output float array with exact Euclidean distance transform.
  \param[in] sizeRow   Image widht
  \param[in] sizeCol   Image height 
  \return void
*/
__global__ void computeRow(int* out, float* res, int sizeRow, int sizeCol) {
	extern __shared__ int imgRow[]; // allocates shared memory
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * sizeRow;
	int untilPixel = min(x + sizeRow/blockDim.x, sizeRow);
 	int col, coli;
 	int value;
 	int d;
 	for(col = threadIdx.x; col < sizeRow; col += blockDim.x) {
 		imgRow[col] = out[y + col]; // copy rows to shared memory
 	}
 	__syncthreads();
 	for(col = x; col < untilPixel; col += blockDim.x) {
 		value = imgRow[col];
 		if(value != 0) {
	 		d = 1;
			for(coli = 1; coli < sizeRow - col; coli++) { // scan 1
				value = min(value, imgRow[col + coli] + d);
				d += 1 + 2 * coli;
				if(d > value) break;
			}
	 		d = 1;
			for(coli = 1; coli <= col; coli++) { // scan 2
				value = min(value, imgRow[col - coli] + d);
				d += 1 + 2 * coli;
				if(d > value) break;
			}
		}
 		res[y + col] = sqrt((double)value);
	}
}


/// Compute exact Euclidean distance transform on GPU.
/**
  The computation is split into two phases -- for rows and columns.
  In each phase, two scans are performed to find squared distance to the closest pixel
  in the row / column. Then, the squared root is taken,

  \param[in] diffData  Source array with 8bit binary image.
  \param[out] dtData   Output float array with Euclidean distance transform values.
  \param[in] w         Image widht
  \param[in] h         Image height 
  \return void
*/
void gpuDTfast(const BYTE *diffData, float *dtData, int w, int h) {
    /// Maximal number of threads per row/column (can be changed if w/h is lower)
    int MAXTH = 1024;
    int ARRAY_SIZE = w*h;
    
    BYTE *devSrc;
    int *devTemp;
    float *devOut;
    cudaMalloc((void **) &devSrc, ARRAY_SIZE * sizeof(BYTE));
    cudaMalloc((void **) &devTemp, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void **) &devOut, ARRAY_SIZE * sizeof(float));
    cudaMemcpy(devSrc, diffData, ARRAY_SIZE*sizeof(BYTE), cudaMemcpyHostToDevice);

    int TH = MAXTH;
    if(h < TH) TH = h;
    int DH = (int) ceil(h/(float)TH);
    dim3 dimGrid(DH, w, 1);
	computeCol<<<dimGrid, TH, h*sizeof(int)>>>(devSrc, devTemp, w, h);
	cudaDeviceSynchronize();
	CHECK_ERROR(cudaGetLastError());

	int TW = MAXTH;
    if(w < TW) TW = w;
    int DW = (int) ceil(w/(float)TW);
    dim3 dimGridr(DW, h, 1);
	computeRow<<<dimGridr, TW, w*sizeof(int)>>>(devTemp, devOut, w, h);
	cudaDeviceSynchronize();
	CHECK_ERROR(cudaGetLastError());

    cudaMemcpy(dtData, devOut, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
    CHECK_ERROR(cudaGetLastError());

    cudaFree(devSrc);
    cudaFree(devTemp);
    cudaFree(devOut);
}