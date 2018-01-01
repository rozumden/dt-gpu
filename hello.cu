#include <stdio.h>
#include "init.h"

__global__ void mykernel(){
    printf("Hello from GPU!\n");
}

void hello(int dev){
	cudaSetDevice(dev);
    CHECK_ERROR(cudaGetLastError());
    mykernel<<<1,1>>>();
    CHECK_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
    CHECK_ERROR(cudaGetLastError());
}