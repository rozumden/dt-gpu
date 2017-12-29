#include <stdio.h>
#include "init.h"

__global__ void mykernel(){
    printf("Hello from GPU!\n");
}

void hello(){
	cudaSetDevice(0);
    mykernel<<<1,1>>>();
    cudaDeviceSynchronize();
    CHECK_ERROR(cudaGetLastError());
}