#include "matrix_gpu.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void TiledMatMulKernel(const float* A, const float* B, float* C, int size) {
    __shared__ float sharedA[MatrixGPU::TILE_SIZE][MatrixGPU::TILE_SIZE];
    __shared__ float sharedB[MatrixGPU::TILE_SIZE][MatrixGPU::TILE_SIZE];

    int row = blockIdx.y * MatrixGPU::TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * MatrixGPU::TILE_SIZE + threadIdx.x;
    float result = 0.0f;

    int numTiles = (size + MatrixGPU::TILE_SIZE - 1) / MatrixGPU::TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * MatrixGPU::TILE_SIZE + threadIdx.x;
        int tiledRow = t * MatrixGPU::TILE_SIZE + threadIdx.y;

        sharedA[threadIdx.y][threadIdx.x] = (row < size && tiledCol < size) ? A[row * size + tiledCol] : 0.0f;
        sharedB[threadIdx.y][threadIdx.x] = (tiledRow < size && col < size) ? B[tiledRow * size + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < MatrixGPU::TILE_SIZE; k++)
            result += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < size && col < size)
        C[row * size + col] = result;
}

auto MatrixGPU::Multiply(const float* A, const float* B, float* C, int size, float* kernelTimeMs) -> bool {
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    size_t bytes = static_cast<size_t>(size) * size * sizeof(float);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (3 * bytes > freeMem) {
        std::cerr << "[MatrixGPU] Not enough GPU memory for size " << size << std::endl;
        return false;
    }

    cudaError_t err;

    err = cudaMalloc(&devA, bytes);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Alloc A failed: " << cudaGetErrorString(err) << std::endl; return false; }

    err = cudaMalloc(&devB, bytes);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Alloc B failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); return false; }

    err = cudaMalloc(&devC, bytes);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Alloc C failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); cudaFree(devB); return false; }

    err = cudaMemcpy(devA, A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Copy A failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); cudaFree(devB); cudaFree(devC); return false; }

    err = cudaMemcpy(devB, B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Copy B failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); cudaFree(devB); cudaFree(devC); return false; }

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    TiledMatMulKernel<<<grid, block>>>(devA, devB, devC, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernelTimeMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Kernel failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); cudaFree(devB); cudaFree(devC); return false; }

    err = cudaMemcpy(C, devC, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "[MatrixGPU] Copy result failed: " << cudaGetErrorString(err) << std::endl; cudaFree(devA); cudaFree(devB); cudaFree(devC); return false; }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    return true;
}
