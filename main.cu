#include "matrix_cpu.h"
#include "matrix_gpu.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>

static bool Validate(const float* cpuResult, const float* gpuResult, size_t totalElements, float tolerance = 1e-2f) {
    for (size_t i = 0; i < totalElements; i++) {
        if (std::fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            std::cerr << "[Validate] Mismatch at index " << i
                      << " CPU=" << cpuResult[i]
                      << " GPU=" << gpuResult[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int size = 1024;

    if (argc == 2) {
        size = std::atoi(argv[1]);
        if (size <= 0) {
            std::cerr << "Matrix size must be a positive integer." << std::endl;
            return 1;
        }
    }

    size_t total = static_cast<size_t>(size) * size;

    float* A         = new float[total];
    float* B         = new float[total];
    float* cpuResult = new float[total];
    float* gpuResult = new float[total];

    std::srand(42);
    for (int i = 0; i < total; i++) {
        A[i] = static_cast<float>(std::rand() % 10);
        B[i] = static_cast<float>(std::rand() % 10);
    }

    std::cout << "Matrix Size     : " << size << " x " << size << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    const auto cpuStart = std::chrono::steady_clock::now();
    MatrixCPU::Multiply(A, B, cpuResult, size);
    const auto cpuEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> cpuTime{ cpuEnd - cpuStart };

    std::cout << "CPU Time        : " << cpuTime.count() << " ms" << std::endl;

    float kernelTimeMs = 0.0f;
    bool gpuSuccess = MatrixGPU::Multiply(A, B, gpuResult, size, &kernelTimeMs);

    if (!gpuSuccess) {
        std::cerr << "GPU computation failed. Skipping validation and speedup." << std::endl;
        delete[] A; delete[] B; delete[] cpuResult; delete[] gpuResult;
        return 1;
    }

    std::cout << "GPU Kernel Time : " << kernelTimeMs << " ms (kernel only)" << std::endl;

    bool passed = Validate(cpuResult, gpuResult, total);
    std::cout << "Validation      : " << (passed ? "PASSED" : "FAILED") << std::endl;

    if (passed) {
        double speedup = cpuTime.count() / kernelTimeMs;
        std::cout << "Speedup         : " << speedup << "x" << std::endl;
    }

    std::cout << "-----------------------------------------------" << std::endl;

    delete[] A;
    delete[] B;
    delete[] cpuResult;
    delete[] gpuResult;

    return 0;
}
