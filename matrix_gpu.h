#pragma once

class MatrixGPU {
public:
    static const int TILE_SIZE = 16;
    static auto Multiply(const float* A, const float* B, float* C, int size, float* kernelTimeMs) -> bool;

private:
    MatrixGPU();
};
