#pragma once

class MatrixCPU {
public:
    static auto Multiply(const float* A, const float* B, float* C, int size) -> void;

private:
    MatrixCPU();
};
