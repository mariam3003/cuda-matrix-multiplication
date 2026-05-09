#include "matrix_cpu.h"

auto MatrixCPU::Multiply(const float* A, const float* B, float* C, int size) -> void {
    for (int row = 0; row < size; row++) {
        const float* rowA = A + row * size;
        float* rowC = C + row * size;

        for (int col = 0; col < size; col++)
            rowC[col] = 0.0f;

        for (int k = 0; k < size; k++) {
            float aVal = rowA[k];
            const float* rowB = B + k * size;
            for (int col = 0; col < size; col++)
                rowC[col] += aVal * rowB[col];
        }
    }
}

MatrixCPU::MatrixCPU() {}
