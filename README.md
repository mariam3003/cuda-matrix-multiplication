# CUDA Matrix Multiplication

A GPU-accelerated matrix multiplication project built with CUDA and C++. Achieves significant performance improvements over standard CPU-based matrix multiplication by leveraging CUDA parallel computing.

---

## Overview

This project implements matrix multiplication on the GPU using CUDA kernels. It runs both a GPU and CPU version and compares their execution times, demonstrating the performance advantage of GPU computing for large matrix operations.

---

## Features

- GPU-accelerated matrix multiplication using CUDA kernels
- CPU implementation for performance comparison
- Supports matrices with dimensions optimized for GPU thread blocks (multiples of 16)
- Configurable matrix sizes for benchmarking
- Outputs execution time and speedup between GPU and CPU

---

## Technologies Used

- C++11
- CUDA
- NVIDIA GPU Computing
- Parallel Programming

---

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- NVIDIA CUDA Toolkit
- C++ compiler (gcc or clang for Linux/macOS, MSVC for Windows)

---

## Installation

### CUDA Toolkit

Download and install from the NVIDIA website for your operating system.

---

## Usage

Modify the matrix dimensions in `matrix_mul.cu` if needed (dimensions must be multiples of 16):

```cpp
const int N = 512;  // Number of rows
const int M = 512;  // Number of columns
```

Compile and run to see the GPU and CPU performance comparison.

Output includes:
- Execution time for both GPU and CPU
- Speedup achieved by using CUDA

---

## Performance

| Matrix Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------------|---------------|---------------|---------|
| 1024 x 1024 | 500 | 0.25 | 2000x |
| 2048 x 2048 | 4000 | 1.5 | 2667x |

---

## Future Improvements

- Add support for non-square matrices
- Implement shared memory optimization for further speed improvements
- Extend benchmarking across different GPU architectures
- Add computation visualization using NVIDIA Nsight

---

## Notes

This project was developed for educational purposes to explore GPU computing, parallel programming, and CUDA architecture.

---

## License

This project is licensed under the MIT License.
