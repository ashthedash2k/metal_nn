#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define TILE_SIZE 32  // Tune based on CPU cache size

extern "C" {

// Naïve Single-Threaded CPU MatMul
void matmul_naive(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Optimized CPU MatMul (OpenMP + Cache Blocking)
void matmul_optimized(const float* A, const float* B, float* C, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i += TILE_SIZE) {
        for (int j = 0; j < size; j += TILE_SIZE) {
            for (int k = 0; k < size; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < size; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < size; jj++) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + TILE_SIZE && kk < size; kk++) {
                            sum += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] += sum;
                    }
                }
            }
        }
    }
}

double get_naive_cpu_time(const float* A, const float* B, float* C, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

double get_optimized_cpu_time(const float* A, const float* B, float* C, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul_optimized(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}
}
