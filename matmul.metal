#include <metal_stdlib>
using namespace metal;

template <int N>
inline void unrolledMultiply(threadgroup float A[][16], 
                             threadgroup float B[][16], 
                             thread float& sum, uint x, uint y) {
    sum += A[y][N - 1] * B[N - 1][x]; 
    unrolledMultiply<N - 1>(A, B, sum, x, y);
}

template <>
inline void unrolledMultiply<0>(threadgroup float[][16], 
                                threadgroup float[][16], 
                                thread float&, uint, uint) {}

kernel void matmul_unrolled(device const float* A, 
                            device const float* B, 
                            device float* C, 
                            uint2 id [[thread_position_in_grid]],
                            uint2 local_id [[thread_position_in_threadgroup]],
                            uint2 tg_id [[threadgroup_position_in_grid]],
                            constant uint& widthA [[buffer(3)]]) {

    threadgroup float sharedA[16][16];
    threadgroup float sharedB[16][16];

    thread float sum = 0.0;
    uint row = id.y;
    uint col = id.x;

    for (uint tileIdx = 0; tileIdx < widthA / 16; tileIdx++) {
        uint loadRow = local_id.y;
        uint loadCol = local_id.x;

        uint globalIdxA = (tileIdx * 16 + loadRow) * widthA + loadCol;
        uint globalIdxB = (tileIdx * 16 + loadRow) * widthA + col;

        sharedA[loadRow][loadCol] = A[globalIdxA];
        sharedB[loadRow][loadCol] = B[globalIdxB];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        unrolledMultiply<16>(sharedA, sharedB, sum, local_id.x, local_id.y);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * widthA + col] = sum;
}
