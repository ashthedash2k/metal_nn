#include <metal_stdlib>
using namespace metal;

kernel void matmul(device const float* A, 
                   device const float* B, 
                   device float* C, 
                   uint2 id [[thread_position_in_grid]], 
                   constant uint& widthA [[buffer(3)]]) {
    
    float sum = 0.0;
    
    if (id.x < widthA && id.y < widthA) {
        for (uint k = 0; k < widthA; k++) {
            sum += A[id.y * widthA + k] * B[k * widthA + id.x];
        }
        C[id.y * widthA + id.x] = sum;  // Store the result
    }
}
