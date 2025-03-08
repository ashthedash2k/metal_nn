#include <metal_stdlib>
using namespace metal;

kernel void relu (device float* X, uint id [[thread_position_in_grid]]){
    X[id] = max(0.0, X[id]);
}

kernel void sigmoid (device float* X, uint id [[thread_position_in_grid]]){
    X[id] = 1.0 / (1.0 + exp(-X[id]));
}

kernel void tanh(device float* X, uint id [[thread_position_in_grid]]) {
    X[id] = tanh(X[id]);
}

//softmax will be an asse
kernel void softmax(device float* X, constant uint& size [[buffer(1)]]){
    float sum_exp = 0.0;
    for (uint i = 0; i < size; i++){
        sum_exp += exp(X[i]);
    }
}