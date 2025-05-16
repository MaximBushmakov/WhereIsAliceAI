#pragma once

#include "cuda_fp16.h"
//#include "cudnn.h"
#include "curand_kernel.h"
#include <iostream>

using uint = unsigned int;
using uint2_pair = std::pair<uint2, uint2>;
using float2_pair = std::pair<float2, float2>;
using ull = unsigned long long;

#define checkCuda(err) {\
    if (err != cudaSuccess) { \
        std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ \
            << "\n" << cudaGetErrorString(err) << " (" << err << ")" << "\n"; \
        std::exit(1); \
    } \
} \

/*
__host__ __forceinline__ void checkCuDNN(cudnnStatus_t err) {
    if (err != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnn error at " << __FILE__ << ":" << __LINE__ + 1
            << "\n" << cudnnGetErrorString(err) << " (" << err << ")";
        std::exit(1);
    }
}
*/

namespace Utils {

    // read GPU time (ns) in a kernel
    __device__ __forceinline__ ull get_globaltimer() {
        ull time;
        asm ("mov.u64 %0, %globaltimer;" : "=l"(time):);
        return time;
    }

    // uniformely choose int value from [left, right]
    __device__ __forceinline__ int curandInt (curandState* seed, int left = 0, int right = 1) {
        return __float2int_rd(curand_uniform(seed) * (right - left + 1 - 1e-6) + left);
    }

    // uniformely choose unsigned int value from [0, right]
    __device__ __forceinline__ int curandUint (curandState* seed, uint right) {
        return __float2uint_rd(curand_uniform(seed) * (right + 1 - 1e-6));
    }

    template <typename T>
    __host__ __device__ __forceinline__ half getValue(T type) {
        return __float2half_rn((float) type / (uint) T::Last);
    }

    template <typename T>
    __host__ __device__ __forceinline__ T getTyped(half val) {
        return (T)((uint) (__half2float(val) * (uint) T::Last + 1e-6));
    }
}