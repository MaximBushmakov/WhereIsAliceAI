#pragma once

using uint = unsigned int;
using uint2_pair = std::pair<uint2, uint2>;
using ull = unsigned long long;
using half = __half;

__forceinline__ __host__ void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ + 1
            << "\n" << cudaGetErrorString(err) << " (" << err << ")";
        std::exit(1);
    }
}

__forceinline__ __host__ void checkCuDNN(cudnnStatus_t err) {
    if (err != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnn error at " << __FILE__ << ":" << __LINE__ + 1
            << "\n" << cudnnGetErrorString(err) << " (" << err << ")";
        std::exit(1);
    }
}

namespace Utils {

    __device__ int curandInt (int left, int right, curandState* seed) {
        return __float2int_rd(curand_uniform(seed) * (right - left - 1e-6) + left);
    }

    __device__ int curandUint (uint left, uint right, curandState* seed) {
        return __float2uint_rd(curand_uniform(seed) * (right - left - 1e-6) + left);
    }

    template <typename T>
    __host__ __device__ half getValue(T type) {
        return __uint2half_rn((uint) type) / T::Last;
    }

    template <typename T>
    __host__ __device__ T getTyped(half val) {
        return (T)(__half2uint_rn(val * T::Last));
    }
}