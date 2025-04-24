%%writefile ai.cu

#include "ai_data.h"
#include "ai_methods.h"
#include "simulator_data.h"
#include "utils.h"

#inlude <random>
#include "cudnn.h"

namespace AI {

    cudnnBackendDescriptor_t tensor3D(int64_t n, int64_t m, int64_t k, int64_t uid, cudnnDataType_t dtype = CUDNN_DATA_HALF, int64_t alignment = 64) {
        cudnnBackendDescriptor_t desc;
        checkCuDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &desc));
        int64_t dim[] = {n, m, k};
        int64_t stride[] = {m * k, k, 1};
        checkCuDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
        checkCuDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 3, dim));
        checkCuDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 3, stride));
        checkCuDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
        checkCuDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
        checkCuDNN(cudnnBackendFinalize(desc));
        return desc;
    }

    cudnnBackendDescriptor_t matmul(cudnnBackendDescriptor_t A, cudnnBackendDescriptor_t B, cudnnBackendDescriptor_t C, cudnnDataType_t dtype = CUDNN_DATA_HALF) {
        cudnnBackendDescriptor_t matmul_params;
        checkCuDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &matmul_params));
        checkCuDNN(cudnnBackendSetAttribute(matmul_params, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
        checkCuDNN(cudnnBackendFinalize(matmul_params));

        cudnnBackendDescriptor_t matmul_oper;
        checkCuDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &matmul_oper));
        checkCuDNN(cudnnBackendSetAttribute(matmul_oper, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &A));
        checkCuDNN(cudnnBackendSetAttribute(matmul_oper, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &B));
        checkCuDNN(cudnnBackendSetAttribute(matmul_oper, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &C));
        checkCuDNN(cudnnBackendSetAttribute(matmul_oper, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul_params));
        checkCuDNN(cudnnBackendFinalize(matmul_oper));

        return matmul_oper;
    }

    cudnnBackendDescriptor_t conv2DParams(cudnnDataType_t dtype = CUDNN_DATA_HALF, cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION) {
        cudnnBackendDescriptor_t conv_params;
        int64_t dim = 2;
        int64_t pre_pad[] = {0, 0};
        int64_t post_pad[] = {0,  0};
        int64_t filter_stride[] = {1, 1};
        int64_t dilation[] = {1, 1};

        cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv);
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &dim));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, dim, pre_pad));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, dim, post_pad));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, dim, filter_stride));
        checkCuDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dim, dilation));
        checkCuDNN(cudnnBackendFinalize(conv));

        return conv;
    }

    __global__ initSyncUpstream(DataShared* data) {
        data->sync->compute = new cuda::counting_semaphore<cuda::cuda_threadscope_device, 256>(0);
        data->sync->write = new cuda::binary_semaphore<cuda::cuda_treadscope_device>(1);
    }

    __host__ DataShared* initShared() {
        DataShared* data_host = (DataShared*) malloc(sizeof(DataShared));

        // user: fill tensors sizes
        uint tensors_sizes_host[] = {
            // conv
            300 * 150 * 5 * 4 * 3 * 3,
            // IB + shortcut
            320 * 160 * 4 * 16,
            320 * 160 * 16 * 3 * 3,
            160 * 160 * 16 * 4,
            160 * 160 * 4 * 4 * 2 * 2,
            // IB + shortcut
            160 * 160 * 4 * 16,
            80 * 80 * 16 * 3 * 3,
            80 * 80 * 16 * 8,
            80 * 80 * 4 * 8 * 2 * 2,
            // IB + shortcut
            80 * 80 * 8 * 32,
            40 * 40 * 32 * 3 * 3,
            40 * 40 * 32 * 16,
            40 * 40 * 8 * 16 * 2 * 2,
            // IB + shortcut
            40 * 40 * 16 * 64,
            20 * 20 * 64 * 3 * 3,
            20 * 20 * 64 * 32,
            20 * 20 * 16 * 32 * 2 * 2,
            // IB + shortcut
            20 * 20 * 32 * 128,
            10 * 10 * 128 * 3 * 3,
            10 * 10 * 128 * 64,
            10 * 10 * 32 * 64 * 2 * 2,
            // conv
            8 * 8 * 64 * 3 * 3,
            // conv 1x1
            8 * 8 * 64 * 64,
            // dense
            4096 * 1024,
            // dense
            1024 * 256
        };

        data_host->tensors_num = sizeof(tensors_sizes_host) / sizeof(tensors_sizes_host[0]);

        cudaMalloc(&data_host->tensor_sizes, sizeof(tensors_sizes_host));
        cudaMemcpy(data_host->tensor_sizes, tensors_sizes_host, sizeof(tensors_sizes_host), cudaMemcpyHostToDevice);

        cudaMalloc(&data_host->tensors_compute, data_host->tensors_num * sizeof(half*));
        cudaMalloc(&data_host->tensors_write, data_host->tensors_num * sizeof(half*));

        half** tensors_compute_host = (half**) malloc(data_host->tensors_num * sizeof(half*));
        half** tensors_write_host = (half**) malloc(data_host->tensors_num * sizeof(half*));

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (uint tensor_id = 0; tensor_id < data_host->tensors_num; ++tensor_id) {
            uint size = tensors_sizes_host[tensor_id];

            half* random_tensor = (half*) malloc(size * sizeof(half));
            for (uint x = 0; x < size; ++x) {
                random_tensor[x] = __float2half_rn(dist(gen));
            }

            cudaMalloc(&tensors_compute_host[tensor_id], size * sizeof(half));
            cudaMalloc(&tensors_write_host[tensor_id], size * sizeof(half));

            cudaMemcpy(tensors_compute_host[tensor_id], random_tensor, size * sizeof(half), cudaMemcpyHostToDevice);
            cudaMemcpy(tensors_write_host[tensor_id], random_tensor, size * sizeof(half), cudaMemcpyHostToDevice);

            free(random_tensor);
        }

        cudaMemcpy(data_host->tensors_compute, tensors_compute_host, data_host->tensors_num * sizeof(half*), cudaMemcpyHostToDevice);
        cudaMemcpy(data_host->tensors_write, tensors_write_host, data_host->tensors_num * sizeof(half*), cudaMemcpyHostToDevice);

        free(tensors_compute_host);
        free(tensors_write_host);

        Data* data_device;
        cudaMalloc(&data, sizeof(DataShared));
        cudaMemcpy(data_device, data_host, sizeof(DataShared), cudaMemcpyHostToDevice);

        initSyncUpstream<<<1, 1>>>(data_device);

        return data_device;
    }

    __global__ releaseComputeUpstream(DataShared* data) {
        data->sync->compute.release();
    }

    __host__ DataCopied* initCopied(Simulator::Data* simulator) {

        DataCopied* data_host = (DataCopied*) malloc(sizeof(DataCopied));

        // user: fill tensors sizes
        uint tensors_sizes_host[] = {
            300 * 150 * 5,
            // conv
            320 * 160 * 4,
            // IB + shortcut
            320 * 160 * 16,
            160 * 160 * 16,
            160 * 160 * 4,
            // IB + shortcut
            160 * 160 * 16,
            80 * 80 * 16,
            80 * 80 * 8,
            // IB + shortcut
            80 * 80 * 32,
            40 * 40 * 32,
            40 * 40 * 16,
            // IB + shortcut
            40 * 40 * 64,
            20 * 20 * 64,
            20 * 20 * 32,
            // IB + shortcut
            20 * 20 * 128,
            10 * 10 * 128,
            10 * 10 * 64,
            // conv without paddings
            8 * 8 * 64,
            // 1x1 conv
            8 * 8 * 64,
            // dense
            1024,
            // dense,
            256 + 16
        };

        data_host->tensors_num = sizeof(tensors_sizes_host) / sizeof(tensors_sizes_host[0]);

        cudaMalloc(&data_host->tensor_sizes, sizeof(tensors_sizes_host));
        cudaMemcpy(data_host->tensor_sizes, tensors_sizes_host, sizeof(tensors_sizes_host), cudaMemcpyHostToDevice);

        cudaMalloc(&data_host->tensors, data_host->tensors_num * sizeof(half*));

        half** tensors_host = (half**) malloc(data_host->tensors_num * sizeof(half*));

        for (uint tensor_id = 0; tensor_id < data_host->tensors_num; ++tensor_id) {
            uint size = tensors_sizes_host[tensor_id];

            half* zero_tensor = (half*) malloc(size * sizeof(half));
            for (uint x = 0; x < size; ++x) {
                zero_tensor[x] = __float2half_rn(0.f);
            }

            cudaMalloc(&tensors_host[tensor_id], size * sizeof(half));
            cudaMemcpy(tensors_host[tensor_id], zero_tensor, size * sizeof(half), cudaMemcpyHostToDevice);
            free(random_tensor);
        }

        cudaMemcpy(data_host->tensors, tensors_host, data_host->tensors_num * sizeof(half*), cudaMemcpyHostToDevice);

        // user: fill pointers to input data and agent data
        data_host->input_data = tensors_host[0];
        data_host->agent_data = tensors_host[20] + 256;

        free(tensors_host);

        Data* data_device;
        cudaMalloc(&data, sizeof(DataCopied));
        cudaMemcpy(data_device, data_host, sizeof(DataCopied), cudaMemcpyHostToDevice);

        releaseComputeUpstream<<<1, 1>>>(data_device);

        return data_device;
    }

    cudaGraph_t forwardStepBase(cudaGraph_t graph) {
        cudnnDataType_t dtype_half = CUDNN_DATA_HALF;
        int64_t alignment = 64;
    }

    cudaGraph_t forwardStep() {
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);

        // user: implement forward pass of training
        // all operations done before simulator update
        forwardStepBase(graph);

        return graph;
    }

    cudaGraph_t backwardStep() {
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
        // user: implement backward pass of training
        // all operations done after simulator update
        return graph;
    }
}