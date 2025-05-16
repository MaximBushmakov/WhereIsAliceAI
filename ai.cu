#include "ai_data.h"
#include "ai_methods.h"
#include "simulator_data.h"
#include "utils.h"

#include <random>
// #include "cudnn.h"

namespace AI {

    /*
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
    */

    __global__ void initSyncUpstream(DataShared* data) {
        data->sync.compute = new cuda::counting_semaphore<cuda::thread_scope_device, 256>(0);
        data->sync.write = new cuda::binary_semaphore<cuda::thread_scope_device>(1);
    }

    __global__ void acquireComputeUpstream(DataShared* data) {
        data->sync.compute->acquire();
    }

    __global__ void releaseComputeUpstream(DataShared* data) {
        data->sync.compute->release();
    }

    __global__ void acquireAllComputeUpstream(DataShared* data) {
        for (uint i = 0; i < data->copies_num; ++i) {
            data->sync.compute->acquire();
        }
    }

    __global__ void releaseAllComputeUpstream(DataShared* data) {
        for (uint i = 0; i < data->copies_num; ++i) {
            data->sync.compute->release();
        }
    }

    __global__ void acquireWriteUpstream(DataShared* data) {
        data->sync.write->acquire();
    }

    __global__ void releaseWriteUpstream(DataShared* data) {
        data->sync.write->release();
    }

    __global__ void increaseCopiesNumUpstream(DataShared* data) {
        ++data->copies_num;
    }

    __host__ DataShared* initShared() {
        DataShared* data_host = (DataShared*) malloc(sizeof(DataShared));

        // user: fill tensors sizes
        uint tensors_sizes_host[] = {
            0
        };

        data_host->copies_num = 0;

        data_host->tensors_num = sizeof(tensors_sizes_host) / sizeof(tensors_sizes_host[0]);

        cudaMalloc(&data_host->tensors_sizes, sizeof(tensors_sizes_host));
        cudaMemcpy(data_host->tensors_sizes, tensors_sizes_host, sizeof(tensors_sizes_host), cudaMemcpyHostToDevice);

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

        DataShared* data_device;
        cudaMalloc(&data_device, sizeof(DataShared));
        cudaMemcpy(data_device, data_host, sizeof(DataShared), cudaMemcpyHostToDevice);

        initSyncUpstream<<<1, 1>>>(data_device);

        return data_device;
    }

    __host__ DataCopied* initCopied(Simulator::Data* simulator, DataShared* data_shared) {
        
        DataCopied* data_host = (DataCopied*) malloc(sizeof(DataCopied));

        data_host->simulator.player_input = simulator->player_tensor.data;
        data_host->simulator.monsters_input = simulator->monsters_tensor.data;
        data_host->simulator.agents_data = (half**) malloc(simulator->agents_size * sizeof(half*));
        for (uint id = 0; id < simulator->agents_size; ++id) {
            data_host->simulator.agents_data[id] = simulator->agents[id].agent->data;
        }

        // user: fill tensors sizes
        uint tensors_sizes_host[] = {
            0
        };

        data_host->tensors_num = sizeof(tensors_sizes_host) / sizeof(tensors_sizes_host[0]);

        cudaMalloc(&data_host->tensors_sizes, sizeof(tensors_sizes_host));
        cudaMemcpy(data_host->tensors_sizes, tensors_sizes_host, sizeof(tensors_sizes_host), cudaMemcpyHostToDevice);

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
            free(zero_tensor);
        }

        cudaMemcpy(data_host->tensors, tensors_host, data_host->tensors_num * sizeof(half*), cudaMemcpyHostToDevice);

        free(tensors_host);

        DataCopied* data_device;
        cudaMalloc(&data_device, sizeof(DataCopied));
        cudaMemcpy(data_device, data_host, sizeof(DataCopied), cudaMemcpyHostToDevice);

        releaseComputeUpstream<<<1, 1>>>(data_shared);
        increaseCopiesNumUpstream<<<1, 1>>>(data_shared);

        return data_device;
    }

    void trainForwardStep(cudaGraph_t* graph, Thread::Data* data) {

        cudaGraphNodeParams compute_acquire_params = { cudaGraphNodeTypeKernel };
        void* compute_acquire_args[] = {&data->ai_shared};
        compute_acquire_params.kernel.func = acquireComputeUpstream;
        compute_acquire_params.kernel.gridDim = dim3(1, 1, 1);
        compute_acquire_params.kernel.blockDim = dim3(1, 1, 1);
        compute_acquire_params.kernel.kernelParams = compute_acquire_args;
        cudaGraphNode_t compute_acquire_node;
        cudaGraphAddNode(&compute_acquire_node, *graph, NULL, 0, &compute_acquire_params);

        // user: implement forward pass of training
        // all operations done before simulator update

        /* sample of cudnn populate:
        cudaGraph_t forward_graph;
        cudaGraphCreate(&forward_graph, 0);
        auto [forward_handle, forward_plan, forward_varpack] = ...;
        cudnnBackendPopulateCudaGraph(forward_handle, forward_plan, forward_varpack, forward_graph);
        cudaGraphNode_t forward_node;
        cudaGraphAddChildGraphNode(&forward_node, *graph, NULL, 0, forward_graph);
        cudaGraphDestroy(forward_graph);
        */
    }

    void trainBackwardStep(cudaGraph_t* graph, Thread::Data* data) {

        cudaGraphConditionalHandle update_handle;
        cudaGraphConditionalHandleCreate(&update_handle, *graph, 0, cudaGraphCondAssignDefault);

        // user: implement backward pass of training
        // all operations done after simulator update
        // set update_handle to true for copy

        cudaGraphNodeParams write_acquire_params = { cudaGraphNodeTypeKernel };
        void* write_acquire_args[] = {&data->ai_shared};
        write_acquire_params.kernel.func = acquireWriteUpstream;
        write_acquire_params.kernel.gridDim = dim3(1, 1, 1);
        write_acquire_params.kernel.blockDim = dim3(1, 1, 1);
        write_acquire_params.kernel.kernelParams = write_acquire_args;
        cudaGraphNode_t write_acquire_node;
        cudaGraphAddNode(&write_acquire_node, *graph, NULL, 0, &write_acquire_params);

        // user: implement update

        cudaGraphNodeParams write_release_params = { cudaGraphNodeTypeKernel };
        void* write_release_args[] = {&data->ai_shared};
        write_release_params.kernel.func = releaseWriteUpstream;
        write_release_params.kernel.gridDim = dim3(1, 1, 1);
        write_release_params.kernel.blockDim = dim3(1, 1, 1);
        write_release_params.kernel.kernelParams = write_release_args;
        cudaGraphNode_t write_release_node;
        cudaGraphAddNode(&write_release_node, *graph, NULL, 0, &write_release_params);

        cudaGraphNodeParams update_cond_params = { cudaGraphNodeTypeConditional };
        update_cond_params.conditional.handle = update_handle;
        update_cond_params.conditional.type = cudaGraphCondTypeIf;
        update_cond_params.conditional.size = 1;
        cudaGraphNode_t update_cond_node;
        cudaGraphAddNode(&update_cond_node, *graph, {&write_release_node}, 1, &update_cond_params);
        cudaGraph_t update_cond_graph = update_cond_params.conditional.phGraph_out[0];

        cudaGraphNodeParams compute_acquire_all_params = { cudaGraphNodeTypeKernel };
        void* compute_acquire_all_args[] = {&data->ai_shared};
        compute_acquire_all_params.kernel.func = acquireAllComputeUpstream;
        compute_acquire_all_params.kernel.gridDim = dim3(1, 1, 1);
        compute_acquire_all_params.kernel.blockDim = dim3(1, 1, 1);
        compute_acquire_all_params.kernel.kernelParams = compute_acquire_all_args;
        cudaGraphNode_t compute_acquire_all_node;
        cudaGraphAddNode(&compute_acquire_all_node, update_cond_graph, NULL, 0, &compute_acquire_all_params);

        // user: implement copy

        cudaGraphNodeParams compute_release_all_params = { cudaGraphNodeTypeKernel };
        void* compute_release_all_args[] = {&data->ai_shared};
        compute_release_all_params.kernel.func = releaseAllComputeUpstream;
        compute_release_all_params.kernel.gridDim = dim3(1, 1, 1);
        compute_release_all_params.kernel.blockDim = dim3(1, 1, 1);
        compute_release_all_params.kernel.kernelParams = compute_release_all_args;
        cudaGraphNode_t compute_release_all_node;
        cudaGraphAddNode(&compute_release_all_node, update_cond_graph, NULL, 0, &compute_release_all_params);

        cudaGraphNodeParams compute_release_params = { cudaGraphNodeTypeKernel };
        void* compute_release_args[] = {&data->ai_shared};
        compute_release_params.kernel.func = releaseComputeUpstream;
        compute_release_params.kernel.gridDim = dim3(1, 1, 1);
        compute_release_params.kernel.blockDim = dim3(1, 1, 1);
        compute_release_params.kernel.kernelParams = compute_release_args;
        cudaGraphNode_t compute_release_node;
        cudaGraphAddNode(&compute_release_node, *graph, {&update_cond_node}, 1, &compute_release_params);
    }
}