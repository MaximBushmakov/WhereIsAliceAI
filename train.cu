// binary semaphore
#include <cuda/semaphore>

// neural network
#include "cudnn.h"

// curand on device
#include <curand_kernel.h>

// for debug purposes
#include <iostream>

#include "utils.h"

// read GPU time (ns) in a kernel
__forceinline__ __device__ ull get_globaltimer() {
    ull time;
    asm volatile ("mov.u64 %0, %globaltimer;" : "=l"(time));
    return time;
}

// simulator_base and ai_shared are shared between different threads
// links from host to device
struct ThreadData {
    Simulator::Data* simulator_base;
    Simulator::Data* simulator;
    AI::DataShared* ai_shared;
    AI::DataCopied* ai_copied;
}

__host__ ThreadData* init(uint threads_num) {
    ThreadData* data = (ThreadData*) malloc(threads_num * sizeof(ThreadData));

    Simulator::Data* simulator_host = Simulator::initHost();

    data[0]->ai_shared = AI::initShared();
    for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {
        data[thread_id]->ai_shared = data[0]->ai_shared;
        data[thread_id]->ai_copied = AI::initCopied(simulator_host);
    }

    data[0]->simulator_base = Simulator::copyHostToDevice(simulator_host);
    for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {
        data[thread_id]->simulator_base = data[0]->simulator_base;
        data[thread_id]->simulator = Simulator::copyHostToDevice(simulator_host, data[thread_id]->ai_copied);
    }

    return data;
}


/*  startFunc<<<1, 1>>>(start_params);
    while (handle) {
        run out_graph;
        updateFunc<<<1, 1>>>(update_args);
    }
*/
cudaGraph_t cudaGraphWhile(cudaGraph_t graph, cudaGraphConditionalHandle handle,
    void* startFunc, void* start_args[], void* updateFunc, void* update_args[]) {

    cudaGraphNodeParams start_params = { cudaGraphNodeTypeKernel };
    start_params.kernel = {
        .func = startFunc,
        .gridDim = dim3(1, 1, 1),
        .blockDim = dim3(1, 1, 1),
        .kernelParams = start_args
    };
    cudaGraphNode_t start_node;
    cudaGraphAddNode(&start_node, graph, NULL, 0, &start_params);

    cudaGraphNodeParams cond_params = { cudaGraphNodeTypeConditional };
    cond_params.conditional = {
        .handle = handle,
        .type = cudaGraphCondTypeWhile,
        .size = 1
    };
    cudaGraphNode_t cond_node;
    cudaGraphAddNode(&cond_node, graph, {&start_node}, 1, &cond_params);

    cudaGraph_t body_graph = condParams.conditional.phGraph_out[0];

    cudaGraphNode_t out_node;
    cudaGraphAddEmptyNode(&out_node, body_graph, NULL, 0);
    cudaGraph_t out_graph;
    cudaGraphCreate(&out_graph, 0);
    cudaGraphChildGraphNodeGetGraph(out_node, &out_graph);

    cudaGraphNodeParams update_params = { cudaGraphNodeTypeKernel };
    update_params.kernel = {
        .func = updateFunc,
        .gridDim = dim3(1, 1, 1),
        .blockDim = dim3(1, 1, 1),
        .kernelParams = update_args
    };
    cudaGraphNode_t update_node;
    cudaGraphAddNode(&update_node, body_graph, {&out_node}, 1, &update_params);

    return out_graph;
}

__global__ void whileTimeStart(cudaGraphConditionalHandle handle, ull* finish_time, ull time) {
    *finish_time = get_globaltimer() + time;
    cudaGraphSetConditional(handle, 1);
}

__global__ void whileTimeUpdate(cudaGraphConditionalHandle handle, ull* finish_time) {
    cudaGraphSetConditional(handle, get_globaltimer() < *finish_time ? 1 : 0);
}

// while globaltime < start + time: run subgraph
// return subgraph, input time in ns
cudaGraph_t whileTime(ull time, cudaGraph_t graph) {
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);
    volatile ull* finish_time;
    cudaMalloc(&finish_time, sizeof(ull));
    return cudaGraphWhile(graph, handle,
        (void*) whileTimeStart, (void*[3]){&handle, &finish_time, &time},
        (void*) whileTimeUpdate, (void*[2]){&handle, &finish_time});
}


/*  while work time < [work_time] s:
        forward step
        simulator step
        backward step
        if fin: reset simulator
*/
void runAll() {
    // user:: fill train parameters
    const int threads_num = 10;
    const int batch_size = 100;
    const int work_time = 30; // seconds

    ThreadData* data = init(threads_num);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {

        cudaGraph_t body_graph = whileTime(work_time * (ull) 1e9, graph);

        // forward step
        cudaGraph_t forward_graph = AI::forwardStep();
        cudaGraphNode_t forward_node;
        cudaGraphAddChildGraphNode(&forward_node, body_graph, NULL, 0, forward_graph);
        cudaGraphDestroy(forward_graph);

        // simulator step
        cudaGraphConditionalHandle reset_handle;
        cudaGraphConditionalHandleCreate(&reset_handle, body_graph, 0, cudaGraphCondAssignDefault);
        cudaGraph simualtor_graph = Simulator::step(data[thread_id], reset_handle);
        cudaGraphNode_t simulator_node;
        cudaGraphAddChildGraphNode(&simulator_node, body_graph, {&forward_node}, 1, simulator_graph);
        cudaGraphDestroy(simulator_graph);

        // backward step
        cudaGraph_t backward_graph = AI::backwardStep();
        cudaGraphNode_t backward_node;
        cudaGraphAddChildGraphNode(&backward_node, body_graph, {&simulator_node}, 1, backward_graph);
        cudaGraphDestroy(backward_graph);

        // if reached end of simulation: reset simulator
        cudaGraphNodeParams reset_cond_params = { cudaGraphNodeTypeConditional };
        reset_cond_params.conditional = {
            .handle = reset_handle;
            .type = cudaGraphCondTypeIf;
            .size = 1;
        }
        cudaGraphNode_t reset_cond_node;
        cudaGraphAddNode(&reset_cond_node, body_graph, {&backward_node}, 1, &reset_cond_params);
        cudaGraph_t reset_graph = reset_cond_params.conditional.phGraph_out[0];
        Simulator::reset(reset_graph);
    }

    cudaGraphExec_t graph_exec;

    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graph_exec, 0);

    cudaDeviceSynchronize();

    // there is some internal data allocated in device memory
    // should be cleaned by OS

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    // write weights to system

    // for each thread:
        // clear simulator
        // clear ai
    // clear collector
    // clear host data

}


int main() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    runAll();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "CUDA time (ms): " << ms << std::endl;

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
}