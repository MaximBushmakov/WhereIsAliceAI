#include "ai_data.h"
#include "ai_methods.h"
#include "simulator_data.h"
#include "simulator_methods.h"
#include "thread_data.h"
#include "thread_methods.h"
#include "utils.h"

// curand on device
#include <curand_kernel.h>

// for debug purposes
#include <iostream>


/*  startFunc<<<1, 1>>>(start_params);
    while (handle) {
        run out_graph;
        updateFunc<<<1, 1>>>(update_args);
    }
*/
cudaGraph_t cudaGraphWhile(cudaGraph_t graph, cudaGraphConditionalHandle handle,
    void* startFunc, void* start_args[], void* updateFunc, void* update_args[]) {

    cudaGraphNodeParams start_params = { cudaGraphNodeTypeKernel };
    start_params.kernel.func = (void*) startFunc;
    start_params.kernel.gridDim = dim3(1, 1, 1);
    start_params.kernel.blockDim = dim3(1, 1, 1);
    start_params.kernel.kernelParams = start_args;
    cudaGraphNode_t start_node;
    cudaGraphAddNode(&start_node, graph, NULL, 0, &start_params);

    cudaGraphNodeParams cond_params = { cudaGraphNodeTypeConditional };
    cond_params.conditional.handle = handle;
    cond_params.conditional.type = cudaGraphCondTypeWhile;
    cond_params.conditional.size = 1;
    cudaGraphNode_t cond_node;
    cudaGraphAddNode(&cond_node, graph, {&start_node}, 1, &cond_params);

    cudaGraph_t body_graph = condParams.conditional.phGraph_out[0];

    cudaGraphNode_t out_node;
    cudaGraphAddEmptyNode(&out_node, body_graph, NULL, 0);
    cudaGraph_t out_graph;
    cudaGraphCreate(&out_graph, 0);
    cudaGraphChildGraphNodeGetGraph(out_node, &out_graph);

    cudaGraphNodeParams update_params = { cudaGraphNodeTypeKernel };
    update_params.kernel.func = (void*) updateFunc;
    update_params.kernel.gridDim = dim3(1, 1, 1);
    update_params.kernel.blockDim = dim3(1, 1, 1);
    update_params.kernel.kernelParams = update_args;
    cudaGraphNode_t update_node;
    cudaGraphAddNode(&update_node, body_graph, {&out_node}, 1, &update_params);

    return out_graph;
}

__global__ void whileTimeStart(cudaGraphConditionalHandle handle, ull* finish_time, ull time) {
    *finish_time = Utils::get_globaltimer() + time;
    cudaGraphSetConditional(handle, 1);
}

__global__ void whileTimeUpdate(cudaGraphConditionalHandle handle, ull* finish_time) {
    cudaGraphSetConditional(handle, Utils::get_globaltimer() < *finish_time ? 1 : 0);
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

    auto [data, size] = init(threads_num);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraph_t empty_graph;
    cudaGraphCreate(&empty_graph, 0);

    for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {

        cudaGraph_t body_graph = whileTime(work_time * (ull) 1e9, graph);

        // forward step
        cudaGraphNode_t forward_node;
        cudaGraphAddChildGraphNode(&forward_node, body_graph, NULL, 0, empty_graph);
        cudaGraph_t forward_graph;
        cudaGraphChildGraphNodeGetGraph(forward_node, &forward_graph);
        AI::forwardStep(&forward_graph, data[thread_id]);
        
        cudaGraphDestroy(forward_graph);

        // simulator step
        cudaGraphNode_t simulator_node;
        cudaGraphAddChildGraphNode(&simulator_node, body_graph, {&forward_node}, 1, empty_graph);
        cudaGraph_t simulator_graph;
        cudaGraphChildGraphNodeGetGraph(simulator_node, &simulator_graph);
        Simulator::stepReset(&simulator_graph, data[thread_id]->simulator, size, reset_handle);

        // backward step
        cudaGraphNode_t backward_node;
        cudaGraphAddChildGraphNode(&backward_node, body_graph, {&simulator_node}, 1, empty_graph);
        cudaGraph_t backward_graph;
        cudaGraphChildGraphNodeGetGraph(backward_node, &backward_graph);
        AI::backwardStep(&backward_graph, data[thread_id]);

        // if reached end of simulation: reset simulator
        cudaGraphNodeParams reset_cond_params = { cudaGraphNodeTypeConditional };
        reset_cond_params.conditional.handle = reset_handle;
        update_params.kernel.type = cudaGraphCondTypeIf;
        update_params.kernel.size = 1;
        cudaGraphNode_t reset_cond_node;
        cudaGraphAddNode(&reset_cond_node, body_graph, {&backward_node}, 1, &reset_cond_params);
        cudaGraph_t reset_cond_graph = reset_cond_params.conditional.phGraph_out[0];

        cudaGraphNodeParams reset_params = { cudaGraphNodeTypeKernel };
        void* reset_args[] = {&data, &data_base};
        reset_params.kernel.func = (void*) Simulator::copyDeviceToDevice;
        reset_params.kernel.gridDim = dim3(1);
        reset_params.kernel.blockDim = dim3(1);
        reset_params.kernel.kernelParams = reset_args;
        cudaGraphNode_t reset_node;
        cudaGraphAddNode(&reset_node, reset_cond_graph, NULL, 0, &reset_params);
    }

    cudaGraphExec_t graph_exec;

    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graph_exec, 0);

    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    // write weights to system

    // there is some data allocated in host and device memory
    // should be cleaned by OS
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