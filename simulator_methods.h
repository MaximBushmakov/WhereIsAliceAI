#pragma once
#include "ai_data.h"
#include "thread_data.h"
#include "simulator_data.h"
#include "utils.h"

namespace Simulator {
    __host__ Data* initHost();
    __host__ Data* copyHostToDevice(Data* data_orig);
    __host__ Data* copyDeviceToHost(Data* data_orig);
    __host__ void deleteHost(Data* data);
    __device__ void copyDeviceToDevice(Data* data_orig, Data* data_dest);
    __host__ void deleteDevice(Data* data);

    __host__ cudaGraph_t step(Data* data, uint size);
    __host__ cudaGraph_t step(Data* data, uint size, cudaGraphConditionalHandle reset_handle);
}