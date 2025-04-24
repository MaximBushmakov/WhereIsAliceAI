#pragma once
#include "simulator_data.h"
#include "ai_data.h"
#include "utils.h"

namespace Simulator {
    __host__ void initHost();
    __host__ Data* copyHostToDevice(Data* data_orig);
    __host__ void deleteHost(Data* data);
    __device__ Data* copyDeviceToDevice(Data* data_orig);
    __host__ void deleteDevice(Data* data);

    __host__ cudaGraph_t step(Data* data);
}