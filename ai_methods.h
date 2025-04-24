#pragma once
#include "ai_data.h"
#include "simulator_data.h"
#include "utils.h"

namespace AI {
    cudnnBackendDescriptor_t tensor3D(int64_t n, int64_t m, int64_t k, int64_t uid, cudnnDataType_t dtype = CUDNN_DATA_HALF, int64_t alignment = 64);

    __host__ DataShared* initShared();
    __host__ DataCopied* initCopied(Simulator::Data simulator);

    cudaGraph_t forwardStep();
    cudaGraph_t backwardStep();
}