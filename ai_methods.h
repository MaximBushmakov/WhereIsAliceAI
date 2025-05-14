#pragma once
#include "ai_data.h"
#include "thread_data.h"
#include "utils.h"

namespace AI {

    __host__ DataShared* initShared();
    __host__ DataCopied* initCopied(Simulator::Data* simulator, DataShared* data_shared);

    cudaGraph_t trainForwardStep(Thread::Data* data);
    cudaGraph_t trainBackwardStep(Thread::Data* data);
}