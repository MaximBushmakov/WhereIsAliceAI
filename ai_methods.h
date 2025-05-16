#pragma once
#include "ai_data.h"
#include "thread_data.h"
#include "utils.h"

namespace AI {

    __host__ DataShared* initShared();
    __host__ DataCopied* initCopied(Simulator::Data* simulator, DataShared* data_shared);

    void trainForwardStep(cudaGraph_t* graph, Thread::Data* data);
    void trainBackwardStep(cudaGraph_t* graph, Thread::Data* data);
}