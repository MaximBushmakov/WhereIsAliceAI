#pragma once
#include "ai_data.h"
#include "ai_methods.h"
#include "simulator_data.h"
#include "simulator_methods.h"

namespace Thread {
    __host__ std::pair<Data*, uint> init(uint threads_num) {
        Data* data = (Data*) malloc(threads_num * sizeof(Data));

        Simulator::Data* simulator_host = Simulator::initHost();

        data[0].simulator_base = Simulator::copyHostToDevice(simulator_host);
        for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {
            data[thread_id].simulator_base = data[0].simulator_base;
            data[thread_id].simulator = Simulator::copyHostToDevice(simulator_host);
        }

        uint size = simulator_host->height * simulator_host->width;

        Simulator::deleteHost(simulator_host);

        data[0].ai_shared = AI::initShared();
        for (uint thread_id = 0; thread_id < threads_num; ++thread_id) {
            data[thread_id].ai_shared = data[0].ai_shared;
            data[thread_id].ai_copied = AI::initCopied(data[thread_id].simulator, data[0].ai_shared);
        }

        return {data, size};
    }
}