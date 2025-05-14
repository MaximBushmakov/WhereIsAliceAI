#pragma once

#include "ai_data.h"
#include "simulator_data.h"

namespace Thread {
    // links from host to device
    struct Data {
        Simulator::Data* simulator_base;
        Simulator::Data* simulator;
        AI::DataShared* ai_shared;
        AI::DataCopied* ai_copied;
    };
}