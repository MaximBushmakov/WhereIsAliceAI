# pragma once
#include "utils.h"

namespace AI {

    struct DataShared {
        uint tensors_num;
        uint* tensors_sizes;
        half** tensors_compute;
        half** tensors_write;
        struct {
            // number of threads must be less or equal than 256
            cuda::counting_semaphore<cuda::cuda_threadscope_device, 256>* compute;
            cuda::binary_semaphore<cuda::cuda_treadscope_device>* write;
        } sync;
    };

    struct DataCopied {
        uint tensors_num;
        uint* tensors_sizes;
        half** tensors;
        struct {
            half* input_data;
            half* agent_data;
        } simulator;
    }
}