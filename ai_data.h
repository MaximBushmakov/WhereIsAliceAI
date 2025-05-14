# pragma once
#include "utils.h"
#include <cuda/semaphore>

namespace AI {

    struct DataShared {
        uint copies_num;
        uint tensors_num;
        uint* tensors_sizes;
        half** tensors_compute;
        half** tensors_write;
        struct {
            // number of threads must be less or equal than 256
            cuda::counting_semaphore<cuda::thread_scope_device, 256>* compute;
            cuda::binary_semaphore<cuda::thread_scope_device>* write;
        } sync;
    };

    struct DataCopied {
        uint tensors_num;
        uint* tensors_sizes;
        half** tensors;
        struct {
            half* player_input;
            half* monsters_input;
            half** agents_data;
        } simulator;
    };
}
