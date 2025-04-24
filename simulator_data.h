#pragma once
#include "utils.h"

namespace Simulator {

    enum class ObjectType {
        Clear,
        NoMove,
        Boundary,
        DoorClosed,
        DoorOpen,
        Window,
        Cultist,
        Astral,
        Player,
        Last = Player
    };

    enum class LightingType {
        Dark,
        Normal,
        Madness,
        Last = Madness
    };

    struct Data {

        struct TensorData {
            const uint size;
            const half* data;

            __host__ __device__ half* getObjectMap() {
                return data;
            }

            __host__ __device__ half* getLightingMap() {
                return data + size;
            }

            __host__ __device__ half* getSoundMap() {
                return data + 2 * size;
            }

            __host__ __device__ half* getHearingMap() {
                return data + 3 * size;
            }

            __host__ __device__ half* getSightMap() {
                return data + 4 * size;
            }

            __host__ __device__ uint getSize() {
                return 5 * size;
            }
        };

        uint width;
        uint height;

        uint boundaries_size;
        uint2_pair* boundaries;
        uint* boundaries_map;

        uint roofed_size;
        uint2_pair* roofed_area;
        uint lighted_size;
        uint2_pair* lighted_area;

        uint agents_size;
        Character::Agent** agents;

        bool madness;

        TensorData player;
        TensorData monsters;
    }
}