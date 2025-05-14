#pragma once
#include "character.h"
#include "utils.h"

namespace Simulator {

    enum class ObjectType {
        Clear,
        NoMove,
        Boundary,
        Door,
        Window,
        Cultist,
        CultistChant,
        Astral,
        AstralRush,
        Player,
        Last = Player
    };

    enum class LightingType {
        Dark,
        Normal,
        Madness,
        Last = Madness
    };

    struct Door {
        float2 coords;
        float2 coords_in;
        uint boundary_id;
        uint2 coords_open[4];
        uint2 coords_closed[4];
        bool is_open;
    };

    struct Window {
        float2 coords;
        float2 coords_in;
        float2 coords_out;
    };

    struct InteractableData {
        ObjectType type;
        void* object;
    };

    struct AgentData {
        Character::Type type;
        Character::Agent* agent;
    };

    struct LightingMap {
        LightingType* normal;
        LightingType* madness;
        LightingType* dynamic;
    };

    struct Data {

        uint width;
        uint height;

        struct TensorData {
            half* data;

            __host__ __device__ half* getObject(uint ind) {
                return data + ind * 5;
            }

            __host__ __device__ half* getLighting(uint ind) {
                return data + ind * 5 + 1;
            }

            __host__ __device__ half* getSound(uint ind) {
                return data + ind * 5 + 2;
            }

            __host__ __device__ half* getHearing(uint ind) {
                return data + ind * 5 + 3;
            }

            __host__ __device__ half* getSight(uint ind) {
                return data + ind * 5 + 4;
            }
        };

        TensorData player_tensor;
        TensorData monsters_tensor;

        half* player_hearing_map;
        half* monsters_hearing_map;

        uint agents_size;
        AgentData* agents;

        float health;
        float madness;

        bool madness_world;

        uint boundaries_size;
        float2_pair* boundaries;
        bool** boundaries_masks;
        uint* boundaries_map;

        uint interactables_size;
        InteractableData* interactables;
        bool* interactables_mask;
        uint* interactables_map;

        ObjectType* object_map;

        LightingMap lighting_map;

        bool** sight_masks;

        curandState* seed;
    };
}