#include "character.h"
#include "simulator_data.h"
#include "simulator_methods.h"
#include "thread_data.h"
#include "utils.h"

#include <iostream>
#include <cstring>
#include <algorithm>

#define _USE_MATH_DEFINES
#include "math.h"

#include "curand_kernel.h"

using Character::Agent;
using Character::Player;
using Character::Astral;
using Character::Cultist;

namespace Simulator {

    template <typename T>
    __host__ __forceinline__ void copyAllocHostToDevice(T** p, uint size) {
        T* orig = *p;
        cudaMalloc(p, size * sizeof(T));
        cudaMemcpy(*p, orig, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    __host__ __forceinline__ void copyAllocDeviceToHost(T** p, uint size) {
        T* orig = *p;
        *p = (T*) malloc(size * sizeof(T));
        cudaMemcpy(*p, orig, size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <typename T> 
    __host__ __forceinline__ T* copyAllocHostToHost(T** p, uint size) {
        T* dest = (T*) malloc(size * sizeof(T));
        std::memcpy(dest, *p, size * sizeof(T));
        *p = dest;
        return dest;
    }

    __global__ void curandInitUpstream(curandState* seed) {
        // use pointer address as initial seed
        curand_init((unsigned long long) seed, 0, 0, seed);
    }

    __host__ Data* copyHostToDevice(Data* data_orig) {

        Data data = *data_orig;

        copyAllocHostToDevice<half>(&data.player_tensor.data, 5 * data.height * data.width);
        copyAllocHostToDevice<half>(&data.monsters_tensor.data, 5 * data.height * data.width);

        copyAllocHostToDevice<half>(&data.player_hearing_map, data.height * data.width);
        copyAllocHostToDevice<half>(&data.monsters_hearing_map, data.height * data.width);

        bool** sight_masks = copyAllocHostToHost<bool*>(&data.sight_masks, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            if (agent_id == 0 || data.agents[agent_id].type == Character::Type::Astral) {
                copyAllocHostToDevice<bool>(&sight_masks[agent_id], data.height * data.width);
            }
        }
        copyAllocHostToDevice<bool*>(&data.sight_masks, data.agents_size);
        free(sight_masks);

        bool** boundaries_masks = copyAllocHostToHost<bool*>(&data.boundaries_masks, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            if (agent_id == 0 || data.agents[agent_id].type == Character::Type::Astral) {
                copyAllocHostToDevice<bool>(&boundaries_masks[agent_id], data.boundaries_size);
            }
        }
        copyAllocHostToDevice<bool*>(&data.boundaries_masks, data.agents_size);
        free(boundaries_masks);

        AgentData* agents = copyAllocHostToHost<AgentData>(&data.agents, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            switch (agents[agent_id].type) {
            case Character::Type::Player: {
                Player* agent = copyAllocHostToHost<Player>((Player**) &agents[agent_id].agent, 1);
                copyAllocHostToDevice<half>(&agent->data, Player::data_size);
                copyAllocHostToDevice<Player>((Player**) &agents[agent_id].agent, 1);
                free(agent);
                break;
            }
            case Character::Type::Astral: {
                Astral* agent = copyAllocHostToHost<Astral>((Astral**) &agents[agent_id].agent, 1);
                copyAllocHostToDevice<half>(&agent->data, Astral::data_size);
                copyAllocHostToDevice<Astral>((Astral**) &agents[agent_id].agent, 1);
                free(agent);
                break;
            }
            case Character::Type::Cultist: {
                Cultist* agent = copyAllocHostToHost<Cultist>((Cultist**) &agents[agent_id].agent, 1);
                copyAllocHostToDevice<half>(&agent->data, Cultist::data_size);
                copyAllocHostToDevice<Cultist>((Cultist**) &agents[agent_id].agent, 1);
                free(agent);
                break;
            }
            }
        }
        copyAllocHostToDevice<AgentData>(&data.agents, data.agents_size);
        free(agents);

        copyAllocHostToDevice<float2_pair>(&data.boundaries, data.boundaries_size);
        copyAllocHostToDevice<uint>(&data.boundaries_map, data.height * data.width);
        
        InteractableData* interactables = copyAllocHostToHost<InteractableData>(&data.interactables, data.interactables_size);
        for (uint id = 0; id < data.interactables_size; ++id) {
            switch (interactables[id].type) {
            case ObjectType::Door:
                copyAllocHostToDevice<Door>((Door**) &interactables[id].object, 1);
                break;
            case ObjectType::Window:
                copyAllocHostToDevice<Window>((Window**) &interactables[id].object, 1);
            }
        }
        copyAllocHostToDevice<InteractableData>(&data.interactables, data.interactables_size);

        copyAllocHostToDevice<bool>(&data.interactables_mask, data.interactables_size);
        copyAllocHostToDevice<uint>(&data.interactables_map, data.height * data.width);
        copyAllocHostToDevice<ObjectType>(&data.object_map, data.height * data.width);
        copyAllocHostToDevice<LightingType>(&data.lighting_map.normal, data.height * data.width);
        copyAllocHostToDevice<LightingType>(&data.lighting_map.madness, data.height * data.width);
        copyAllocHostToDevice<LightingType>(&data.lighting_map.dynamic, data.height * data.width);

        cudaMalloc(&data.seed, sizeof(curandState));
        curandInitUpstream<<<1, 1>>>(data.seed);

        Data* data_dest;
        cudaMalloc(&data_dest, sizeof(Data));
        cudaMemcpy(data_dest, &data, sizeof(Data), cudaMemcpyHostToDevice);

        return data_dest;
    }

    __host__ Data* copyDeviceToHost(Data* data_orig) {

        Data data;
        cudaMemcpy(&data, data_orig, sizeof(Data), cudaMemcpyDeviceToHost);

        copyAllocDeviceToHost<half>(&data.player_tensor.data, 5 * data.height * data.width);
        copyAllocDeviceToHost<half>(&data.monsters_tensor.data, 5 * data.height * data.width);

        copyAllocDeviceToHost<half>(&data.player_hearing_map, data.height * data.width);
        copyAllocDeviceToHost<half>(&data.monsters_hearing_map, data.height * data.width);

        copyAllocDeviceToHost<AgentData>(&data.agents, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            switch (data.agents[agent_id].type) {
            case Character::Type::Player:
                copyAllocDeviceToHost<Player>((Player**) &data.agents[agent_id].agent, 1);
                copyAllocDeviceToHost<half>(&data.agents[agent_id].agent->data, Player::data_size);
                break;
            case Character::Type::Astral:
                copyAllocDeviceToHost<Astral>((Astral**) &data.agents[agent_id].agent, 1);
                copyAllocDeviceToHost<half>(&data.agents[agent_id].agent->data, Astral::data_size);
                break;
            case Character::Type::Cultist:
                copyAllocDeviceToHost<Cultist>((Cultist**) &data.agents[agent_id].agent, 1);
                copyAllocDeviceToHost<half>(&data.agents[agent_id].agent->data, Cultist::data_size);
            }
        }
        
        copyAllocDeviceToHost<bool*>(&data.sight_masks, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            if (agent_id == 0 || data.agents[agent_id].type == Character::Type::Astral) {
                copyAllocDeviceToHost<bool>(&data.sight_masks[agent_id], data.height * data.width);
            }
        }

        copyAllocDeviceToHost<bool*>(&data.boundaries_masks, data.agents_size);
        for (uint agent_id = 0; agent_id < data.agents_size; ++agent_id) {
            if (agent_id == 0 || data.agents[agent_id].type == Character::Type::Astral) {
                copyAllocDeviceToHost<bool>(&data.boundaries_masks[agent_id], data.boundaries_size);
            }
        }

        copyAllocDeviceToHost<float2_pair>(&data.boundaries, data.boundaries_size);
        copyAllocDeviceToHost<uint>(&data.boundaries_map, data.height * data.width);
        
        copyAllocDeviceToHost<InteractableData>(&data.interactables, data.interactables_size);
        for (uint id = 0; id < data.interactables_size; ++id) {
            switch (data.interactables[id].type) {
            case ObjectType::Door:
                copyAllocDeviceToHost<Door>((Door**) &data.interactables[id].object, 1);
                break;
            case ObjectType::Window:
                copyAllocDeviceToHost<Window>((Window**) &data.interactables[id].object, 1);
            }
        }

        copyAllocDeviceToHost<bool>(&data.interactables_mask, data.interactables_size);
        copyAllocDeviceToHost<uint>(&data.interactables_map, data.height * data.width);
        copyAllocDeviceToHost<ObjectType>(&data.object_map, data.height * data.width);
        copyAllocDeviceToHost<LightingType>(&data.lighting_map.normal, data.height * data.width);
        copyAllocDeviceToHost<LightingType>(&data.lighting_map.madness, data.height * data.width);
        copyAllocDeviceToHost<LightingType>(&data.lighting_map.dynamic, data.height * data.width);

        Data* data_dest = (Data*) malloc(sizeof(Data));
        *data_dest = data;

        return data_dest;
    }

    __host__ void deleteHost(Data* data) {
        free(data->player_tensor.data);
        free(data->monsters_tensor.data);
        free(data->player_hearing_map);
        free(data->monsters_hearing_map);
        for (uint id = 0; id < data->agents_size; ++id) {
            if (id == 0 || data->agents[id].type == Character::Type::Astral) {
                free(data->boundaries_masks[id]);
                free(data->sight_masks[id]);
            }
        }
        free(data->boundaries_masks);
        free(data->sight_masks);
        for (uint id = 0; id < data->agents_size; ++id) {
            free(data->agents[id].agent->data);
            free(data->agents[id].agent);
        }
        free(data->agents);
        free(data->boundaries);
        free(data->boundaries_map);
        for (uint id = 0; id < data->interactables_size; ++id) {
            free(data->interactables[id].object);
        }
        free(data->interactables);
        free(data->interactables_mask);
        free(data->interactables_map);
        free(data->object_map);
        free(data->lighting_map.normal);
        free(data->lighting_map.madness);
        free(data->lighting_map.dynamic);
        free(data);
    }

    __global__ void copyDeviceToDevice(Data* data_orig, Data* data_dest) {
        memcpy(data_dest->player_tensor.data, data_orig->player_tensor.data, 5 * data_orig->height * data_orig->width * sizeof(half));
        memcpy(data_dest->monsters_tensor.data, data_orig->monsters_tensor.data, 5 * data_orig->height * data_orig->width * sizeof(half));
        memcpy(data_dest->player_hearing_map, data_orig->player_hearing_map, data_orig->height * data_orig->width * sizeof(half));
        memcpy(data_dest->monsters_hearing_map, data_orig->monsters_hearing_map, data_orig->height * data_orig->width * sizeof(half));
        for (uint agent_id = 0; agent_id < data_orig->agents_size; ++agent_id) {
            Agent* agent_orig = data_orig->agents[agent_id].agent;
            Agent* agent_dest = data_dest->agents[agent_id].agent;
            half* data = agent_dest->data;
            switch (data_orig->agents[agent_id].type) {
            case Character::Type::Player:
                memcpy(agent_dest->data, agent_orig->data, Player::data_size * sizeof(half));
                memcpy(agent_dest, agent_orig, sizeof(Player));
                break;
            case Character::Type::Astral:
                memcpy(agent_dest->data, agent_orig->data, Astral::data_size * sizeof(half));
                memcpy(agent_dest, agent_orig, sizeof(Astral));
                break;
            case Character::Type::Cultist:
                memcpy(agent_dest->data, agent_orig->data, Cultist::data_size * sizeof(half));
                memcpy(agent_dest, agent_orig, sizeof(Cultist));
            }
            agent_dest->data = data;
        }
        data_dest->health = data_orig->health;
        data_dest->madness = data_orig->madness;
        data_dest->madness_world = data_dest->madness_world;
        memcpy(data_dest->boundaries, data_orig->boundaries, data_orig->boundaries_size * sizeof(uint2_pair));
        memcpy(data_dest->boundaries_map, data_orig->boundaries_map, data_orig->height * data_orig->width * sizeof(uint));
        for (uint agent_id = 0; agent_id < data_orig->agents_size; ++agent_id) {
            if (agent_id == 0 || data_orig->agents[agent_id].type == Character::Type::Astral){
                memcpy(data_dest->boundaries_masks[agent_id], data_orig->boundaries_masks[agent_id], data_orig->boundaries_size * sizeof(bool));
                memcpy(data_dest->sight_masks[agent_id], data_orig->sight_masks[agent_id], data_orig->height * data_orig->width * sizeof(bool));
            }
        }
        for (uint id = 0; id < data_orig->interactables_size; ++id) {
            switch (data_orig->interactables[id].type) {
            case ObjectType::Door:
                memcpy(data_dest->interactables[id].object, data_orig->interactables[id].object, sizeof(Door));
                break;
            case ObjectType::Window:
                memcpy(data_dest->interactables[id].object, data_orig->interactables[id].object, sizeof(Window));
            }
        }
        memcpy(data_dest->interactables_mask, data_orig->interactables_mask, data_orig->interactables_size * sizeof(bool));
        memcpy(data_dest->interactables_map, data_orig->interactables_map, data_orig->height * data_orig->width * sizeof(uint));
        memcpy(data_dest->object_map, data_orig->object_map, data_orig->height * data_orig->width * sizeof(ObjectType));
        memcpy(data_dest->lighting_map.normal, data_orig->lighting_map.normal, data_orig->height * data_orig->width * sizeof(LightingType));
        memcpy(data_dest->lighting_map.madness, data_orig->lighting_map.madness, data_orig->height * data_orig->width * sizeof(LightingType));
        memcpy(data_dest->lighting_map.dynamic, data_orig->lighting_map.dynamic, data_orig->height * data_orig->width * sizeof(LightingType));
    }

    __host__ void deleteDevice(Data* data) {
        Data data_host;
        cudaMemcpy(&data_host, data, sizeof(Data), cudaMemcpyDeviceToHost);

        cudaFree(data_host.player_tensor.data);
        cudaFree(data_host.monsters_tensor.data);
        cudaFree(data_host.player_hearing_map);
        cudaFree(data_host.monsters_hearing_map);
        cudaFree(data_host.boundaries);
        cudaFree(data_host.boundaries_map);
        InteractableData* interactables = (InteractableData*) malloc(data_host.interactables_size * sizeof(InteractableData));
        cudaMemcpy(interactables, data_host.interactables, data_host.interactables_size * sizeof(InteractableData), cudaMemcpyDeviceToHost);
        for (uint id = 0; id < data_host.interactables_size; ++id) {
            cudaFree(interactables[id].object);
        }
        free(interactables);
        cudaFree(data_host.interactables);
        cudaFree(data_host.interactables_mask);
        cudaFree(data_host.interactables_map);
        cudaFree(data_host.object_map);
        cudaFree(data_host.lighting_map.normal);
        cudaFree(data_host.lighting_map.madness);
        cudaFree(data_host.lighting_map.dynamic);

        bool** boundaries_masks = (bool**) malloc(data_host.agents_size * sizeof(bool*));
        cudaMemcpy(boundaries_masks, data_host.boundaries_masks, data_host.agents_size * sizeof(bool*), cudaMemcpyDeviceToHost);
        bool** sight_masks = (bool**) malloc(data_host.agents_size * sizeof(bool*));
        cudaMemcpy(sight_masks, data_host.sight_masks, data_host.agents_size * sizeof(bool*), cudaMemcpyDeviceToHost);
        AgentData* agents = (AgentData*) malloc(data_host.agents_size * sizeof(AgentData));
        cudaMemcpy(agents, data_host.agents, data_host.agents_size * sizeof(AgentData), cudaMemcpyDeviceToHost);
        for (uint id = 0; id < data_host.agents_size; ++id) {
            Agent agent;
            cudaMemcpy(&agent, agents[id].agent, sizeof(Agent), cudaMemcpyDeviceToHost);
            cudaFree(agent.data);
            if (id == 0 || agents[id].type == Character::Type::Astral) {
                cudaFree(boundaries_masks[id]);
                cudaFree(sight_masks[id]);
            }
            cudaFree(agents[id].agent);
        }
        cudaFree(data_host.boundaries_masks);
        free(boundaries_masks);
        cudaFree(data_host.sight_masks);
        free(sight_masks);
        cudaFree(data_host.agents);
        free(agents);

        cudaFree(data_host.seed);

        cudaFree(data);
    }

    __host__ __device__ __inline__ void updateMadness(Data* data) {
        for (uint id = 0; id < data->agents_size; ++id) {
            data->agents[id].agent->updateMadness(data->madness);
        }
    }

    __host__ __device__ __inline__ void updateHealth(Data* data) {
        for (uint id = 0; id < data->agents_size; ++id) {
            data->agents[id].agent->updateHealth(data->health);
        }
    }

    __device__ __forceinline__ void increaseMadness(Data* data,  float madness) {
        data->madness += madness;
        data->health -= max(data->madness - 100.f, 0.f);
        data->madness = min(data->madness, 100.f);
        data->madness = max(data->madness, 0.f);
    }

    __device__ __inline__ bool monsterUpdatePosition(Data* data, Agent* agent, ObjectType type, float2 coords_dest) {
        uint2 coords_round = {(uint) floor(agent->coords.x) - 1, (uint) floor(agent->coords.y) - 1};
        uint2 coords_dest_round = {(uint) floor(coords_dest.x) - 1, (uint) floor(coords_dest.y) - 1};

        if (coords_dest_round.x == coords_round.x && coords_dest_round.y == coords_round.y) {
            agent->coords = coords_dest;
            return true;
        }

        ObjectType type_prev = data->object_map[coords_round.y * data->width + coords_round.x];

        for (uint y = 0; y < 3; ++y) {
            for (uint x = 0; x < 3; ++x) {
                uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                data->object_map[ind] = ObjectType::Clear;
                *data->monsters_tensor.getObject(ind) = Utils::getValue<ObjectType>(ObjectType::Clear);
            }
        }

        bool success = true;
        for (uint y = 0; y < 3; ++y) {
            for (uint x = 0; x < 3; ++x) {
                uint ind = (coords_dest_round.y + y) * data->width + (coords_dest_round.x + x);
                if (data->object_map[ind] != ObjectType::Clear) {
                    success = false;
                    goto for_end;
                }
            }
        }
        for_end:

        if (success) {
            for (uint y = 0; y < 3; ++y) {
                for (uint x = 0; x < 3; ++x) {
                    uint ind = (coords_dest_round.y + y) * data->width + (coords_dest_round.x + x);
                    data->object_map[ind] = type;
                    *data->monsters_tensor.getObject(ind) = Utils::getValue<ObjectType>(type);
                }
            }
            agent->coords = coords_dest;
        } else {
            for (uint y = 0; y < 3; ++y) {
                for (uint x = 0; x < 3; ++x) {
                    uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                    data->object_map[ind] = type_prev;
                    *data->monsters_tensor.getObject(ind) = Utils::getValue<ObjectType>(type_prev);
                }
            }
        }
        return success;
    }

    __device__ __forceinline__ void astralMakeSound(Data* data, Astral* astral) {
        Player* player = (Player*) data->agents[0].agent;
        float2 delta = {astral->coords.x - player->coords.x, astral->coords.y - player->coords.y};
        float dist = sqrt(delta.x * delta.x + delta.y * delta.y);
        if (dist > astral->sound_radius) {
            return;
        }
        float2 delta_rand = curand_normal2(data->seed);
        delta_rand = {delta_rand.x * dist / 12, delta_rand.y * dist / 12};
        delta_rand = {min(dist/6, max(-dist/6, delta_rand.x)), min(dist/6, max(-dist/6, delta_rand.y))};
        float2 sound_coords = {astral->coords.x + delta_rand.x, astral->coords.y + delta_rand.y};
        uint2 sound_coords_round;
        sound_coords_round.x = min(data->width, max(0, (uint) floor(sound_coords.x)));
        sound_coords_round.y = min(data->height, max(0, (uint) floor(sound_coords.y)));
        uint sound_ind = sound_coords_round.y * data->width + sound_coords_round.x;
        *data->player_tensor.getHearing(sound_ind) = 1.f;
        if (astral->ability_cooldown < 8) {
            *data->player_tensor.getObject(sound_ind) = Utils::getValue<ObjectType>(ObjectType::Astral);
        } else {
            *data->player_tensor.getObject(sound_ind) = Utils::getValue<ObjectType>(ObjectType::AstralRush);
        }
    }

    __device__ __forceinline__ bool astralGo(Data* data, Astral* astral) {
        float2 coords_dest = {
            astral->coords.x + astral->speed * cospif(astral->rotation),
            astral->coords.y + astral->speed * sinpif(astral->rotation)
        };
        bool success;
        if (astral->ability_cooldown < 8) {
            success = monsterUpdatePosition(data, (Agent*) astral, ObjectType::Astral, coords_dest);
        } else {
            success = monsterUpdatePosition(data, (Agent*) astral, ObjectType::AstralRush, coords_dest);
        }
        return success;
    }

    __device__ __forceinline__ bool doActionAstral(Data* data, Astral* astral) {

        if (astral->see_player && astral->scream_cooldown == 0) {   
            astral->scream_cooldown = 10;
        }

        if (astral->attack_player && astral->cooldown == 0) {
            astral->cooldown = 2;
            data->health -= 10.f;
        }

        bool success = false;
        if (astral->cooldown > 0) {
            success = astral->action == Astral::Action::Nothing;
        } else {
            switch (astral->action) {
            case Astral::Action::Nothing:
                success = true;
                break;
            case Astral::Action::GoForward:
                success = astralGo(data, astral);
                break;
            case Astral::Action::TurnRightFull:
                astral->rotation = fmodf(astral->rotation - Astral::rotation_speed, 2.f);
                if (astral->rotation < 0.f) {
                    astral->rotation += 2.f;
                }
                success = true;
                break;
            case Astral::Action::TurnRightHalf:
                astral->rotation = fmodf(astral->rotation - Astral::rotation_speed / 2, 2.f);
                if (astral->rotation < 0.f) {
                    astral->rotation += 2.f;
                }
                success = true;
                break;
            case Astral::Action::TurnLeftFull:
                astral->rotation = fmodf(astral->rotation + Astral::rotation_speed, 2.f);
                if (astral->rotation < 0.f) {
                    astral->rotation += 2.f;
                }
                success = true;
                break;
            case Astral::Action::TurnLeftHalf:
                astral->rotation = fmodf(astral->rotation + Astral::rotation_speed / 2, 2.f);
                if (astral->rotation < 0.f) {
                    astral->rotation += 2.f;
                }
                success = true;
                break;
            case Astral::Action::Ability:
                if (astral->ability_cooldown > 0) {
                    success = false;
                    break;
                }
                astral->ability_cooldown = 16;
                astral->speed = 1.f;
                monsterUpdatePosition(data, (Agent*) astral, ObjectType::AstralRush, astral->coords);
                success = true;
                break;
            }
        }


        if (astral->cooldown > 0) {
            --astral->cooldown;
        }
        if (astral->ability_cooldown > 0) {
            --astral->ability_cooldown;
        }
        if (astral->scream_cooldown > 0) {
            --astral->scream_cooldown;
        }

        if (astral->ability_cooldown == 8) {
            astral->speed = 0.375f;
            monsterUpdatePosition(data, (Agent*) astral, ObjectType::Astral, astral->coords);
        }

        if (astral->scream_cooldown == 9) {
            astral->sound_radius = 150.f;
        } else if (astral->action == Astral::Action::GoForward && success) {
            astral->sound_radius = 65.f;
        } else {
            astral->sound_radius = 0.f;
        }

        if (astral->sound_radius > 0.f) {
            astralMakeSound(data, astral);
        }

        return success;
    }

    __device__ __forceinline__ void updateUseAstral(Data* data, Astral* astral) {
        astral->see_player = false;
        Player* player = (Player*) data->agents[0].agent;
        float2 delta = {player->coords.x - astral->coords.x, player->coords.y - astral->coords.y};
        float delta_rot = atanf(delta.y / delta.x) / M_PI;
        if (delta.x < 0.f) {
            delta_rot = delta_rot + 1.f;
        }
        delta_rot = fmodf(delta_rot - astral->rotation, 2.f);
        if (delta_rot < 0.f) {
            delta_rot += 2.f;
        }
        astral->attack_player = (delta.x * delta.x + delta.y * delta.y <= 4 * 4) && (delta_rot < 1./3 || delta_rot > 2. - 1./3);
    }

    __device__ __forceinline__ void cultistMakeSound(Data* data, Cultist* cultist) {
        Player* player = (Player*) data->agents[0].agent;
        float2 delta = {cultist->coords.x - player->coords.x, cultist->coords.y - player->coords.y};
        float dist = sqrt(delta.x * delta.x + delta.y * delta.y);
        if (dist > cultist->sound_radius) {
            return;
        }
        float2 delta_rand = curand_normal2(data->seed);
        delta_rand = {delta_rand.x * dist / 12, delta_rand.y * dist / 12};
        delta_rand = {min(dist/6, max(-dist/6, delta_rand.x)), min(dist/6, max(-dist/6, delta_rand.y))};
        float2 sound_coords = {cultist->coords.x + delta_rand.x, cultist->coords.y + delta_rand.y};
        uint2 sound_coords_round;
        sound_coords_round.x = min(data->width, max(0, (uint) floor(sound_coords.x)));
        sound_coords_round.y = min(data->height, max(0, (uint) floor(sound_coords.y)));
        uint sound_ind = sound_coords_round.y * data->width + sound_coords_round.x;
        *data->player_tensor.getHearing(sound_ind) = 1.f;
        if (cultist->cooldown == 0) {
            *data->player_tensor.getObject(sound_ind) = Utils::getValue<ObjectType>(ObjectType::Cultist);
        } else {
            *data->player_tensor.getObject(sound_ind) = Utils::getValue<ObjectType>(ObjectType::CultistChant);
        }
    }

    __device__ __forceinline__ bool cultistGo(Data* data, Cultist* cultist) {
        float2 coords_dest = {
            cultist->coords.x + Cultist::speed * cospif(cultist->rotation),
            cultist->coords.y + Cultist::speed * sinpif(cultist->rotation)
        };
        return monsterUpdatePosition(data, (Agent*) cultist, ObjectType::Cultist, coords_dest);
    }

    __device__ __inline__ void updateAbilityLight(Data* data, Cultist* cultist) {
        if (cultist->ability_cooldown == 0) {
            return;
        }
        int2 coords_round = {(int) floor(cultist->ability_coords.x), (int) floor(cultist->ability_coords.y)};
        for (uint y = max(coords_round.y - 20, 0); y <= min(coords_round.y + 20, data->height - 1); ++y) {
            for (uint x = max(coords_round.x - 20, 0); x <= min(coords_round.x + 20, data->width - 1); ++x) {
                float2 delta = {x - cultist->ability_coords.x, y - cultist->ability_coords.y};
                if (delta.x * delta.x + delta.y * delta.y <= 20.f * 20.f) {
                    data->lighting_map.dynamic[y * data->width + x] = LightingType::Madness;
                }
            }
        }
    }

    __device__ __forceinline__ bool cultistAbility(Data* data, Cultist* cultist) {
        if (cultist->ability_cooldown > 0 || cultist->noise < 1.f) {
            return false;
        }
        cultist->ability_cooldown = 11;
        cultist->ability_coords = data->agents[0].agent->coords;
        return true;
    }

    __device__ __forceinline__ bool doActionCultist(Data* data, Cultist* cultist) {
        bool success = false;
        if (cultist->noise == 1.f && cultist->cooldown == 0) {
            increaseMadness(data, 10.f);
            cultist->cooldown = 6;
        }

        if (cultist->cooldown > 0) {
            if (cultist->action == Cultist::Action::Nothing) {
                success = true;
            } else if (cultist->action == Cultist::Action::Ability) {
                success = cultistAbility(data, cultist);
            } else {
                success = false;
            }
        } else {
            switch (cultist->action) {
            case Cultist::Action::Nothing:
                success = true;
                break;
            case Cultist::Action::GoForward:
                success = cultistGo(data, cultist);
                break;
            case Cultist::Action::TurnRightFull:
                cultist->rotation = fmodf(cultist->rotation - Cultist::rotation_speed, 2.f);
                if (cultist->rotation < 0.f) {
                    cultist->rotation += 2.f;
                }
                success = true;
                break;
            case Cultist::Action::TurnRightHalf:
                cultist->rotation = fmodf(cultist->rotation - Cultist::rotation_speed / 2, 2.f);
                if (cultist->rotation < 0.f) {
                    cultist->rotation += 2.f;
                }
                success = true;
                break;
            case Cultist::Action::TurnLeftFull:
                cultist->rotation = fmodf(cultist->rotation + Cultist::rotation_speed, 2.f);
                if (cultist->rotation < 0.f) {
                    cultist->rotation += 2.f;
                }
                success = true;
                break;
            case Cultist::Action::TurnLeftHalf:
                cultist->rotation = fmodf(cultist->rotation + Cultist::rotation_speed / 2, 2.f);
                if (cultist->rotation < 0.f) {
                    cultist->rotation += 2.f;
                }
                success = true;
                break;
            case Cultist::Action::Ability:
                success = cultistAbility(data, cultist);
                break;
            }
        }

        ObjectType type = data->object_map[(uint) floor(cultist->coords.y) * data->width + (uint) floor(cultist->coords.x)];
        if (cultist->cooldown > 0 && type == ObjectType::Cultist) {
            monsterUpdatePosition(data, (Agent*) cultist, ObjectType::CultistChant, cultist->coords);
        } else if (cultist->cooldown == 0 && type == ObjectType::CultistChant) {
            monsterUpdatePosition(data, (Agent*) cultist, ObjectType::Cultist, cultist->coords);
        }

        if (cultist->cooldown > 0) {
            --cultist->cooldown;
        }
        if (cultist->ability_cooldown > 0) {
            --cultist->ability_cooldown;
            if (cultist->ability_cooldown == 0) {
                cultist->ability_coords = {0.f, 0.f};
            }
        }

        if (cultist->action == Cultist::Action::GoForward && success) {
            cultist->sound_radius = 65.f;
        } else {
            cultist->sound_radius = 0.f;
        }

        if (cultist->sound_radius > 0.f) {
            cultistMakeSound(data, cultist);
        }

        return success;
    }

    __device__ bool playerUpdatePosition(Data* data, Player* player, float2 coords_dest) {
        uint2 coords_round = {(uint) floor(player->coords.x - 0.5f), (uint) floor(player->coords.y - 0.5f)};
        uint2 coords_dest_round = {(uint) floor(coords_dest.x - 0.5f), (uint) floor(coords_dest.y - 0.5f)};

        bool success = true;
        bool no_monsters = true;
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                ObjectType type = data->object_map[(coords_dest_round.y + y) * data->width + (coords_dest_round.x + x)];
                if (
                    type == ObjectType::Astral ||
                    type == ObjectType::AstralRush ||
                    type == ObjectType::Cultist ||
                    type == ObjectType::CultistChant
                ) {
                    no_monsters = false;
                } else if (!(
                    type == ObjectType::Clear ||
                    type == ObjectType::Player
                )) {
                    success = false;
                    goto for_end;
                }
            }
        }
        for_end:

        if (!success) {
            return false;
        }

        if (no_monsters) {
            for (uint y = 0; y < 2; ++y) {
                for (uint x = 0; x < 2; ++x) {
                    uint ind = (player->shadow_coords.y + y) * data->width + (player->shadow_coords.x + x);
                    data->object_map[ind] = ObjectType::Clear;
                    *data->player_tensor.getObject(ind) = Utils::getValue<ObjectType>(ObjectType::Clear);
                }
            }
            for (uint y = 0; y < 2; ++y) {
                for (uint x = 0; x < 2; ++x) {
                    uint ind = (coords_dest_round.y + y) * data->width + (coords_dest_round.x + x);
                    data->object_map[ind] = ObjectType::Player;
                    *data->player_tensor.getObject(ind) = Utils::getValue<ObjectType>(ObjectType::Player);
                }
            }
            player->shadow_coords = coords_dest_round;
        }

        player->coords = coords_dest;

        return true;
    }

    __device__ __inline__ void playerMakeSound(Data* data, Player* player) {
        for (uint agent_id = 1; agent_id < data->agents_size; ++agent_id) {
            if (data->agents[agent_id].type == Character::Type::Cultist) {
                Cultist* cultist = (Cultist*) data->agents[agent_id].agent;
                float2 delta = {player->coords.x - cultist->coords.x, player->coords.y - cultist->coords.y};
                float dist = sqrt(delta.x * delta.x + delta.y * delta.y);
                if (dist <= player->sound_radius) {
                    uint sound_ind = floor(player->coords.y) * data->width + floor(player->coords.x);
                    *data->monsters_tensor.getHearing(sound_ind) = 1.f;
                    *data->monsters_tensor.getObject(sound_ind) = Utils::getValue<ObjectType>(ObjectType::Player);
                    cultist->noise = min(cultist->noise + 0.3f, 1.f);
                } else {
                    cultist->noise = max(cultist->noise - 0.15f, 0.f);
                }
            }
        }
    }

    __device__ bool playerGo(Data* data, Player* player, float rotation) {
        float2 coords_dest = {
            player->coords.x + player->speed * cospif(player->rotation + rotation),
            player->coords.y + player->speed * sinpif(player->rotation + rotation)
        };
        bool success = playerUpdatePosition(data, player, coords_dest);
        if (!success) {
            return false;
        }
        
        if (player->movement_state == Player::MovementState::Run) {
            player->stamina = max(player->stamina - 2.f, 0.f);
            if (player->stamina == 0.f) {
                player->movement_state = Player::MovementState::Walk;
                player->speed = 0.625f;
            }
        } else {
            player->stamina = min(player->stamina + 2.f, 100.f);
        }

        return true;
    }

    __device__ __inline__ bool doActionPlayer(Data* data, Player* player) {
        bool success = false;
        if (player->cooldown > 0) {
            success = (player->action == Player::Action::Nothing);
        } else {
            switch (player->action) {
            case Player::Action::Nothing:
                success = true;
                break;
            case Player::Action::GoForward: {
                success = playerGo(data, player, 0.f);
                break;
            }
            case Player::Action::GoLeft: {
                success = playerGo(data, player, 0.5f);
                break;
            }
            case Player::Action::GoBack: {
                success = playerGo(data, player, 1.f);
                break;
            }
            case Player::Action::GoRight: {
                success = playerGo(data, player, -0.5f);
                break;
            }
            case Player::Action::TurnRightFull: {
                player->rotation = fmodf(player->rotation - Player::rotation_speed, 2.f);
                if (player->rotation < 0.f) {
                    player->rotation += 2.f;
                }
                success = true;
                break;
            }
            case Player::Action::TurnRightHalf: {
                player->rotation = fmodf(player->rotation - Player::rotation_speed / 2, 2.f);
                if (player->rotation < 0.f) {
                    player->rotation += 2.f;
                }
                success = true;
                break;
            }
            case Player::Action::TurnLeftFull: {
                player->rotation = fmodf(player->rotation + Player::rotation_speed, 2.f);
                if (player->rotation < 0.f) {
                    player->rotation += 2.f;
                }
                success = true;
                break;
            }
            case Player::Action::TurnLeftHalf: {
                player->rotation = fmodf(player->rotation + Player::rotation_speed / 2, 2.f);
                if (player->rotation < 0.f) {
                    player->rotation += 2.f;
                }
                success = true;
                break;
            }
            case Player::Action::UseWindow: {
                if (player->window == (uint) -1) {
                    break;
                }
                Window* window = (Window*) data->interactables[player->window].object;
                float2 delta_in = {player->coords.x - window->coords_in.x, player->coords.y - window->coords_in.y};
                float dist_in = delta_in.x * delta_in.x + delta_in.y * delta_in.y;
                float2 delta_out = {player->coords.x - window->coords_out.x, player->coords.y - window->coords_out.y};
                float dist_out = delta_out.x * delta_out.x + delta_out.y * delta_out.y;
                // all windows must face clear area
                playerUpdatePosition(data, player, dist_in <= dist_out ? window->coords_out : window->coords_in);
                player->cooldown = 3;
                success = true;
                break;
            }
            case Player::Action::UseDoor: {
                if (player->door == (uint) -1) {
                    break;
                }
                Door* door = (Door*) data->interactables[player->door].object;
                if (door->is_open) {
                    door->is_open = false;
                    for (uint i = 1; i <= 2; ++i) {
                        uint ind = door->coords_open[i].y * data->width + door->coords_open[i].x;
                        data->object_map[ind] = ObjectType::Clear;
                    }
                    for (uint i = 1; i <= 2; ++i) {
                        uint ind = door->coords_closed[i].y * data->width + door->coords_closed[i].x;
                        if (data->object_map[ind] == ObjectType::Player) {
                            playerUpdatePosition(data, player, door->coords_in);
                            player->cooldown = 3;
                        }
                        data->object_map[ind] = ObjectType::Door;
                    }
                    float2_pair* boundary = &data->boundaries[door->boundary_id];
                    boundary->first.x = door->coords_closed[0].x + 0.5f;
                    boundary->first.y = door->coords_closed[0].y + 0.5f;
                    boundary->second.x = door->coords_closed[3].x + 0.5f;
                    boundary->second.y = door->coords_closed[3].y + 0.5f;
                } else {
                    door->is_open = true;
                    for (uint i = 1; i <= 2; ++i) {
                        data->object_map[door->coords_closed[i].y * data->width + door->coords_closed[i].x] = ObjectType::Clear;
                    }
                    for (uint i = 1; i <= 2; ++i) {
                        uint ind = door->coords_open[i].y * data->width + door->coords_open[i].x;
                        if (data->object_map[ind] == ObjectType::Player) {
                            playerUpdatePosition(data, player, door->coords_in);
                            player->cooldown = 3;
                        }
                        data->object_map[ind] = ObjectType::Door;
                    }
                    float2_pair* boundary = &data->boundaries[door->boundary_id];
                    boundary->first.x = door->coords_open[0].x + 0.5f;
                    boundary->first.y = door->coords_open[0].y + 0.5f;
                    boundary->second.x = door->coords_open[2].x + 0.5f;
                    boundary->second.y = door->coords_open[2].y + 0.5f;
                }
                success = true;
                break;
            }
            case Player::Action::MagicSee: {
                if (player->magic_cooldown_see > 0) {
                    break;
                }
                player->magic_cooldown_see = 40;
                increaseMadness(data, 10.f);
                player->sight_radius = 300.f;
                success = true;
                break;
            }
            case Player::Action::MagicInvisibility: {
                if (player->magic_cooldown_invisibility > 0) {
                    break;
                }
                player->magic_cooldown_invisibility = 60;
                increaseMadness(data, 15.f);
                player->sight_radius = 5.f;
                success = true;
                break;
            }
            case Player::Action::MagicInaudibility: {
                if (player->magic_cooldown_inaudibility > 0) {
                    break;
                }
                player->magic_cooldown_inaudibility = 60;
                increaseMadness(data, 15.f);
                player->sound_radius = 0.f;
                player->hearing_radius = 0.f;
                success = true;
                break;
            }
            case Player::Action::ChangeMovementRun: {
                switch (player->movement_state) {
                case Player::MovementState::Walk:
                    player->movement_state = Player::MovementState::Run;
                    player->speed = 1.f;
                    success = true;
                    break;
                case Player::MovementState::Run:
                    player->movement_state = Player::MovementState::Walk;
                    player->speed = 0.625f;
                    success = true;
                    break;
                }
                break;
            }
            case Player::Action::ChangeMovementCrouch: {
                switch (player->movement_state) {
                case Player::MovementState::Crouch:
                    player->movement_state = Player::MovementState::Walk;
                    player->speed = 0.625f;
                    success = true;
                    break;
                case Player::MovementState::Walk:
                    player->movement_state = Player::MovementState::Crouch;
                    player->speed = 0.375f;
                    success = true;
                    break;
                }
                break;
            }
            }
        }

        data->madness_world = (data->madness >= 80.f && !data->madness_world) || (data->madness > 20.f && data->madness_world);

        if (player->cooldown > 0) {
            --player->cooldown;
        }
        if (player->magic_cooldown_see > 0) {
            --player->magic_cooldown_see;
        }
        if (player->magic_cooldown_invisibility > 0) {
            --player->magic_cooldown_invisibility;
        }
        if (player->magic_cooldown_inaudibility > 0) {
            --player->magic_cooldown_inaudibility;
            if (player->magic_cooldown_inaudibility == 0) {
                player->hearing_radius = 65.f;
            }
        }

        if ((
            player->action == Player::Action::GoForward ||
            player->action == Player::Action::GoLeft ||
            player->action == Player::Action::GoBack ||
            player->action == Player::Action::GoRight ||
            player->action == Player::Action::UseWindow
        ) && success && player->magic_cooldown_inaudibility < 46) {
            switch (player->movement_state) {
            case Player::MovementState::Crouch:
                player->sound_radius = 2.5f;
                break;
            case Player::MovementState::Walk:
                player->sound_radius = 25.f;
                break;
            case Player::MovementState::Run:
                player->sound_radius = 50.f;
                break;
            }
        } else {
            player->sound_radius = 0.f;
        }

        if (player->sound_radius > 0.f) {
            playerMakeSound(data, player);
        } else {
            for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
                auto [type, agent] = data->agents[agent_id];
                if (type == Character::Type::Cultist) {
                    Cultist* cultist = (Cultist*) agent;
                    cultist->noise = max(cultist->noise - 0.15f, 0.f);
                }
            }
        }

        if (player->magic_cooldown_see >= 38 || player->magic_cooldown_see < 28 && player->magic_cooldown_invisibility < 46) {
            if (!data->madness_world) {
                if (data->madness <= 30.f) {
                    player->sight_radius = 300.f;
                } else if (data->madness < 59.7f) {
                    player->sight_radius = 10.f * (60.f - data->madness);
                } else {
                    player->sight_radius = 3.f;
                }
            } else {
                player->sight_radius = 300.f;
            }
        }

        uint2 coords_round = {(uint) floor(player->coords.x - 0.5f), (uint) floor(player->coords.y - 0.5f)};
        uint lighting = 0;
        for (uint y = 0; y < 2; ++y) {    
            for (uint x = 0; x < 2; ++x) {
                uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                if (!data->madness_world) {
                    lighting = max(lighting, (uint) data->lighting_map.normal[ind]);
                } else {
                    lighting = max(lighting, (uint) data->lighting_map.madness[ind]);
                }
                lighting = max(lighting, (uint) data->lighting_map.dynamic[ind]);
            }
        }
        if (lighting == 0) {
            increaseMadness(data, 0.25f);
        } else if (lighting == 2) {
            increaseMadness(data, 0.05f);
        } else if (data->madness_world) {
            increaseMadness(data, -1.5f);
        } else {
            increaseMadness(data, -1.f);
        }

        player->noise = 0.f;
        for (uint agent_id = 1; agent_id < data->agents_size; ++agent_id) {
            if (data->agents[agent_id].type == Character::Type::Cultist) {
                Cultist* cultist = (Cultist*) data->agents[agent_id].agent;
                player->noise = max(player->noise, cultist->noise);
            }
        }

        return success;
    }
    
    __device__ void updateUsePlayer(Data* data, Player* player) {
        player->door = (uint) -1;
        player->window = (uint) -1;
        for (uint id = 0; id < data->interactables_size; ++id) {
            data->interactables_mask[id] = false;
        }
        int2 coords_round = {(int) floor(player->coords.x), (int) floor(player->coords.y)};
        for (uint y = max(0, coords_round.y - 3); y <= min(data->height - 1, coords_round.y + 3); ++y) {
            for (uint x = max(0, coords_round.x - 3); x <= min(data->width - 1, coords_round.x + 3); ++x) {
                uint i = y * data->width + x;
                if (data->interactables_map[i] != (uint) -1) {
                    data->interactables_mask[data->interactables_map[i]] = true;
                }
            }
        }

        for (uint id = 0; id < data->interactables_size; ++id) {
            if (!data->interactables_mask[id]) {
                continue;
            }
            float2 coords;
            switch (data->interactables[id].type) {
            case ObjectType::Door:
                coords = ((Door*) data->interactables[id].object)->coords;
                break;
            case ObjectType::Window:
                coords = ((Window*) data->interactables[id].object)->coords;
            }
            if (player->coords.x == coords.x) {
                data->interactables_mask[id] = (fabsf(player->coords.y - coords.y) <= 3);
            } else {
                float2 delta = {coords.x - player->coords.x, coords.y - player->coords.y};
                float dist = delta.x * delta.x + delta.y * delta.y;
                float rotation_door = atanf(delta.y / delta.x) / M_PI;
                if (delta.x < 0.f) {
                    rotation_door += 1.f;
                }
                float rotation_delta = fmodf(rotation_door - player->rotation, 2.f);
                if (rotation_delta < 0.f) {
                    rotation_delta += 2.f;
                }
                data->interactables_mask[id] = (dist <= 3 * 3) & (rotation_delta < 1.f / 6 || rotation_delta > 2.f - 1.f / 6);
            }
            
            if (data->interactables_mask[id]) {
                switch (data->interactables[id].type) {
                case ObjectType::Door:
                    player->door = id;
                    break;
                case ObjectType::Window:
                    player->window = id;
                }
            }
        }
    }

    __device__ void doActionsDevice(Data* data) {
        for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
            switch (data->agents[agent_id].type) {
            case Character::Type::Player: {
                Player* player = (Player*) data->agents[agent_id].agent;
                player->action_success = doActionPlayer(data, player);
                updateUsePlayer(data, player);
                player->max_x = max(player->max_x, player->coords.x);
                player->update();
                break;
            }
            case Character::Type::Astral: {
                Astral* astral = (Astral*) data->agents[agent_id].agent;
                astral->action_success = doActionAstral(data, astral);
                updateUseAstral(data, astral);
                astral->update();
                break;
            }
            case Character::Type::Cultist: {
                Cultist* cultist = (Cultist*) data->agents[agent_id].agent;
                cultist->action_success = doActionCultist(data, cultist);
                updateAbilityLight(data, cultist);
                cultist->update();
            }
            }
        }
        updateHealth(data);
        updateMadness(data);
    }

    __global__ void doActions(Data* data) {
        doActionsDevice(data);
    }

    __global__ void updateResetHandle(Data* data, cudaGraphConditionalHandle reset_handle) {
        cudaGraphSetConditional(reset_handle, data->health <= 0.f);
    }

    __device__ __forceinline__ bool checkSightAngle(float2 delta, float rotation, float sight_angle) {
        float angle_right = rotation - sight_angle / 2;
        float2 coords_right;
        sincospif(angle_right, &coords_right.y, &coords_right.x);
        float det_right = coords_right.x * delta.y - coords_right.y * delta.x;

        float angle_left = rotation + sight_angle / 2;
        float2 coords_left;
        sincospif(angle_left, &coords_left.y, &coords_left.x);
        float det_left = coords_left.x * delta.y - coords_left.y * delta.x;
        
        return det_right > 0.f && det_left < 0.f;
    }

    __device__ void fillSightAstral(Data* data, uint agent_id, uint2 point_coords) {
        Astral* astral = (Astral*) data->agents[agent_id].agent;
        uint point_ind = point_coords.y * data->width + point_coords.x;
        float2 coords = {point_coords.x + 0.5f, point_coords.y + 0.5f};

        float2 delta = {coords.x - astral->coords.x, coords.y - astral->coords.y};
        bool* sight = &(data->sight_masks[agent_id][point_ind]);
        
        if (!checkSightAngle(delta, astral->rotation, Astral::sight_angle)) {
            *sight = false;
            return;
        }

        if (delta.x * delta.x + delta.y * delta.y > Astral::sight_radius * Astral::sight_radius) {
            *sight = false;
            return;
        }

        if (!data->madness_world) {
            *sight = data->lighting_map.normal[point_ind] != LightingType::Dark;
        } else {
            *sight = data->lighting_map.madness[point_ind] != LightingType::Dark;
        }
        *sight |= data->lighting_map.dynamic[point_ind] != LightingType::Dark;

        uint boundary_id = data->boundaries_map[point_ind];
        while (boundary_id > data->boundaries_size) {
            data->boundaries_masks[agent_id][boundary_id % data->boundaries_size] = true;
            boundary_id /= data->boundaries_size;
        }
        if (boundary_id != (uint) -1) {
            data->boundaries_masks[agent_id][boundary_id % data->boundaries_size] = true;
        }
    }

    __device__ void fillSightPlayer(Data* data, uint2 point_coords) {
        Player* player = (Player*) data->agents[0].agent;
        uint point_ind = point_coords.y * data->width + point_coords.x;
        float2 coords = {point_coords.x + 0.5f, point_coords.y + 0.5f};

        float2 delta = {coords.x - player->coords.x, coords.y - player->coords.y};
        bool* sight = &(data->sight_masks[0][point_ind]);
        
        if (!checkSightAngle(delta, player->rotation, Player::sight_angle)) {
            *sight = false;
            return;
        }

        *sight = (delta.x * delta.x + delta.y * delta.y <= player->sight_radius * player->sight_radius);
        if (player->magic_cooldown_invisibility < 46) {
            if (!data->madness_world) {
                *sight |= data->lighting_map.normal[point_ind] != LightingType::Dark;
            } else {
                *sight |= data->lighting_map.madness[point_ind] != LightingType::Dark;
            }
            *sight |= data->lighting_map.dynamic[point_ind] != LightingType::Dark;
        }

        uint boundary_id = data->boundaries_map[point_ind];
        while (boundary_id > data->boundaries_size) {
            data->boundaries_masks[0][boundary_id % data->boundaries_size] = true;
            boundary_id /= data->boundaries_size;
        }
        if (boundary_id != (uint) -1) {
            data->boundaries_masks[0][boundary_id % data->boundaries_size] = true;
        }
    }

    __device__ void fillBoundaries(Data* data, uint agent_id, uint2 point_coords) {
        uint point_ind = point_coords.y * data->width + point_coords.x;
        float2 agent_coords = data->agents[agent_id].agent->coords;
        bool* sight = &data->sight_masks[agent_id][point_ind];
        if (!*sight) {
            return;
        }

        for (uint boundary_id = 0; boundary_id < data->boundaries_size; ++boundary_id) {
            if (!data->boundaries_masks[agent_id][boundary_id]) {
                continue;
            }

            float
                x_point = point_coords.x + 0.5f,
                y_point = point_coords.y + 0.5f,
                x_agent = agent_coords.x,
                y_agent = agent_coords.y,
                x_first = data->boundaries[boundary_id].first.x,
                y_first = data->boundaries[boundary_id].first.y,
                x_second = data->boundaries[boundary_id].second.x,
                y_second = data->boundaries[boundary_id].second.y;
            
            // point <- first -> second
            float det_point = (x_second - x_first) * (y_point - y_first) - (y_second - y_first) * (x_point - x_first);
            if (det_point == 0.f) {
                continue;
            }

            // agent <- first -> second
            float det_agent = (x_second - x_first) * (y_agent - y_first) - (y_second - y_first) * (x_agent - x_first);
            if (det_point == 0.f) {
                continue;
            }
            if ((det_agent < 0) ^ (det_point > 0)) {
                continue;
            }

            // point <- agent -> first
            float det_first = (x_first - x_agent) * (y_point - y_agent) - (y_first - y_agent) * (x_point - x_agent);
            if (det_first == 0.f) {
                *sight = false;
                return;
            }
            if ((det_agent < 0) ^ (det_first < 0)) {
                continue;
            }

            // point <- agent -> second
            float det_second = (x_second - x_agent) * (y_point - y_agent) - (y_second - y_agent) * (x_point - x_agent);
            if (det_second == 0.f) {
                *sight = false;
                return;
            }
            if ((det_agent < 0) ^ (det_second > 0)) {
                continue;
            }

            *sight = false;
            return;
        }
    }

    // hearing and sound
    __global__ void updateSound(Data* data) {
        uint ind = blockIdx.x * blockDim.x + threadIdx.x;
        if (ind >= data->width * data->height) {
            return;
        }

        *data->player_tensor.getHearing(ind) *= 0.99f;
        *data->player_tensor.getSound(ind) *= 0.99f;
        *data->monsters_tensor.getHearing(ind) *= 0.99f;
        *data->monsters_tensor.getSound(ind) *= 0.99f;
        data->player_hearing_map[ind] *= 0.99f;
        data->monsters_hearing_map[ind] *= 0.99f;

        float2 coords = {(ind % data->width) + 0.5f, (ind / data->width) + 0.5f};

        for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
            AgentData agent_data = data->agents[agent_id];
            float2 delta = {coords.x - agent_data.agent->coords.x, coords.y - agent_data.agent->coords.y};
            float dist = sqrt(delta.x * delta.x + delta.y * delta.y);
            switch (agent_data.type) {
            case Character::Type::Player: {
                Player* player = (Player*) agent_data.agent;
                if (player->magic_cooldown_inaudibility >= 46) {
                    continue;
                }
                if (dist <= player->hearing_radius) {
                    data->player_hearing_map[ind] = 1.f;
                }
                if (dist <= player->sound_radius) {
                    *data->player_tensor.getSound(ind) = __float2half_rn(1.f);
                }
                break;
            }
            case Character::Type::Cultist: {
                Cultist* cultist = (Cultist*) agent_data.agent;
                if (dist <= cultist->hearing_radius) {
                    data->monsters_hearing_map[ind] = __float2half_rn(1.f);
                }
                if (dist <= cultist->sound_radius) {
                    *data->monsters_tensor.getSound(ind) = __float2half_rn(1.f);
                }
                break;
            }
            case Character::Type::Astral: {
                Astral* astral = (Astral*) agent_data.agent;
                if (dist <= astral->sound_radius) {
                    *data->monsters_tensor.getSound(ind) = __float2half_rn(1.f);
                }
                break;
            }
            }
        }   
    }

    __global__ void clearMasks(Data* data) {
        uint ind = blockIdx.x * blockDim.x + threadIdx.x;
        if (ind >= data->width * data->height) {
            return;
        }
        for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
            Character::Type type = data->agents[agent_id].type;
            if (type == Character::Type::Player || type == Character::Type::Astral) {
                data->sight_masks[agent_id][ind] = false;
                if (ind < data->boundaries_size) {
                    data->boundaries_masks[agent_id][ind] = false;
                }
            }
        }

        data->lighting_map.dynamic[ind] = LightingType::Dark;
    }

    __global__ void fillSight(Data* data) {
        uint ind = blockIdx.x * blockDim.x + threadIdx.x;
        if (ind >= data->width * data->height) {
            return;
        }
        for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
            switch (data->agents[agent_id].type) {
            case Character::Type::Player:
                fillSightPlayer(data, {ind % data->width, ind / data->width});
                break;
            case Character::Type::Astral:
                fillSightAstral(data, agent_id, {ind % data->width, ind / data->width});
            }
        }
    }

    // sight, lighting and object
    __global__ void fillShadows(Data* data) {
        uint ind = blockIdx.x * blockDim.x + threadIdx.x;
        if (ind >= data->width * data->height) {
            return;
        }
        Player* player = (Player*) data->agents[0].agent;
        *data->player_tensor.getSight(ind) *= 0.99f;
        *data->monsters_tensor.getSight(ind) *= 0.99f;
        if (!(player->magic_cooldown_see >= 28 && player->magic_cooldown_see < 38)) {
            fillBoundaries(data, 0, {ind % data->width, ind / data->width});
        }
        if (data->sight_masks[0][ind]) {
            *data->player_tensor.getSight(ind) = __float2half_rn(1.f);
            uint lighting;
            if (!data->madness_world) {
                lighting = (uint) data->lighting_map.normal[ind];
            } else {
                lighting = (uint) data->lighting_map.madness[ind];
            }
            lighting = max(lighting, (uint) data->lighting_map.dynamic[ind]);
            *data->player_tensor.getLighting(ind) = Utils::getValue<LightingType>((LightingType) lighting);
            *data->player_tensor.getObject(ind) = Utils::getValue<ObjectType>(data->object_map[ind]);
        }
        for (uint agent_id = 1; agent_id < data->agents_size; ++agent_id) {
            if (data->agents[agent_id].type == Character::Type::Astral) {
                fillBoundaries(data, agent_id, {ind % data->width, ind / data->width});
                if (data->sight_masks[agent_id][ind]) {
                    *data->monsters_tensor.getSight(ind) = __float2half_rn(1.f);
                    if (data->object_map[ind] == ObjectType::Player && player->magic_cooldown_invisibility < 46) {
                        Astral* astral = (Astral*) data->agents[agent_id].agent;
                        astral->see_player = true;
                        astral->updateSeePlayer();
                    }
                }
            }
        }
        if (*data->monsters_tensor.getSight(ind) == __float2half_rn(1.f)) {
            uint lighting;
            if (!data->madness_world) {
                lighting = (uint) data->lighting_map.normal[ind];
            } else {
                lighting = (uint) data->lighting_map.madness[ind];
            }
            lighting = max(lighting, (uint) data->lighting_map.dynamic[ind]);
            *data->monsters_tensor.getLighting(ind) = Utils::getValue<LightingType>((LightingType) lighting);
            if (data->object_map[ind] == ObjectType::Player) {
                *data->monsters_tensor.getObject(ind) =
                    player->magic_cooldown_invisibility < 46 ?
                    Utils::getValue<ObjectType>(ObjectType::Player) :
                    Utils::getValue<ObjectType>(ObjectType::Clear);
            } else {
                *data->monsters_tensor.getObject(ind) = Utils::getValue<ObjectType>(data->object_map[ind]);
            }
        }
    }

    __host__ void step(cudaGraph_t* graph, Data* data, uint size) {

        cudaGraphNodeParams clear_masks_params = { cudaGraphNodeTypeKernel };
        void* clear_masks_input[] = {&data};
        clear_masks_params.kernel.func = (void*) clearMasks;
        clear_masks_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        clear_masks_params.kernel.blockDim = dim3(1024);
        clear_masks_params.kernel.kernelParams = clear_masks_input;
        cudaGraphNode_t clear_masks_node;
        cudaGraphAddNode(&clear_masks_node, *graph, NULL, 0, &clear_masks_params);

        cudaGraphNodeParams actions_params = { cudaGraphNodeTypeKernel };
        void* actions_input[] = {&data};
        actions_params.kernel.func = (void*) doActions;
        actions_params.kernel.gridDim = dim3(1);
        actions_params.kernel.blockDim = dim3(1);
        actions_params.kernel.kernelParams = actions_input;
        cudaGraphNode_t actions_node;
        cudaGraphAddNode(&actions_node, *graph, {&clear_masks_node}, 1, &actions_params);

        cudaGraphNodeParams sound_params = { cudaGraphNodeTypeKernel };
        void* sound_input[] = {&data};
        sound_params.kernel.func = (void*) updateSound;
        sound_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        sound_params.kernel.blockDim = dim3(1024);
        sound_params.kernel.kernelParams = sound_input;
        cudaGraphNode_t sound_node;
        cudaGraphAddNode(&sound_node, *graph, {&actions_node}, 1, &sound_params);

        cudaGraphNodeParams fill_sight_params = { cudaGraphNodeTypeKernel };
        void* fill_sight_input[] = {&data};
        fill_sight_params.kernel.func = (void*) fillSight;
        fill_sight_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        fill_sight_params.kernel.blockDim = dim3(1024);
        fill_sight_params.kernel.kernelParams = fill_sight_input;
        cudaGraphNode_t fill_sight_node;
        cudaGraphAddNode(&fill_sight_node, *graph, {&actions_node}, 1, &fill_sight_params);

        cudaGraphNodeParams fill_shadows_params = { cudaGraphNodeTypeKernel };
        void* fill_shadows_input[] = {&data};
        fill_shadows_params.kernel.func = (void*) fillShadows;
        fill_shadows_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        fill_shadows_params.kernel.blockDim = dim3(1024);
        fill_shadows_params.kernel.kernelParams = fill_shadows_input;
        cudaGraphNode_t fill_shadows_node;
        cudaGraphAddNode(&fill_shadows_node, *graph, {&fill_sight_node}, 1, &fill_shadows_params);
    }

    __host__ void stepReset(cudaGraph_t* graph, Data* data, Data* data_base, uint size) {

        cudaGraphConditionalHandle reset_handle;
        cudaGraphConditionalHandleCreate(&reset_handle, *graph, 0, cudaGraphCondAssignDefault);

        cudaGraphNodeParams clear_masks_params = { cudaGraphNodeTypeKernel };
        void* clear_masks_input[] = {&data};
        clear_masks_params.kernel.func = (void*) clearMasks;
        clear_masks_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        clear_masks_params.kernel.blockDim = dim3(1024);
        clear_masks_params.kernel.kernelParams = clear_masks_input;
        cudaGraphNode_t clear_masks_node;
        cudaGraphAddNode(&clear_masks_node, *graph, NULL, 0, &clear_masks_params);

        cudaGraphNodeParams actions_params = { cudaGraphNodeTypeKernel };
        void* actions_input[] = {&data};
        actions_params.kernel.func = (void*) doActions;
        actions_params.kernel.gridDim = dim3(1);
        actions_params.kernel.blockDim = dim3(1);
        actions_params.kernel.kernelParams = actions_input;
        cudaGraphNode_t actions_node;
        cudaGraphAddNode(&actions_node, *graph, {&clear_masks_node}, 1, &actions_params);

        cudaGraphNodeParams sound_params = { cudaGraphNodeTypeKernel };
        void* sound_input[] = {&data};
        sound_params.kernel.func = (void*) updateSound;
        sound_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        sound_params.kernel.blockDim = dim3(1024);
        sound_params.kernel.kernelParams = sound_input;
        cudaGraphNode_t sound_node;
        cudaGraphAddNode(&sound_node, *graph, {&actions_node}, 1, &sound_params);

        cudaGraphNodeParams fill_sight_params = { cudaGraphNodeTypeKernel };
        void* fill_sight_input[] = {&data};
        fill_sight_params.kernel.func = (void*) fillSight;
        fill_sight_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        fill_sight_params.kernel.blockDim = dim3(1024);
        fill_sight_params.kernel.kernelParams = fill_sight_input;
        cudaGraphNode_t fill_sight_node;
        cudaGraphAddNode(&fill_sight_node, *graph, {&actions_node}, 1, &fill_sight_params);

        cudaGraphNodeParams fill_shadows_params = { cudaGraphNodeTypeKernel };
        void* fill_shadows_input[] = {&data};
        fill_shadows_params.kernel.func = (void*) fillShadows;
        fill_shadows_params.kernel.gridDim = dim3((size - 1) / 1024 + 1);
        fill_shadows_params.kernel.blockDim = dim3(1024);
        fill_shadows_params.kernel.kernelParams = fill_shadows_input;
        cudaGraphNode_t fill_shadows_node;
        cudaGraphAddNode(&fill_shadows_node, *graph, {&fill_sight_node}, 1, &fill_shadows_params);

        cudaGraphNodeParams reset_update_params = { cudaGraphNodeTypeKernel };
        void* reset_update_input[] = {&data, &reset_handle};
        reset_update_params.kernel.func = (void*) updateResetHandle;
        reset_update_params.kernel.gridDim = dim3(1);
        reset_update_params.kernel.blockDim = dim3(1);
        reset_update_params.kernel.kernelParams = reset_update_input;
        cudaGraphNode_t reset_update_node;
        cudaGraphAddNode(&reset_update_node, *graph, {&actions_node}, 1, &reset_update_params);

        cudaGraphNodeParams reset_cond_params = { cudaGraphNodeTypeConditional };
        reset_cond_params.conditional.handle = reset_handle;
        reset_cond_params.conditional.type = cudaGraphCondTypeIf;
        reset_cond_params.conditional.size = 1;
        cudaGraphNode_t reset_cond_node;
        cudaGraphNode_t reset_cond_dependencies[] = {fill_shadows_node, reset_update_node}; 
        cudaGraphAddNode(&reset_cond_node, *graph, reset_cond_dependencies, 2, &reset_cond_params);
        cudaGraph_t reset_cond_graph = reset_cond_params.conditional.phGraph_out[0];

        cudaGraphNodeParams reset_params = { cudaGraphNodeTypeKernel };
        void* reset_args[] = {&data_base, &data};
        reset_params.kernel.func = (void*) Simulator::copyDeviceToDevice;
        reset_params.kernel.gridDim = dim3(1);
        reset_params.kernel.blockDim = dim3(1);
        reset_params.kernel.kernelParams = reset_args;
        cudaGraphNode_t reset_node;
        cudaGraphAddNode(&reset_node, reset_cond_graph, NULL, 0, &reset_params);
    }

    __host__ Data* initHost() {

        Data* data = (Data*) malloc(sizeof(Data));

        data->width = 300;
        data->height = 150;

        uint2_pair boundaries_desc[] = {
            // walls
            {{0, 0}, {0, 149}},
            {{0, 149}, {299, 149}},
            {{299, 0}, {299, 149}},
            {{0, 0}, {299, 0}},
            // left house
            {{57, 48}, {57, 70}},
            {{57, 73}, {57, 80}},
            {{56, 79}, {74, 79}},
            {{73, 48}, {73, 61}},
            {{73, 64}, {73, 80}},
            {{56, 49}, {74, 49}},
            {{56, 65}, {67, 65}},
            {{70, 65}, {74, 65}},
            // right house
            {{204, 68}, {204, 72}},
            {{204, 74}, {204, 83}},
            {{204, 85}, {204, 90}},
            {{204, 93}, {204, 99}},
            {{204, 99}, {224, 99}},
            {{224, 68}, {224, 99}},
            {{204, 68}, {205, 68}},
            {{208, 68}, {210, 68}},
            {{213, 68}, {224, 68}},
        };
        uint boundaries_size = sizeof(boundaries_desc) / sizeof(boundaries_desc[0]);

        Door doors_desc[] = {
            {{73.5f, 63.f}, {72.f, 63.f}, boundaries_size, {{73, 61}, {72, 61}, {71, 61}, {73, 64}}, {{73, 61}, {73, 62}, {73, 63}, {73, 64}}, false},
            {{69.f, 65.5f}, {69.f, 67.f}, boundaries_size + 1, {{67, 65}, {67, 66}, {67, 67}, {70, 65}}, {{67, 65}, {68, 65}, {69, 65}, {70, 65}}, false},
            {{204.5f, 92.f}, {206.f, 92.f}, boundaries_size + 2, {{204, 90}, {205, 90}, {206, 90}, {204, 93}}, {{204, 90}, {204, 91}, {204, 92}, {204, 93}}, false}
        };
        uint doors_size = sizeof(doors_desc) / sizeof(doors_desc[0]);

        Window windows_desc[] = {
            {{57.5f, 72.f}, {56.f, 72.f}, {59.f, 72.f}},
            {{207.f, 68.5f}, {207.f, 67.f}, {207.f, 70.f}},
            {{212.f, 68.5f}, {212.f, 67.f}, {212.f, 70.f}}
        };
        uint windows_size = sizeof(windows_desc) / sizeof(windows_desc[0]);

        std::pair<uint2, ObjectType> others_desc[] = {
            // windows
            {{204, 73}, ObjectType::NoMove},
            {{204, 84}, ObjectType::NoMove},
            // lanterns
            {{161, 100}, ObjectType::NoMove},
            {{146, 79}, ObjectType::NoMove},
            {{121, 58}, ObjectType::NoMove}
        };
        uint others_size = sizeof(others_desc) / sizeof(others_desc[0]);

        uint2_pair roofed_desc[] = {
            {{57, 49}, {73, 79}},
            {{204, 68}, {224, 99}}
        };
        uint roofed_size = sizeof(roofed_desc) / sizeof(roofed_desc[0]);

        uint2_pair lighted_desc[] = {
            {{60, 51}, {71, 56}},
            {{59, 58}, {71, 63}},
            {{59, 66}, {68, 71}},
            {{61, 73}, {70, 78}},
            {{51, 66}, {56, 75}},
            {{74, 58}, {80, 65}},
            {{214, 69}, {222, 75}},
            {{216, 94}, {222, 98}},
            {{161, 98}, {167, 102}},
            {{162, 97}, {166, 103}},
            {{146, 77}, {152, 81}},
            {{147, 76}, {151, 82}},
            {{118, 59}, {124, 63}},
            {{119, 58}, {123, 64}}
        };
        uint lighted_size = sizeof(lighted_desc) / sizeof(lighted_desc[0]);

        Player* player = (Player*) malloc(sizeof(Player));
        player->data = (half*) malloc(Player::data_size * sizeof(half));
        player->init({59.f, 77.f});
        player->rotation = 1.5f;
        player->updateRotationHost();

        Astral* astral[2];
        astral[0] = (Astral*) malloc(sizeof(Astral));
        astral[0]->data = (half*) malloc(Astral::data_size * sizeof(half));
        astral[0]->init({136.f, 48.f});
        astral[1] = (Astral*) malloc(sizeof(Astral));
        astral[1]->data = (half*) malloc(Astral::data_size * sizeof(half));
        astral[1]->init({136.f, 109.f});

        Cultist* cultist[2];
        cultist[0] = (Cultist*) malloc(sizeof(Cultist));
        cultist[0]->data = (half*) malloc(Cultist::data_size * sizeof(half));
        cultist[0]->init({99.f, 89.f});
        cultist[1] = (Cultist*) malloc(sizeof(Cultist));
        cultist[1]->data = (half*) malloc(Cultist::data_size * sizeof(half));
        cultist[1]->init({174.f, 79.f});

        data->agents_size = 5;
        data->agents = (AgentData*) malloc(data->agents_size * sizeof(AgentData));
        data->agents[0] = {Character::Type::Player, (Agent*) player};
        data->agents[1] = {Character::Type::Astral, (Agent*) astral[0]};
        data->agents[2] = {Character::Type::Astral, (Agent*) astral[1]};
        data->agents[3] = {Character::Type::Cultist, (Agent*) cultist[0]};
        data->agents[4] = {Character::Type::Cultist, (Agent*) cultist[1]};

        data->madness = 0.f;
        updateMadness(data);

        data->madness_world = false;

        data->health = 100.f;
        updateHealth(data);

        data->boundaries_size = boundaries_size + doors_size;
        data->boundaries = (float2_pair*) malloc(data->boundaries_size * sizeof(float2_pair));

        for (uint id = 0; id < boundaries_size; ++id) {
            uint2_pair boundary = boundaries_desc[id];
            data->boundaries[id] = {{boundary.first.x + 0.5f, boundary.first.y + 0.5f}, {boundary.second.x + 0.5f, boundary.second.y + 0.5f}};
        }

        for (uint id = 0; id < doors_size; ++id) {
            uint2_pair boundary =
                doors_desc[id].is_open ?
                uint2_pair{doors_desc[id].coords_open[0], doors_desc[id].coords_open[2]} :
                uint2_pair{doors_desc[id].coords_closed[0], doors_desc[id].coords_closed[3]};
            data->boundaries[boundaries_size + id] = {{boundary.first.x + 0.5f, boundary.first.y + 0.5f}, {boundary.second.x + 0.5f, boundary.second.y + 0.5f}};
        }

        data->boundaries_map = (uint*) malloc(data->width * data->height * sizeof(uint));
        
        std::fill(data->boundaries_map, data->boundaries_map + data->height * data->width, (uint) -1);

        for (uint boundary_id = 0; boundary_id < data->boundaries_size; ++boundary_id) {
            auto [a, b] = data->boundaries[boundary_id];
            float tan = (float) (b.y - a.y) / (b.x - a.x);
            auto [x_min, x_max] = std::minmax(a.x, b.x);
            for (uint x = x_min; x < x_max; ++x) {
                uint y = a.y + (uint)((x - a.x) * tan);
                uint i = y * data->width + x;
                if (data->boundaries_map[i] == (uint) -1) {
                    data->boundaries_map[i] = boundary_id;
                } else {
                    data->boundaries_map[i] = data->boundaries_map[i] * data->boundaries_size + boundary_id;
                }
            }
            const auto [y_min, y_max] = std::minmax(a.y, b.y);
            for (uint y = y_min; y < y_max; ++y) {
                uint x = a.x + (uint)((y - a.y) / tan);
                uint i = y * data->width + x;
                if (data->boundaries_map[i] == (uint) -1) {
                    data->boundaries_map[i] = boundary_id;
                } else {
                    data->boundaries_map[i] = data->boundaries_map[i] * data->boundaries_size + boundary_id;
                }
            }
        }

        data->interactables_size = doors_size + windows_size;
        data->interactables = (InteractableData*) malloc(data->interactables_size * sizeof(InteractableData));

        for (uint id = 0; id < doors_size; ++id) {
            Door* door = (Door*) malloc(sizeof(Door));
            *door = doors_desc[id];
            data->interactables[id] = {ObjectType::Door, door};
        }

        for (uint id = 0; id < windows_size; ++id) {
            Window* window = (Window*) malloc(sizeof(Window));
            *window = windows_desc[id];
            data->interactables[doors_size + id] = {ObjectType::Window, window};
        }

        data->interactables_mask = (bool*) malloc(data->interactables_size * sizeof(bool));

        data->interactables_map = (uint*) malloc(data->width * data->height * sizeof(uint));

        for (uint i = 0; i < data->width * data->height; ++i) {
            data->interactables_map[i] = (uint) -1;
        }

        data->object_map = (ObjectType*) malloc(data->width * data->height * sizeof(ObjectType));

        for (uint i = 0; i < data->height * data->width; ++i) {
            data->object_map[i] =
                data->boundaries_map[i] != (uint) -1 ?
                ObjectType::Boundary :
                ObjectType::Clear;
        }

        for (uint id = 0; id < doors_size; ++id) {
            uint2* coords =
                doors_desc[id].is_open ?
                doors_desc[id].coords_open :
                doors_desc[id].coords_closed;
            for (uint i = 0; i < 4; ++i) {
                data->interactables_map[coords[i].y * data->width + coords[i].x] = id;
                data->object_map[coords[i].y * data->width + coords[i].x] = ObjectType::Door;
            }
        }


        for (uint id = 0; id < windows_size; ++id) {
            float2 coords = windows_desc[id].coords;
            uint ind_first = (uint) std::floor(coords.y - 0.5f) * data->width + (uint) std::floor(coords.x - 0.5f);
            uint ind_second = (uint) std::ceil(coords.y - 0.5f) * data->width + (uint) std::ceil(coords.x - 0.5f);
            data->interactables_map[ind_first] = doors_size + id;
            data->interactables_map[ind_second] = doors_size + id;
            data->object_map[ind_first] = ObjectType::Window;
            data->object_map[ind_second] = ObjectType::Window;
        }

        for (uint id = 0; id < others_size; ++id) {
            uint2 coords = std::get<0>(others_desc[id]);
            data->object_map[coords.y * data->width + coords.x] = std::get<1>(others_desc[id]);
        }

        for (uint agent_id = 0; agent_id < data->agents_size; ++agent_id) {
            auto [type, agent] = data->agents[agent_id];

            switch (type) {
            case Character::Type::Astral: {
                uint2 coords_round = {(uint) std::floor(agent->coords.x) - 1, (uint) std::floor(agent->coords.y) - 1};
                for (uint y = 0; y < 3; ++y) {
                    for (uint x = 0; x < 3; ++x) {
                        uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                        data->object_map[ind] = ObjectType::Astral;
                    }
                }
                break;
            }
            case Character::Type::Cultist: {
                uint2 coords_round = {(uint) std::floor(agent->coords.x) - 1, (uint) std::floor(agent->coords.y) - 1};
                for (uint y = 0; y < 3; ++y) {
                    for (uint x = 0; x < 3; ++x) {
                        uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                        data->object_map[ind] = ObjectType::Cultist;
                    }
                }
                break;
            }
            case Character::Type::Player: {
                uint2 coords_round = {(uint) std::floor(agent->coords.x - 0.5f), (uint) std::floor(agent->coords.y - 0.5f)};
                for (uint y = 0; y < 2; ++y) {
                    for (uint x = 0; x < 2; ++x) {
                        uint ind = (coords_round.y + y) * data->width + (coords_round.x + x);
                        data->object_map[ind] = ObjectType::Player;
                    }
                }
            }
        }
        }

        data->lighting_map.normal = (LightingType*) malloc(data->height * data->width * sizeof(LightingType));
        data->lighting_map.madness = (LightingType*) malloc(data->height * data->width * sizeof(LightingType));
        data->lighting_map.dynamic = (LightingType*) malloc(data->height * data->width * sizeof(LightingType));
        for (uint i = 0; i < data->height * data->width; ++i) {
            data->lighting_map.normal[i] = LightingType::Dark;
            data->lighting_map.madness[i] = LightingType::Madness;
            data->lighting_map.dynamic[i] = LightingType::Dark;
        }
        for (uint area_id = 0; area_id < lighted_size; ++area_id) {
            uint2_pair area = lighted_desc[area_id];
            for (uint y = area.first.y; y < area.second.y; ++y) {
                for (uint x = area.first.x; x < area.second.x; ++x) {
                    data->lighting_map.normal[y * data->width + x] = LightingType::Normal;
                }
            }
        }
        for (uint area_id = 0; area_id < roofed_size; ++area_id) {
            uint2_pair area = roofed_desc[area_id];
            for (uint y = area.first.y; y < area.second.y; ++y) {
                for (uint x = area.first.x; x < area.second.x; ++x) {
                    data->lighting_map.madness[y * data->width + x] = data->lighting_map.normal[y * data->width + x];
                }
            }
        }

        data->sight_masks = (bool**) malloc(data->agents_size * sizeof(bool*));
        data->boundaries_masks = (bool**) malloc(data->agents_size * sizeof(bool*));
        for (uint id = 0; id < data->agents_size; ++id) {
            if (id == 0 || data->agents[id].type == Character::Type::Astral) {
                data->sight_masks[id] = (bool*) malloc(data->width * data->height * sizeof(bool));
                data->boundaries_masks[id] = (bool*) malloc(data->boundaries_size * sizeof(bool));
                std::fill(data->boundaries_masks[id], data->boundaries_masks[id] + data->boundaries_size, false);
            }
        }

        data->player_tensor.data = (half*) malloc(5 * data->width * data->height * sizeof(half));
        std::fill(data->player_tensor.data, data->player_tensor.data + 5 * data->width * data->height, 0.f);

        data->monsters_tensor.data = (half*) malloc(5 * data->width * data->height * sizeof(half));
        std::fill(data->monsters_tensor.data, data->monsters_tensor.data + 5 * data->width * data->height, 0.f);

        for (uint i = 0; i < data->height * data->width; ++i) {
            *(data->player_tensor.getObject(i)) = Utils::getValue<ObjectType>(data->object_map[i]);
            *(data->monsters_tensor.getObject(i)) = Utils::getValue<ObjectType>(data->object_map[i]);
            *(data->player_tensor.getLighting(i)) = Utils::getValue<LightingType>((LightingType) std::max((uint) data->lighting_map.normal[i], (uint) data->lighting_map.dynamic[i]));
            *(data->monsters_tensor.getLighting(i)) = *(data->player_tensor.getLighting(i));
        }

        data->player_hearing_map = (half*) malloc(data->width * data->height * sizeof(half));
        std::fill(data->player_hearing_map, data->player_hearing_map + data->width * data->height, 0.f);

        data->monsters_hearing_map = (half*) malloc(data->width * data->height * sizeof(half));
        std::fill(data->monsters_hearing_map, data->monsters_hearing_map + data->width * data->height, 0.f);

        Data* data_device = copyHostToDevice(data);
        
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
        step(&graph, data_device, data->height * data->width);
        
        deleteHost(data);
        
        cudaGraphExec_t graph_exec;
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        cudaGraphLaunch(graph_exec, 0);
        cudaDeviceSynchronize();
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);

        data = copyDeviceToHost(data_device);
        deleteDevice(data_device);

        return data;
    }
}