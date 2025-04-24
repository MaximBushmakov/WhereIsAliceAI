#include "character.h"
#include "simulator_data.h"
#include "simulator_methods.h"
#include "utils.h"

#include <iostream>
// minmax, copy
#include <algorithm>

namespace Simulator {

    __host__ __device__ void removeAgent(half* object_map, Character::Agent* agent) {

        float2 coords = agent->getCoords();
        uint x = (uint) std::roundf(coords.x);
        uint y = (uint) std::roundf(coords.y);

        switch (agent->getType()) {
        case Character::Type::Astral:
            for (uint r = y; r < y + 3; ++r) {
                for (uint c = x; c < x + 3; ++c) {
                    object_map[r * width + c] = ObjectType::Clear;
                }
            }
            break;
        case Character::Type::Cultist:
            for (uint r = y; r < y + 2; ++r) {
                for (uint c = x; c < x + 2; ++c) {
                    object_map[r * width + c] = ObjectType::Clear;
                }
            }
            break;
        case Character::Type::Player:
            for (uint r = y; r < y + 2; ++r) {
                for (uint c = x; c < x + 2; ++c) {
                    object_map[r * width + c] = ObjectType::Clear;
                }
            }
            break;
        }
    }

    __host__ __device__ bool addAgent(half* object_map, Character::Agent* agent) {

        float2 coords = agent->getCoords();
        uint x = (uint) std::roundf(coords.x);
        uint y = (uint) std::roundf(coords.y);

        switch (agent->getType()) {
        case Character::Type::Astral:
            for (uint r = y; r < y + 3; ++r) {
                for (uint c = x; c < x + 3; ++c) {
                    if (object_map[r * width + c] == ObjectType::Clear) {
                        object_map[r * width + c] = ObjectType::Astral;
                    } else {
                        return false;
                    }
                }
            }
            break;
        case Character::Type::Cultist:
            for (uint r = y; r < y + 2; ++r) {
                for (uint c = x; c < x + 2; ++c) {
                    if (object_map[r * width + c] == ObjectType::Clear) {
                        object_map[r * width + c] = ObjectType::Cultist;
                    } else {
                        return false;
                    }
                }
            }
            break;
        case Character::Type::Player:
            for (uint r = y; r < y + 2; ++r) {
                for (uint c = x; c < x + 2; ++c) {
                    if (object_map[r * width + c] == ObjectType::Clear) {
                        object_map[r * width + c] = ObjectType::Player;
                    } else {
                        return false;
                    }
                }
            }
            break;
        }
        return true;
    }

    __host__ void initBoundariesMap(uint width, uint height, uint boundaries_size, uint2_pair* boundaries, uint* boundaries_map) {

        for (uint r = 0; r < height; ++r) {
            for (uint c = 0; c < width; ++c) {
                boundaries_map[r * width + c] = (uint) -1;
            }
        }

        for (uint boundaries_id = 0; boundaries_id < boundaries_size; ++boundaries_id) {
            const auto [a, b] = boundaries[boundaries_id];
            const float tan = (float) (b.y - a.y) / (b.x - a.x);

            const auto [x_min, x_max] = std::minmax(a.x, b.x);
            for (uint x = x_min; x < x_max; ++x) {
                uint y = a.y + (int)((x - a.x) * tan);
                boundaries_map[y * width + x] = boundaries_id;
            }

            const auto [y_min, y_max] = std::minmax(a.y, b.y);
            for (uint y = y_min; y < y_max; ++y) {
                uint x = a.x + (int)((y - a.y) / tan);
                boundaries_map[y * width + x] = boundaries_id;
            }
        }
    }

    __host__ void initLightingMap(uint width, uint height, uint lighted_size, uint2_pair* lighted_area, uint* lighting_map) {
        for (uint r = 0; r < height; ++r) {
            for (uint c = 0; c < width; ++c) {
                lighting_map[r * width + c] = getValue<LightingType>(LightingType::Dark);
            }
        }

        for (uint area_id = 0; area_id < lighted_size; ++area_id) {
            uint2_pair area = lighted_area[area_id];
            for (uint r = area.first.y; r < area.second.y; ++r) {
                for (uint c = area.first.x; c < area.second.x; ++c) {
                    lighting_map[r * width + c] = getValue<LightingType>(LightingType::Normal);
                }
            }
        }
    }

    __host__ Data* initHost() {

        Data* data = (Data*) malloc(sizeof(Data));

        data->width = 300;
        data->height = 150;

        data->lighted_size = 14;
        data->lighted_area = (uint2_pair*) malloc(data->lighted_size * sizeof(uint2_pair));
        data->roofed_size = 2;
        data->roofed_area = (uint2_pair*) malloc(data->roofed_size * sizeof(uint2_pair));

        std::tuple<uint2, uint2, ObjectType> boundaries_desc[] = {
            // walls
            {{0, 0}, {0, 149}, ObjectType::Boundary},
            {{0, 149}, {299, 149}, ObjectType::Boundary},
            {{299, 0}, {299, 149}, ObjectType::Boundary},
            {{0, 0}, {299, 0}, ObjectType::Boundary},
            // left house
            {{57, 48}, {57, 70}, ObjectType::Boundary},
            {{57, 71}, {57, 72}, ObjectType::Window},
            {{57, 73}, {57, 80}, ObjectType::Boundary},
            {{56, 79}, {74, 79}, ObjectType::Boundary},
            {{73, 48}, {73, 61}, ObjectType::Boundary},
            {{73, 62}, {73, 63}, ObjectType::DoorClosed},
            {{73, 64}, {73, 80}, ObjectType::Boundary},
            {{56, 49}, {74, 49}, ObjectType::Boundary},
            {{56, 64}, {67, 64}, ObjectType::Boundary},
            {{68, 64}, {69, 64}, ObjectType::DoorClosed},
            {{70, 64}, {74, 64}, ObjectType::Boundary},
            // right house
            {{204, 68}, {204, 72}, ObjectType::Boundary},
            {{204, 73}, {204, 73}, ObjectType::NoMove},
            {{204, 74}, {204, 83}, ObjectType::Boundary},
            {{204, 84}, {204, 84}, ObjectType::NoMove},
            {{204, 85}, {204, 91}, ObjectType::Boundary},
            {{204, 92}, {204, 93}, ObjectType::DoorClosed},
            {{204, 94}, {204, 99}, ObjectType::Boundary},
            {{204, 99}, {224, 99}, ObjectType::Boundary},
            {{224, 68}, {224, 99}, ObjectType::Boundary},
            {{204, 68}, {205, 68}, ObjectType::Boundary},
            {{206, 68}, {207, 68}, ObjectType::Window},
            {{208, 68}, {210, 68}, ObjectType::Boundary},
            {{211, 68}, {212, 68}, ObjectType::Window},
            {{213, 68}, {224, 68}, ObjectType::Boundary},
            // lanterns
            {{161, 100}, {161, 100}, ObjectType::Boundary},
            {{146, 79}, {146, 79}, ObjectType::Boundary},
            {{121, 58}, {121, 58}, ObjectType::Boundary}
        };

        uint2_pair roofed_desc[] = {
            {{57, 49}, {73, 79}},
            {{204, 68}, {224, 99}}
        };

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

        data->boundaries_size = sizeof(boundaries_desc) / sizeof(boundaries_desc[0]);
        data->boundaries = (uint2_pair*) malloc(data->boundaries_size * sizeof(uint2_pair));

        ObjectType* types = (ObjectType*) malloc(data->boundaries_size * sizeof(ObjectType));

        for (uint id = 0; id < boundaries_size; ++id) {
            data->boundaries[id] = {std::get<0>(boundaries_desc[id]), std::get<1>(boundaries_desc[id])};
            types[id] = std::get<2>(boundaries_desc[id]);
        }

        data->roofed_size = sizeof(roofed_desc) / sizeof(roofed_desc[0]);
        data->roofed_area = (uint2_pair*) malloc(data->roofed_size * sizeof(uint2_pair));
        std::copy((uint2_pair*) roofed_desc, (uint2_pair*) roofed_desc + data->roofed_size, data->roofed_area);

        data->lighting_size = sizeof(lighting_desc) / sizeof(lighting_desc[0]);
        data->lighting_area = (uint2_pair*) malloc(data->lighting_size * sizeof(uint2_pair));
        std::copy((uint2_pair*) lighting_desc, (uint2_pair*) lighting_desc + data->lighting_size, data->lighting_area);

        data->boundaries_map = (uint*) malloc(data->width * data->height * sizeof(uint));
        initBoundariesMap(data->width, data->height, data->boundaries_size, data->boundaries, data->boundaries_map);

        data->player.size = data->width * data->height;
        data->player.data = (half*) malloc(data->player->getSize() * sizeof(half));

        data->monsters.size = data->width * data->height;
        data->monsters.data = (half*) malloc(data->monsters->getSize() * sizeof(half));

        for (uint r = 0; r < height; ++r) {
            for (uint c = 0; c < width; ++c) {
                data->player->getObjectMap()[r * width + c] = (
                    data->boundaries_map[r * width + c] != (uint) -1 ?
                    getValue<ObjectType>(types[data->boundaries_map[r * width + c]]) :
                    getValue<ObjectType>(ObjectType::Clear)
                );
            }
        }

        free(types);

        Character::Player* player = (Character::Player*) malloc(sizeof(Character::Player));
        player.data = (half*) malloc(Character::Player.getSize() * sizeof(half));
        player->init({58.f, 77.f});

        Character::Astral* astral[2];
        astral[0] = (Character::Astral*) malloc(sizeof(Character::Astral));
        astral[0]->data = (half*) malloc(Character::Astral.getSize() * sizeof(half));
        astral[0]->init({136.f, 48.f});
        astral[1] = (Character::Astral*) malloc(sizeof(Character::Astral));
        astral[1]->data = (half*) malloc(Character::Astral.getSize() * sizeof(half));
        astral[1]->init({136.f, 109.f});

        Character::Cultist* cultist[2];
        cultist[0] = (Character::Cultist*) malloc(sizeof(Character::Cultist));
        cultist[0]->data = (half*) malloc(Character::Cultist.getSize() * sizeof(half));
        cultist[0]->init({99.f, 89.f});
        cultist[1] = (Character::Cultist*) malloc(sizeof(Character::Cultist));
        cultist[1]->data = (half*) malloc(Character::Cultist.getSize() * sizeof(half));
        cultist[1]->init({174.f, 79.f});

        data->agents_size = 5;
        data->agents = (Character::Agent**) malloc(data->agents_size * sizeof(Character::Agent*));
        data->agents[0] = (Character::Agent*) player;
        data->agents[1] = (Character::Agent*) astral[0];
        data->agents[2] = (Character::Agent*) astral[1];
        data->agents[3] = (Character::Agent*) cultist[0];
        data->agents[4] = (Character::Agent*) cultist[1];

        for (uint agent_id = 0; agent_id < agents_size; ++agent_id) {
            addAgent(data->player->getObjectMap(), data->agents[agent_id]);
        }

        std::copy(data->player->getObjectMap(), data->player->getObjectMap() + data->player->getSize(), data->monsters->getObjectMap());

        initLightingMap(data->width, data->height, data->lighted_size, data->lighted_area, data->player->getLightingMap());
        std::copy(data->player->getLightingMap(), data->player->getLightingMap() + data->player->getSize(), data->monsters->getLightingMap());

        std::fill(data->player->getSoundMap(), data->player->getSoundMap() + data->player->getSize(), __float2half_rn(0.f));
        std::fill(data->player->getHearingMap(), data->player->getHearingMap() + data->player->getSize(), __float2half_rn(0.f));
        std::fill(data->player->getSightMap(), data->player->getSightMap() + data->player->getSize(), __float2half_rn(0.f));

        std::fill(data->monsters->getSoundMap(), data->monsters->getSoundMap() + data->monsters->getSize(), __float2half_rn(0.f));
        std::fill(data->monsters->getHearingMap(), data->monsters->getHearingMap() + data->monsters->getSize(), __float2half_rn(0.f));
        std::fill(data->monsters->getSightMap(), data->monsters->getSightMap() + data->monsters->getSize(), __float2half_rn(0.f));

        return data;
    }

    __host__ Data* copyHostToDevice(Data* data_orig) {

        Data* data_dest_host = (Data*) malloc(sizeof(Data));
        *data_dest_host = *data_orig;

        cudaMalloc(&data_dest_host->boundaries, data_orig->boundaries_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest_host->boundaries, data_orig->boundaries, data_orig->boundaries_size * sizeof(uint2_pair), cudaMemcpyHostToDevice);

        cudaMalloc(&data_dest_host->boundaries_map, data_orig->height * data_orig->width * sizeof(uint));
        cudaMemcpy(data_dest_host->boundaries_map, data_orig->boundaries_map, data_orig->height * data_orig->width * sizeof(uint), cudaMemcpyHostToDevice);

        cudaMalloc(&data_dest_host->roofed_area, data_orig->roofed_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest_host->roofed_area, data_orig->roofed_area, data_orig->roofed_size * sizeof(uint2_pair), cudaMemcpyHostToDevice);

        cudaMalloc(&data_dest_host->lighted_area, data_orig->lighted_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest_host->lighted_area, data_orig->lighted_area, data_orig->lighted_size * sizeof(uint2_pair), cudaMemcpyHostToDevice);

        Character::Agent** agents_host = (Character::Agent**) malloc(data_orig->agents_size * sizeof(Character::Agent*));

        for (uint agent_id = 0; agent_id < data_orig->agents_size; ++agent_id) {
            Character::Agent* agent = data_orig->agents[agent_id];
            switch (agent->getType()) {
                case Character::Type::Player:
                    cudaMalloc(&agents_host[agent_id], sizeof(Character::Player));
                    cudaMemcpy(agents_host[agent_id], agent, sizeof(Character::Player), cudaMemcpyHostToDevice);
                    break;
                case Character::Type::Astral:
                    cudaMalloc(&agents_host[agent_id], sizeof(Character::Astral));
                    cudaMemcpy(agents_host[agent_id], agent, sizeof(Character::Astral), cudaMemcpyHostToDevice);
                    break;
                case Character::Type::Cultist:
                    cudaMalloc(&agents_host[agent_id], sizeof(Character::Cultist));
                    cudaMemcpy(agents_host[agent_id], agent, sizeof(Character::Cultist), cudaMemcpyHostToDevice);
                    break;
            }
        }

        cudaMalloc(&data_dest_host->agents, data_orig->agents_size * sizeof(Character::Agent*));
        cudaMemcpy(data_dest_host->agents, agents_host, data_orig->agents_size * sizeof(Character::Agent*), cudaMemcpyHostToDevice);

        free(agents);

        cudaMalloc(&data_dest_host->player.data, data_orig->player->getSize() * sizeof(half));
        cudaMemcpy(data_dest_host->player.data, data_orig->player.data, data_orig->palyer->getSize() * sizeof(half), cudaMemcpyHostToDevice);

        cudaMalloc(&data_dest_host->monsters.data, data_orig->monsters->getSize() * sizeof(half));
        cudaMemcpy(data_dest_host->monsters.data, data_orig->monsters.data, data_orig->palyer->getSize() * sizeof(half), cudaMemcpyHostToDevice);

        Data* data_dest_device;
        cudaMalloc(&data_dest_device, sizeof(Data));
        cudaMemcpy(data_dest_device, data_dest_host, sizeof(Data), cudaMemcpyHostToDevice);

        free(data_dest_host);

        return data_dest_device;
    }

    __host__ void deleteHost(Data* data) {
        free(data->boundaries);
        free(data->boundaries_map);
        free(data->roofed_area);
        free(data->lighted_area);
        for (uint id = 0; id < data->agents_size; ++id) {
            free(data->agents[id]);
        }
        free(data->agents);
        free(player.data);
        free(monsters.data);
        free(data);
    }

    __device__ Data* copyDeviceToDevice(Data* data_orig, Data* data_dest) {

        Data* data_dest;
        cudaMalloc(&data_dest, sizeof(Data));
        cudaMemcpy(data_dest, data_orig, sizeof(Data), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->boundaries, data_orig->boundaries_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest->boundaries, data_orig->boundaries, data_orig->boundaries_size * sizeof(uint2_pair), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->boundaries_map, data_orig->height * data_orig->width * sizeof(uint));
        cudaMemcpy(data_dest->boundaries_map, data_orig->boundaries_map, data_orig->height * data_orig->width * sizeof(uint), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->roofed_area, data_orig->roofed_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest->roofed_area, data_orig->roofed_area, data_orig->roofed_size * sizeof(uint2_pair), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->lighted_area, data_orig->lighted_size * sizeof(uint2_pair));
        cudaMemcpy(data_dest->lighted_area, data_orig->lighted_area, data_orig->lighted_size * sizeof(uint2_pair), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->agents, data_orig->agents_size * sizeof(Character::Agent*));

        for (uint id = 0; id < data_orig->agents_size; ++id) {
            switch (data_orig->agents[id]->getType()) {
                case Character::Type::Player:
                    cudaMalloc(&data_dest->agents[id], sizeof(Character::Player));
                    cudaMemcpy(data_dest->agents[id], data_orig->agents[id], sizeof(Character::Player), cudaMemcpyDeviceToDevice);
                    break;
                case Character::Type::Astral:
                    cudaMalloc(&data_dest->agents[id], sizeof(Character::Astral));
                    cudaMemcpy(data_dest->agents[id], data_orig->agents[id], sizeof(Character::Astral), cudaMemcpyDeviceToDevice);
                    break;
                case Character::Type::Cultist:
                    cudaMalloc(&data_dest->agents[id], sizeof(Character::Cultist));
                    cudaMemcpy(data_dest->agents[id], data_orig->agents[id], sizeof(Character::Cultist), cudaMemcpyDeviceToDevice);
                    break;
            }
        }

        cudaMalloc(&data_dest->player.data, data_orig->player->getSize() * sizeof(half));
        cudaMemcpy(data_dest->player.data, data_orig->player.data, data_orig->palyer->getSize() * sizeof(half), cudaMemcpyDeviceToDevice);

        cudaMalloc(&data_dest->monsters.data, data_orig->monsters->getSize() * sizeof(half));
        cudaMemcpy(data_dest->monsters.data, data_orig->monsters.data, data_orig->palyer->getSize() * sizeof(half), cudaMemcpyDeviceToDevice);

        return data_dest;
    }

    __host__ void deleteDevice(Data* data) {

        Data* data_host = (Data*) malloc(sizeof(Data));
        cudaMemcpy(data_host, data, sizeof(Data), cudaMemcpyDeviceToHost);

        cudaFree(data_host->boundaries);
        cudaFree(data_host->boundaries_map);
        cudaFree(data_host->roofed_area);
        cudaFree(data_host->lighted_area);

        Character::Agent* agents = (Character::Agent*) malloc(data_host->agents_size * sizeof(Character::Agent*));
        cudaMemcpy(agents, data_host->agents, data_host->agents_size * sizeof(uint), cudaMemcpyDeviceToHost);

        for (uint id = 0; id < data_host->agents_size; ++id) {
            cudaFree(agents[id]);
        }
        cudaFree(data_host->agents);
        free(agents);

        cudaFree(data_host->player.data);
        cudaFree(data_host->monsters.data);

        cudaFree(data);
    }

    __device__ void actionGo(Data* data, Character::Agent* agent, float rotation = 0.f) {
        removeAgent(data, agent);
        float2 coords_origin = agent->coords;
        agent->coords = {
            agent->coords.x + agent->speed * cospif(agent->rotation + rotation),
            agent->coords.y + agent->speed * sinpif(agent->rotation + rotation)
        };
        bool success = addAgent(data, agent);
        if (success) {
            agent->updateCoords();
        } else {
            agent->coords = coords_origin;
            addAgent(data, agent);
        }
        break;
    }

    __device__ void doActionAstral(Data* data, Character::Astral* agent) {
        switch (agent->action) {
        case Character::Astral::Action::Nothing:
            break;
        case Character::Astral::Action::GoForward:
            actionGo(data, (Character::Agent*) agent);
            break;
        case Character::Astral::Action::TurnRightFull:
            agent->rotation -= agent->rotation_speed;
            agent->updateRotationDevice();
            break;
        case Character::Astral::Action::TurnRightHalf:
            agent->rotation -= agent->rotation_speed / 2;
            agent->updateRotationDevice();
            break;
        case Character::Astral::Action::TurnLeftFull:
            agent->rotation += agent->rotation_speed;
            agent->updateRotationDevice();
            break;
        case Character::Astral::Action::TurnLeftHalf:
            agent->rotation += agent->rotation_speed / 2;
            agent->updateRotationDevice();
            break;
        case Character::Astral::Action::Ability:
            if (data->)
            agent->cooldown = -1;
            agent->updateCooldown();

            break;
        }
    }

    __device__ void doActionCultist(Data* data, Character::Cultist* agent) {
        switch (agent->action) {
        case Character::Cultist::Action::Nothing:
            break;
        case Character::Cultist::Action::GoForward:
            actionGo(data, (Character::Agent*) agent);
            break;
        case Character::Cultist::Action::TurnRightFull:
            agent->rotation +=
            break;
        case Character::Cultist::Action::TurnRightHalf:
            break;
        case Character::Cultist::Action::TurnLeftFull:
            break;
        case Character::Cultist::Action::TurnLeftHalf:
            break;
        case Character::Cultist::Action::Ability:
            break;
        }
    }

    __device__ void doActionPlayer(Data* data, Character::Player agent) {
        switch (agent->action) {
        case Character::Player::Action::Nothing:
            break;
        case Character::Player::Action::GoForward:
            actionGo(data, (Character::Agent*) agent);
            break;
        case Character::Player::Action::GoLeft:
            actionGo(data, (Character::Agent*) agent, 0.5f);
            break;
        case Character::Player::Action::GoBack:
            actionGo(data, (Character::Agent*) agent, 1.f);
            break;
        case Character::Player::Action::GoRight:
            actionGo(data, (Character::Agent*) agent, -0.5f);
            break;
        case Character::Player::Action::TurnRightFull:
            agent->rotation -= agent->rotation_speed;
            agent->updateRotationDevice();
            break;
        case Character::Player::Action::TurnRightHalf:
            agent->rotation -= agent->rotation_speed / 2;
            agent->updateRotationDevice();
            break;
        case Character::Player::Action::TurnLeftFull:
            agent->rotation += agent->rotation_speed;
            agent->updateRotationDevice();
            break;
        case Character::Player::Action::TurnLeftHalf:
            agent->rotation += agent->rotation_speed / 2;
            agent->updateRotationDevice();
            break;
        case Character::Player::Action::UseWindow:
            break;
        case Character::Player::Action::UseDoor:
            break;
        case Character::Player::Action::MagicSee:
            break;
        case Character::Player::Action::MagicInvisibility:
            break;
        case Character::Player::Action::MagicInaudibility:
            break;
        case Character::Player::Action::ChangeMovementRun:
            break;
        case Character::Player::Action::ChangeMovementCrouch:
            break;
        }
    }

    __global__ void doAction(Data* data) {
        for (uint agent_id = 0; agent_id < data->agents_num; ++agent_id) {
            Character::Agent* agent = data->agents[agent_id];
            switch (agent->getType()) {
            case Character::Type::Astral:
                doActionAstral(data, (Character::Astral*) agent);
                break;
            case Character::Type::Cultist:
                doActionCultist(data, (Character::Cultist*) agent);
                break;
            case Character::Type::Player:
                doActionPlayer(data, (Character::Cultist*) agent);
                break;
            }
        }
    }

    __host__ cudaGraph_t step(Data* data) {
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);

        doActions<<<1, 1>>>(data);

        return graph;
    }
}