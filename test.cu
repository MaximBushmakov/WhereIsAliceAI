#include "thread_data.h"
#include "thread_methods.h"

#include "character.h"

#include "simulator_data.h"
#include "simulator_methods.h"

#include "utils.h"

#include "lodepng.h"
#include "heatmap.h"

#include "colorschemes/Spectral.h"

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

using Character::Agent;
using Character::Player;
using Character::Astral;
using Character::Cultist;

using PAct = Player::Action;
using AAct = Astral::Action;
using CAct = Cultist::Action;

void render_heatmap(Simulator::Data* data, float (*f) (Simulator::Data*, uint), std::string name) {
    
    heatmap_t* hm = heatmap_new(data->width, data->height);
    heatmap_stamp_t* stamp = heatmap_stamp_gen(0);
    for(uint y = 0; y < data->height; ++y) {
        for (uint x = 0; x < data->width; ++x) {
            heatmap_add_weighted_point_with_stamp(hm, x, y, f(data, y * data->width + x), stamp);
        }
    }

    std::vector<unsigned char> image(4 * data->height * data->width);
    heatmap_render_saturated_to(hm, heatmap_cs_Spectral_discrete, 1.f, &image[0]);

    std::vector<unsigned char> image_rev(4 * data->height * data->width);
    for (uint y = 0; y < data->height; ++y) {
        for (uint x = 0; x < data->width; ++x) {
            for (uint c = 0; c < 4; ++c) {
                image_rev[4 * data->width * y + 4 * x + c] = image[4 * data->width * (data->height - y - 1) + 4 * x + c];
            }
        }
    }

    heatmap_free(hm);

    unsigned err = lodepng::encode(name + ".png", image_rev, data->width, data->height);
}

void init_data_heatmap(float (*f) (Simulator::Data*, uint), std::string name) {
    Simulator::Data* data = Simulator::initHost();
    render_heatmap(data, f, name);
    Simulator::deleteHost(data);
}

void copy_heatmap(float (*f) (Simulator::Data*, uint), std::string name) {
    Simulator::Data* data = Simulator::initHost();
    Simulator::Data* data_device = Simulator::copyHostToDevice(data);
    Simulator::deleteHost(data);
    data = Simulator::copyDeviceToHost(data_device);
    render_heatmap(data, f, name);
    Simulator::deleteHost(data);
    Simulator::deleteDevice(data_device);
}

float object_map(Simulator::Data* data, uint i) {
    return (float) data->object_map[i] / (uint) Simulator::ObjectType::Last;
}

float player_sight_map(Simulator::Data* data, uint i) {
    return (float) *data->player_tensor.getSight(i);
}

float monsters_sight_map(Simulator::Data* data, uint i) {
    return (float) *data->monsters_tensor.getSight(i);
}

float player_hearing_map(Simulator::Data* data, uint i) {
    return (float) *(data->player_tensor.getHearing(i));
}

float monsters_hearing_map(Simulator::Data* data, uint i) {
    return (float) *(data->monsters_tensor.getHearing(i));
}

float lighting_map(Simulator::Data* data, uint i) {
    return (float) max((uint) (
            data->madness_world ?
            data->lighting_map.madness[i] :
            data->lighting_map.normal[i]),
        (uint) data->lighting_map.dynamic[i]) / 2.f;
}

void write_agents(Simulator::Data* data, std::string name) {
    std::ofstream player_file;
    player_file.open("./out/" + name + "/agents/player.csv", std::ios_base::app);
    half* player_data = data->agents[0].agent->data;
    for (uint i = 0; i < Player::data_size - 1; ++i) {
        player_file << std::to_string(__half2float(player_data[i])) << ", ";
    }
    player_file << std::to_string(__half2float(player_data[Player::data_size - 1])) << std::endl;
    player_file.close();

    for (uint astral_id = 0; astral_id < 2; ++astral_id) {
        std::ofstream astral_file;
        astral_file.open("./out/" + name + "/agents/astral_" + std::to_string(astral_id + 1) + ".csv", std::ios_base::app);
        half* astral_data = data->agents[1 + astral_id].agent->data;
        for (uint i = 0; i < Astral::data_size - 1; ++i) {
            astral_file << std::to_string(__half2float(astral_data[i])) << ", ";
        }
        astral_file << std::to_string(__half2float(astral_data[Astral::data_size - 1])) << std::endl;
        astral_file.close();
    }

    for (uint cultist_id = 0; cultist_id < 2; ++cultist_id) {
        std::ofstream cultist_file;
        cultist_file.open("./out/" + name + "/agents/cultist_" + std::to_string(cultist_id + 1) + ".csv", std::ios_base::app);
        half* cultist_data = data->agents[3 + cultist_id].agent->data;
        for (uint i = 0; i < Cultist::data_size - 1; ++i) {
            cultist_file << std::to_string(__half2float(cultist_data[i])) << ", ";
        }
        cultist_file << std::to_string(__half2float(cultist_data[Cultist::data_size - 1])) << std::endl;
        cultist_file.close();
    }
    
}

void clear_agents(std::string name) {
    std::ofstream file;
    std::filesystem::remove_all("./out/");
    std::filesystem::create_directory("./out/");
    std::filesystem::create_directory("./out/" + name + "/");
    std::filesystem::create_directory("./out/" + name + "/agents/");
    std::ofstream {"./out/" + name + "/agents/player.csv"};
    file.open("./out/" + name + "/agents/player.csv", std::ofstream::out | std::ofstream::trunc);
    file << "x_int, x_frac, y_int, y_frac, rot_sin, rot_cos, cooldown, madness, health, " <<
            "sight_rad, sound_rad, hearing_rad, mov_state, speed, mag_see, mag_inv, mag_ina, "
            "noise, door, window, stamina, shadow_x, shadow_y, max_x" << std::endl;
    file.close();
    std::ofstream {"./out/" + name + "/agents/astral_1.csv"};
    file.open("./out/" + name + "/agents/astral_1.csv", std::ofstream::out | std::ofstream::trunc);
    file << "x_int, x_frac, y_int, y_frac, rot_sin, rot_cos, cooldown, madness, health, " <<
            "speed, sound_rad, abil_cooldown, scream_cooldown, see, attack" << std::endl;
    file.close();
    std::ofstream {"./out/" + name + "/agents/astral_2.csv"};
    file.open("./out/" + name + "/agents/astral_2.csv", std::ofstream::out | std::ofstream::trunc);
    file << "x_int, x_frac, y_int, y_frac, rot_sin, rot_cos, cooldown, madness, health, " <<
            "speed, sound_rad, abil_cooldown, scream_cooldown, see, attack" << std::endl;
    file.close();
    std::ofstream {"./out/" + name + "/agents/cultist_1.csv"};
    file.open("./out/" + name + "/agents/cultist_1.csv", std::ofstream::out | std::ofstream::trunc);
    file << "x_int, x_frac, y_int, y_frac, rot_sin, rot_cos, cooldown, madness, health, " <<
            "noise, sound_rad, abil_cooldown" << std::endl;
    file.close();
    std::ofstream {"./out/" + name + "/agents/cultist_2.csv"};
    file.open("./out/" + name + "/agents/cultist_2.csv", std::ofstream::out | std::ofstream::trunc);
    file << "x_int, x_frac, y_int, y_frac, rot_sin, rot_cos, cooldown, madness, health, " <<
            "noise, sound_rad, abil_cooldown" << std::endl;
    file.close();
}

void log_data(Simulator::Data* data, uint action_id, std::string name) {
    std::filesystem::create_directory("./out/" + name + "/" + std::to_string(action_id) + "/");

    std::filesystem::create_directory("./out/" + name + "/" + std::to_string(action_id) + "/player/");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->player_tensor.getObject(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/player/object_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->player_tensor.getLighting(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/player/lighting_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->player_tensor.getSight(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/player/sight_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->player_tensor.getSound(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/player/sound_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->player_tensor.getHearing(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/player/hearing_map");

    std::filesystem::create_directory("./out/" + name + "/" + std::to_string(action_id) + "/monsters/");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->monsters_tensor.getObject(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/monsters/object_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->monsters_tensor.getLighting(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/monsters/lighting_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->monsters_tensor.getSight(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/monsters/sight_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->monsters_tensor.getSound(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/monsters/sound_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return __half2float(*data->monsters_tensor.getHearing(i));},
        "./out/" + name + "/" + std::to_string(action_id) + "/monsters/hearing_map");
    
    std::filesystem::create_directory("./out/" + name + "/" + std::to_string(action_id) + "/common/");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return (float) data->object_map[i] / (uint) Simulator::ObjectType::Last;},
        "./out/" + name + "/" + std::to_string(action_id) + "/common/object_map");
    render_heatmap(data,
        [](Simulator::Data* data, uint i) {return (float) max(
            (uint) (data->madness_world ? data->lighting_map.madness[i] : data->lighting_map.normal[i]),
            (uint) data->lighting_map.dynamic[i]) / 2.f;},
        "./out/" + name + "/" + std::to_string(action_id) + "/common/lighting_map");
    
    write_agents(data, name);
}

void test(std::string name) {
    clear_agents(name);
    Simulator::Data* data = Simulator::initHost();
    log_data(data, 0, name);
    std::ifstream actions_file("./tests/" + name + ".actions");
    uint p, a1, a2, c1, c2;
    uint action_id = 0;
    while (actions_file >> p >> a1 >> a2 >> c1 >> c2) {
        ++action_id;

        ((Player*) data->agents[0].agent)->action = (PAct) p;
        ((Astral*) data->agents[1].agent)->action = (AAct) a1;
        ((Astral*) data->agents[2].agent)->action = (AAct) a2;
        ((Cultist*) data->agents[3].agent)->action = (CAct) c1;
        ((Cultist*) data->agents[4].agent)->action = (CAct) c2;

        Simulator::Data* data_device = Simulator::copyHostToDevice(data);
        Simulator::deleteHost(data);
        
        cudaGraph_t graph = Simulator::step(data_device, 150 * 300);

        cudaGraphExec_t graph_exec;
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        cudaGraphLaunch(graph_exec, 0);
        cudaDeviceSynchronize();
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);

        data = Simulator::copyDeviceToHost(data_device);
        Simulator::deleteDevice(data_device);

        if (action_id % 10 == 0) {
            std::cout << action_id << std::endl;
        }

        log_data(data, action_id, name);
    }
    Simulator::deleteHost(data);
}

__global__ void testDeviceToDevice(Simulator::Data* data_orig, Simulator::Data* data_dest) {
    Simulator::copyDeviceToDevice(data_orig, data_dest);
}

__host__ void test_base() {
    Simulator::Data* data = Simulator::initHost();
    Simulator::Data* data_device = Simulator::copyHostToDevice(data);
    Simulator::deleteHost(data);
    data = Simulator::copyDeviceToHost(data_device);
    Simulator::deleteDevice(data_device);
    data_device = Simulator::copyHostToDevice(data);
    Simulator::Data* data_device_2 = Simulator::copyHostToDevice(data);
    Simulator::deleteHost(data);
    //testDeviceToDevice<<<1, 1>>>(data_device, data_device_2);
    cudaDeviceSynchronize();
    Simulator::deleteDevice(data_device_2);
    data = Simulator::copyDeviceToHost(data_device);
    Simulator::deleteDevice(data_device);
    std::filesystem::remove_all("./out/base/");
    std::filesystem::create_directory("./out/base/");
    render_heatmap(data, object_map, "./out/base/object_map");
    render_heatmap(data, player_sight_map, "./out/base/player_sight_map");
    render_heatmap(data, monsters_sight_map, "./out/base/monsters_sight_map");
    render_heatmap(data, player_hearing_map, "./out/base/player_hearing_map");
    render_heatmap(data, monsters_hearing_map, "./out/base/monsters_hearing_map");
    render_heatmap(data, lighting_map, "./out/base/lighting_map");
    Simulator::deleteHost(data);
}


int main(int argc, char* argv[]) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (argc == 1) {
        test_base();
    } else {
        std::string name {argv[1]};
        test(name);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time (ms): " << ms << std::endl;

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
}