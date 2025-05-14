#pragma once
#include "utils.h"
#include "cuda_fp16.h"
#define _USE_MATH_DEFINES
#include "math.h"

namespace Character {
    enum class Type : uint {
        Player,
        Astral,
        Cultist
    };

    struct Agent {

        half* data;

        bool action_success;

        float2 coords;
        float rotation;
        uint cooldown;

        // a.b -> a b
        __host__ __forceinline__ void updateCoordsHost() {
            data[0] = __float2half_rn(std::floor(coords.x) / 300);
            data[1] = __float2half_rn(coords.x - std::floor(coords.x));
            data[2] = __float2half_rn(std::floor(coords.y) / 150);
            data[3] = __float2half_rn(coords.y - std::floor(coords.y));
        }
        __device__ __forceinline__ void updateCoordsDevice() {
            data[0] = __float2half_rn(floorf(coords.x) / 300);
            data[1] = __float2half_rn(coords.x - floorf(coords.x));
            data[2] = __float2half_rn(floorf(coords.y) / 150);
            data[3] = __float2half_rn(coords.y - floorf(coords.y));
        }

        // radians -> sin cos
        __host__ __forceinline__ void updateRotationHost() {
            data[4] = __float2half_rn(std::sinf(M_PI * rotation));
            data[5] = __float2half_rn(std::cosf(M_PI * rotation));
        }
        __device__ __forceinline__ void updateRotationDevice() {
            data[4] = __float2half_rn(sinpif(rotation));
            data[5] = __float2half_rn(cospif(rotation));
        }

        __host__ __device__ __forceinline__ void updateCooldown() {
            data[6] = __uint2half_rn(cooldown);
        }

        __host__ __device__ __forceinline__ void updateMadness(float madness) {
            data[7] = __float2half_rn(madness / 100);
        }

        __host__ __device__ __forceinline__ void updateHealth(float health) {
            data[8] = __float2half_rn(health / 100);
        }
    };

    struct Astral: Agent {

        enum class Action : uint {
            Nothing,

            GoForward,

            TurnRightFull,
            TurnRightHalf,
            TurnLeftFull,
            TurnLeftHalf,

            Ability
        };

        static constexpr float sight_angle = 0.671f;
        static constexpr float sight_radius = 30.f;
        static constexpr float rotation_speed = 0.25f;

        Action action;

        float speed;
        float sound_radius;
        uint ability_cooldown;
        uint scream_cooldown;
        uint attack_cooldown;
        bool see_player;
        bool attack_player;

        __host__ __device__ __forceinline__ void updateSpeed() {
            data[9] = __float2half_rn(speed);
        }

        __host__ __device__ __forceinline__ void updateSoundRadius() {
            data[10] = __float2half_rn(sound_radius / 150.f);
        }

        __host__ __device__ __forceinline__ void updateAbilityCooldown() {
            data[11] = __float2half_rn((float) ability_cooldown / 16);
        }

        __host__ __device__ __forceinline__ void updateScreamCooldown() {
            data[12] = __float2half_rn((float) scream_cooldown / 10);
        }

        __host__ __device__ __forceinline__ void updateSeePlayer() {
            data[13] = __float2half_rn((float) see_player);
        }

        __host__ __device__ __forceinline__ void updateAttackPlayer() {
            data[14] = __float2half_rn((float) attack_player);
        }

        static constexpr uint data_size = 15;

        __device__ __forceinline__ void update() {
            updateCoordsDevice();
            updateRotationDevice();
            updateCooldown();
            updateSpeed();
            updateSoundRadius();
            updateAbilityCooldown();
            updateScreamCooldown();
            updateSeePlayer();
            updateAttackPlayer();
        }

        __host__ __forceinline__ void updateHost() {
            updateCoordsHost();
            updateRotationHost();
            updateCooldown();
            updateSpeed();
            updateSoundRadius();
            updateAbilityCooldown();
            updateScreamCooldown();
            updateSeePlayer();
            updateAttackPlayer();
        }

        __host__ void init (float2 coords);
    };

    struct Cultist: Agent {

        enum class Action : uint {
            Nothing,

            GoForward,

            TurnRightFull,
            TurnRightHalf,
            TurnLeftFull,
            TurnLeftHalf,

            Ability
        };

        static constexpr float speed = 0.375f;
        static constexpr float hearing_radius = 2.5f;
        static constexpr float rotation_speed = 0.25f;

        Action action;

        float noise;
        float sound_radius;
        uint ability_cooldown;
        float2 ability_coords;


        __host__ __device__ __forceinline__ void updateNoise() {
            data[9] = __float2half_rn(noise);
        }
        
        __host__ __device__ __forceinline__ void updateSoundRadius() {
            data[10] = __float2half_rn(sound_radius / 65.f);
        }

        __host__ __device__ __forceinline__ void updateAbilityCooldown() {
            data[11] = __float2half_rn((float) ability_cooldown / 10);
        }

        static constexpr uint data_size = 12;

        __device__ __forceinline__ void update() {
            updateCoordsDevice();
            updateRotationDevice();
            updateCooldown();
            updateNoise();
            updateSoundRadius();
            updateAbilityCooldown();
        }

        __host__ __forceinline__ void updateHost() {
            updateCoordsHost();
            updateRotationHost();
            updateCooldown();
            updateNoise();
            updateSoundRadius();
            updateAbilityCooldown();
        }

        __host__ void init (float2 coords);
    };

    struct Player: Agent {

        enum class Action : uint {
            Nothing,

            GoForward,
            GoLeft,
            GoBack,
            GoRight,

            TurnRightFull,
            TurnRightHalf,
            TurnLeftFull,
            TurnLeftHalf,

            UseWindow,
            UseDoor,

            MagicSee,
            MagicLight,
            MagicInvisibility,
            MagicInaudibility,

            ChangeMovementRun,
            ChangeMovementCrouch
        };

        enum class MovementState : uint {
            Crouch,
            Walk,
            Run
        };

        enum class Magic : uint {
            See,
            // Light,
            Invisibility,
            Inaudibility
        };

        static constexpr float rotation_speed = 0.5f;
        static constexpr float sight_angle = 0.501f;

        Action action;

        float sight_radius;
        float sound_radius;
        float hearing_radius;
        MovementState movement_state;
        float speed;
        uint magic_cooldown_see;
        uint magic_cooldown_invisibility;
        uint magic_cooldown_inaudibility;
        float noise;
        uint door;
        uint window;
        float stamina;
        uint2 shadow_coords;
        float max_x;

        __host__ __device__ __forceinline__ void updateSightRadius() {
            data[9] = __float2half_rn(sight_radius / 300.f);
        }

        __host__ __device__ __forceinline__ void updateSoundRadius() {
            data[10] = __float2half_rn(sound_radius / 50.f);
        }

        __host__ __device__ __forceinline__ void updateHearingRadius() {
            data[11] = __float2half_rn(hearing_radius / 65.f);
        }

        __host__ __device__ __forceinline__ void updateMovementState() {
            data[12] = __float2half_rn((float) movement_state / 2);
        }

        __host__ __device__ __forceinline__ void updateSpeed() {
            data[13] = __float2half_rn(speed);
        }

        __host__ __device__ __forceinline__ void updateMagicCooldown() {
            data[14] = __float2half_rn((float) magic_cooldown_see / 40);
            data[15] = __float2half_rn((float) magic_cooldown_invisibility / 60);
            data[16] = __float2half_rn((float) magic_cooldown_inaudibility / 60);
        }

        __host__ __device__ __forceinline__ void updateNoise() {
            data[17] = __float2half_rn(noise);
        }

        __host__ __device__ __forceinline__ void updateUseDoor() {
            data[18] = __float2half_rn((float) (door != (uint) -1));
        }

        __host__ __device__ __forceinline__ void updateUseWindow() {
            data[19] = __float2half_rn((float) (window != (uint) -1));
        }

        __host__ __device__ __forceinline__ void updateStamina() {
            data[20] = __float2half_rn(stamina / 100);
        }

        __host__ __device__ __forceinline__ void updateShadowCoords() {
            data[21] = __float2half_rn((float) shadow_coords.x / 300);
            data[22] = __float2half_rn((float) shadow_coords.y / 150);
        }

        static constexpr uint data_size = 24;

        __device__ __forceinline__ void update() {
            updateCoordsDevice();
            updateRotationDevice();
            updateCooldown();
            updateSightRadius();
            updateSoundRadius();
            updateHearingRadius();
            updateMovementState();
            updateSpeed();
            updateMagicCooldown();
            updateNoise();
            updateUseDoor();
            updateUseWindow();
            updateStamina();
            updateShadowCoords();
        }

        __host__ __forceinline__ void updateHost() {
            updateCoordsHost();
            updateRotationHost();
            updateCooldown();
            updateSightRadius();
            updateSoundRadius();
            updateHearingRadius();
            updateMovementState();
            updateSpeed();
            updateMagicCooldown();
            updateNoise();
            updateUseDoor();
            updateUseWindow();
            updateStamina();
            updateShadowCoords();
        }

        __host__ void init (float2 coords);
    };
}