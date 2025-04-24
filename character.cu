#include "character.h"
#include "utils.h"

#include "cuda_fp16.h"
#include "math.h"

namespace Character {

    /*
    struct Flying {
        __device__ void updateNextPosition() {
            next_position = positions[Utils::curandUint(0, positions_num, seed)];
        }
    }
    */

    struct Agent {

        // norm (256)
        __host__ __device__ void updateCoords() {
            data[0] = __float2half_rn(coords.x) / 256;
            data[1] = __float2half_rn(coords.y) / 256;
        }

        // radians -> sin cos
        __host__ void updateRotationHost() {
            data[2] = __float2half_rn(std::asinf(rotation));
            data[3] = __float2half_rn(std::acosf(rotation));
        }
        __device__ void updateRotationDevice() {
            data[2] = __float2half_rn(__asinf(rotation));
            data[3] = __float2half_rn(__acosf(rotation));
        }

        __host__ __device__ void updateCooldown() {
            data[4] = __uint2half_rn(cooldown);
        }
    };

    struct Astral: Agent {

        __host__ __device__ void updateSoundRadius() {
            data[5] = __float2half_rn(sound_radius);
        }

        __host__ __device__ void updateAbilityCooldown() {
            data[6] = __uint2half_rn(abilityCooldown);
        }

        static uint getSize() {
            return 7;
        }

        void initBase (float2 coords) {

            sound_radius_base = {10, 20};

            sight_angle = 0.5f;
            sight_radius = 10;
            speed = 3.f;

            action = Action::Nothing;
            abilityCoords = {0.f, 0.f};

            this->coords = coords;
            updateCoords();

            rotation = 0.f;
            updateRotationHost();

            cooldown = 0;
            updateCooldown();

            sound_radius = 10.f;
            updateSoundRadius();

            abilityCooldown = 0;
            updateAbilityCooldown();
        }
    }

    struct Cultist: Agent {

        __host__ __device__ void updateSoundRadius() {
            data[5] = __float2half_rn(sound_radius);
        }

        __host__ __device__ void updateAbilityCooldown() {
            data[6] = __uint2half_rn(abilityCooldown);
        }

        static uint getSize() {
            return 7;
        }

        void initBase (float2 coords) {

            sound_radius_base = 10.f;
            hearing_radius = 10.f;

            action = Action::Nothing;

            sound_radius = sound_radius_base;
            updateSoundRadius();

            abilityCooldown = 0;
            updateAbilityCooldown();

            this->coords = coords;
            updateCoords();

            rotation = 0.f;
            updateRotationHost();

            cooldown = 0;
            updateCooldown();
        }
    }

    struct Player: Agent {

         // norm (10)
        __host__ __device__ void updateSightRadius() {
            data[5] = __float2half_rn(sight_radius / 10);
        }

        // norm (10)
        __host__ __device__ void updateSoundRadius() {
            data[6] = __float2half_rn(sound_radius / 10);
        }

        // norm (10)
        __host__ __device__ void updateHearingRadius() {
            data[7] = __float2half_rn(hearing_radius / 10);
        }

        // norm (2)
        __host__ __device__ void updateMovementState() {
            data[8] = __uint2half_rn(static_cast<uint>(movementState)) / 2;
        }

        // norm (?)
        __host__ __device__ void updateSpeed() {
            data[9] = __float2half_rn(speed);
        }

        __host__ __device__ void updateMagicCooldown() {
            for (uint magic_id = 0; magic_id < (uint) Magic::Last; ++magic_id) {
                data[10 + magic_id] = __uint2half_rn(magic_cooldown[magic_id]);
            }
        }

        // norm 100
        __host__ __device__ void updateMadness() {
            data[14] = __float2half_rn(madness / 100);
        }

        static uint getSize() {
            return 15;
        }

        __host__ void initBase (float2 coords) {

            sight_angle = 0.5f;
            rotation_speed = 0.5f;

            speed_base = {1.f, 2.f, 3.f};
            sight_radius_base = 10.f;
            sound_radius_base = {1.f, 10.f, 20.f};
            hearing_radius_base = 10.f;
            magic_cooldown_base = {11, 11, 11};
            madness_high = 70.f;
            madness_low = 20.f;

            action = Action::Nothing;

            this->coords = coords;
            updateCoords();

            rotation = 0.f;
            updateRotationHost();

            cooldown = 0;
            updateCooldown();

            sight_radius = sight_radius_base;
            updateSightRadius();

            sound_radius = sound_radius_base[MovementState::Nothing];
            updateSoundRadius();

            hearing_radius = hearing_radius_base;
            updateHearingRadius();

            movementState = MovementState::Walk;
            updateMovementState();

            speed = speed_base[movementState];
            updateSpeed();

            magic_cooldown = {0, 0, 0, 0};
            updateMagicCooldown();

            madness = 0.f;
            updateMadness();
        }
    }
}