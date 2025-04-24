#include "utils.h"

namespace Character {
    enum class Type : uint {
        Player,
        Astral,
        Cultist,
        Flying
    };

    /*
    struct Flying {

        Type getType() {
            return Type::Flying;
        }

        const float lighting_radius;
        const uint positions_num;
        const uint2* positions;
        const curandState* seed;

        float2 coords;
        uint2 next_position;

        static uint getSize() {
            return 5;
        }

        __device__ void getNextPosition();

        void initBase (float2 coords);
    };
    */

    struct Agent {

        half* data;

        const float rotation_speed;

        float2 coords;
        float rotation;
        uint cooldown;

        __host__ __device__ void updateCoords();
        __host__ void updateRotationHost();
        __device__ void updateRotationDevice();
        __host__ __device__ void updateCooldown();
    };

    struct Astral: Agent {

        Type getType() {
            return Type::Astral;
        }

        enum class Action : uint {
            Nothing,

            GoForward,

            TurnRightFull,
            TurnRightHalf,
            TurnLeftFull,
            TurnLeftHalf,

            Ability
        };

        enum class Sound : uint {
            Walk,
            Ability,
            Last = Ability
        };

        const float sound_radius_base[Sound::Last];

        const float sight_angle;
        const float sight_radius;
        const float speed;

        Action action;
        float2 abilityCoords;

        float sound_radius;
        uint abilityCooldown;

        __host__ __device__ void updateSoundRadius();
        __host__ __device__ void updateAbilityCooldown();

        static uint getSize();

        void initBase (float2 coords);
    };

    struct Cultist: Agent {

        Type getType() {
            return Type::Cultist;
        }

        enum class Action : uint {
            Nothing,

            GoForward,

            TurnRightFull,
            TurnRightHalf,
            TurnLeftFull,
            TurnLeftHalf,

            Ability
        };

        const float sound_radius_base;
        const float hearing_radius;

        Action action;

        float sound_radius;
        uint abilityCooldown;

        __host__ __device__ void updateSoundRadius();
        __host__ __device__ void updateAbilityCooldown();

        static uint getSize();

        void initBase (float2 coords);
    };

    struct Player: Agent {

        Type getType() {
            return Type::Player;
        }

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
            Run,
            Last = Run
        };

        enum class Magic : uint {
            See,
            // Light,
            Invisibility,
            Inaudibility,
            Last = Inaudibility
        };

        const float sight_angle;

        const float speed_base[MovementState::Last];
        const float sight_radius_base;
        const float sound_radius_base[MovementState::Last];
        const float hearing_radius_base;
        const uint magic_cooldown_base[Magic::Last];
        const float madness_high;
        const float madness_low;

        Action action;
        float sight_radius;
        float sound_radius;
        float hearing_radius;
        MovementState state;
        float speed;
        uint magic_cooldown[Magic::Last];
        float madness;

        __host__ __device__ void updateSightRadius();
        __host__ __device__ void updateSoundRadius();
        __host__ __device__ void updateHearingRadius();
        __host__ __device__ void updateMovementState();
        __host__ __device__ void updateSpeed();
        __host__ __device__ void updateMagicCooldown();
        __host__ __device__ void updateMadness();

        static uint getSize();

        void initBase (float2 coords);
    }
}