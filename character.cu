#include "character.h"
#include "utils.h"

namespace Character {

    __host__ void Astral::init (float2 coords) {

        action = Action::Nothing;

        this->coords = coords;
        rotation = 0.f;
        cooldown = 0;
        speed = 0.375f;
        sound_radius = 65.f;
        ability_cooldown = 0;
        scream_cooldown = 0;
        see_player = false;
        attack_player = false;

        updateHost();
    }

    __host__ void Cultist::init (float2 coords) {

        action = Action::Nothing;

        this->coords = coords;
        rotation = 0.f;
        cooldown = 0;
        noise = 0.f;
        sound_radius = 65.f;
        ability_cooldown = 0;
        ability_coords = {0.f, 0.f};

        updateHost();
    }

    __host__ void Player::init (float2 coords) {

        action = Action::Nothing;

        this->coords = coords;
        rotation = 0.f;
        cooldown = 0;
        sight_radius = 300.f;
        sound_radius = 0.f;
        hearing_radius = 65.f;
        movement_state = MovementState::Walk;
        speed = 0.375f;
        magic_cooldown_see = 0;
        magic_cooldown_invisibility = 0;
        magic_cooldown_inaudibility = 0;
        noise = 0.f;
        door = (uint) -1;
        window = (uint) -1;
        stamina = 100.f;
        shadow_coords = {(uint) std::floor(coords.x), (uint) std::floor(coords.y)};
        max_x = coords.x;

        updateHost();
    }
}
