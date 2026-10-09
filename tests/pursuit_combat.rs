//! Combat transitions shared by the player and future policy-controlled opponent.
#![cfg(feature = "robots")]

#[path = "../examples/robots/pursuit/combat.rs"]
mod combat;

use bevy::math::Vec3;
use bevy_gym::robots::DroneMotor;
use combat::{
    health::{Damage, DroneHealth, RotorHealth},
    pistol::{FireError, PickupError, Pistol},
};
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// Every named rotor has the same two-hit weak point without killing the body.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rotor_destruction_emits_once_for_each_motor() {
    let mut health = DroneHealth::default();
    for motor in DroneMotor::ALL {
        assert_eq!(health.rotor(motor), RotorHealth::Intact);
        assert_eq!(health.hit_rotor(motor), Damage::Hit);
        assert_eq!(health.rotor(motor), RotorHealth::Damaged);
        assert_eq!(health.hit_rotor(motor), Damage::RotorDestroyed(motor));
        assert_eq!(health.rotor(motor), RotorHealth::Destroyed);
        assert_eq!(health.hit_rotor(motor), Damage::Ignored);
    }
    assert!(health.is_alive());
    assert_eq!(health.body_hits_remaining(), 6);
}

/// A six-hit body dies once, and dead actors cannot emit more damage transitions.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn body_death_is_absorbing_and_reset_repairs_every_part() {
    let mut health = DroneHealth::default();
    health.hit_rotor(DroneMotor::FrontLeft);
    for remaining in (1..6).rev() {
        assert_eq!(health.hit_body(), Damage::Hit);
        assert_eq!(health.body_hits_remaining(), remaining);
    }
    assert_eq!(health.hit_body(), Damage::Destroyed);
    assert!(!health.is_alive());
    assert_eq!(health.body_hits_remaining(), 0);
    assert_eq!(health.hit_body(), Damage::Ignored);
    assert_eq!(health.hit_rotor(DroneMotor::FrontLeft), Damage::Ignored);
    assert_eq!(health.crash(), Damage::Ignored);
    health = DroneHealth::default();
    assert!(health.is_alive());
    assert_eq!(health.body_hits_remaining(), 6);
    assert!(DroneMotor::ALL
        .into_iter()
        .all(|motor| health.rotor(motor) == RotorHealth::Intact));
}

/// A crash shares the same one-time death transition as exhausted body health.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fatal_crash_stops_later_damage() {
    let mut health = DroneHealth::default();
    assert_eq!(health.crash(), Damage::Destroyed);
    assert_eq!(health.crash(), Damage::Ignored);
    assert_eq!(health.hit_body(), Damage::Ignored);
    assert_eq!(health.body_hits_remaining(), 0);
}

/// Pickup checks finite coordinates, inclusive distance, and existing ownership.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pickup_requires_proximity_and_cannot_refill_an_owned_pistol() {
    let mut pistol = Pistol::default();
    assert!(!pistol.is_armed());
    assert_eq!(pistol.rounds(), 0);
    assert_eq!(pistol.fire(), Err(FireError::Unarmed));
    for invalid in [Vec3::NAN, Vec3::INFINITY, Vec3::splat(f32::MAX)] {
        assert_eq!(pistol.pick_up(invalid), Err(PickupError::InvalidPosition));
    }
    assert_eq!(
        pistol.pick_up(Pistol::LOCATION + Vec3::X * 1.501),
        Err(PickupError::TooFar)
    );
    assert_eq!(pistol.pick_up(Pistol::LOCATION + Vec3::X * 1.5), Ok(()));
    assert!(pistol.is_armed());
    assert_eq!(pistol.rounds(), 12);
    assert_eq!(pistol.fire(), Ok(()));
    assert_eq!(
        pistol.pick_up(Pistol::LOCATION),
        Err(PickupError::AlreadyOwned)
    );
    assert_eq!(pistol.rounds(), 11);
}

/// Cooldown and ammunition rejection preserve the magazine, including exact expiry.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn firing_consumes_one_round_and_respects_cooldown() {
    let mut pistol = Pistol::default();
    pistol.advance(Duration::MAX);
    pistol.pick_up(Pistol::LOCATION).unwrap();
    pistol.fire().unwrap();
    assert_eq!(pistol.fire(), Err(FireError::CoolingDown));
    pistol.advance(Duration::from_millis(249));
    assert_eq!(pistol.fire(), Err(FireError::CoolingDown));
    assert_eq!(pistol.rounds(), 11);
    pistol.advance(Duration::from_millis(1));
    for remaining in (0..11).rev() {
        pistol.fire().unwrap();
        assert_eq!(pistol.rounds(), remaining);
        pistol.advance(Duration::MAX);
    }
    assert_eq!(pistol.fire(), Err(FireError::Empty));
    assert_eq!(pistol.rounds(), 0);
    pistol = Pistol::default();
    assert!(!pistol.is_armed());
    assert_eq!(pistol.fire(), Err(FireError::Unarmed));
}

/// Each rotor hit leaves every other rotor and body hit count unchanged.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn weak_points_are_independent() {
    for target in DroneMotor::ALL {
        let mut health = DroneHealth::default();
        health.hit_rotor(target);
        for motor in DroneMotor::ALL {
            let expected = if motor == target {
                RotorHealth::Damaged
            } else {
                RotorHealth::Intact
            };
            assert_eq!(health.rotor(motor), expected);
        }
        assert_eq!(health.body_hits_remaining(), 6);
    }
}

/// Invalid positions precede ownership rejection; an empty magazine precedes cooldown.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rejection_order_and_zero_elapsed_time_are_stable() {
    let mut pistol = Pistol::default();
    pistol.pick_up(Pistol::LOCATION).unwrap();
    assert_eq!(pistol.pick_up(Vec3::NAN), Err(PickupError::InvalidPosition));
    assert_eq!(
        pistol.pick_up(Vec3::splat(100.0)),
        Err(PickupError::AlreadyOwned)
    );
    for _ in 0..12 {
        pistol.advance(Duration::from_millis(250));
        pistol.fire().unwrap();
    }
    pistol.advance(Duration::ZERO);
    assert_eq!(pistol.fire(), Err(FireError::Empty));
}
