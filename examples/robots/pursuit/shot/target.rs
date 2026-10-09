//! Nearest body and rotor intersections with wall occlusion.
//!
//! Rapier 0.36 re-exports Parry 0.31.1. `RayCast::cast_local_ray` uses a unit
//! direction so its time of impact is distance in metres. `solid=true` reports
//! zero for a ray starting inside a hitbox.
//! <https://docs.rs/parry3d/0.31.1/parry3d/query/trait.RayCast.html>

use super::super::{
    arena::Arena,
    combat::{
        health::{Damage, DroneHealth, RotorHealth},
        pistol::{FireError, Pistol},
    },
};
use super::aim::{valid_position, Aim};
use bevy::math::{Quat, Vec3, Vec4};
use bevy_gym::robots::DroneMotor;
use rapier3d::{
    parry::{
        query::RayCast,
        shape::{Ball, Cuboid},
    },
    prelude::{Ray, Vector},
};

/// Validated drone pose and damage state, independent of rendering.
#[derive(Debug)]
pub(crate) struct Target {
    /// World-space body centre, in metres.
    position: Vec3,
    /// Normalized body-to-world rotation.
    rotation: Quat,
    /// Only accepted geometry hits can mutate this target's health.
    health: DroneHealth,
}

/// Invalid target poses never enter persistent shot state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvalidTarget {
    /// Centre is non-finite or outside ±1,000 metres on any axis.
    Position,
    /// Quaternion is zero or contains a non-finite component.
    Rotation,
}

/// Identifies the first live part intersected by the shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    /// Central body, excluding the separately aimable rotor spheres.
    Body,
    /// One named actuator weak point.
    Rotor(DroneMotor),
}

/// Accepted shots always consume ammunition, including misses and blocked shots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Shot {
    /// No solid geometry was reached within range.
    Miss {
        /// World-space endpoint, thirty metres from the muzzle.
        point: Vec3,
    },
    /// Static geometry was closest; exact distance ties also stop here.
    Wall {
        /// First world-space obstruction point.
        point: Vec3,
    },
    /// One live drone part received the hit.
    Hit {
        /// Nearest live body or rotor hitbox.
        part: Part,
        /// World-space impact point.
        point: Vec3,
        /// Health transition for physics and presentation.
        damage: Damage,
    },
}

impl TryFrom<(Vec3, Quat)> for Target {
    type Error = InvalidTarget;

    fn try_from((position, rotation): (Vec3, Quat)) -> Result<Self, Self::Error> {
        if !valid_position(position) {
            return Err(InvalidTarget::Position);
        }
        let scale = Vec4::from_array(rotation.to_array()).abs().max_element();
        if !rotation.is_finite() || scale <= 0.0 {
            return Err(InvalidTarget::Rotation);
        }
        Ok(Self {
            position,
            rotation: (rotation / scale).normalize(),
            health: DroneHealth::default(),
        })
    }
}

impl Target {
    /// Replace a validated pose while retaining body and rotor damage.
    pub(crate) fn move_to(&mut self, position: Vec3, rotation: Quat) -> Result<(), InvalidTarget> {
        let pose = Self::try_from((position, rotation))?;
        self.position = pose.position;
        self.rotation = pose.rotation;
        Ok(())
    }

    /// Physical contact uses the same absorbing damage transition as body hits.
    pub(crate) const fn crash(&mut self) -> Damage {
        self.health.crash()
    }

    /// Read the authoritative world-space body centre.
    pub(crate) const fn position(&self) -> Vec3 {
        self.position
    }

    /// Read the normalized body-to-world rotation.
    pub(crate) const fn rotation(&self) -> Quat {
        self.rotation
    }

    /// Rendering may inspect health but cannot forge a damage transition.
    pub(crate) const fn health(&self) -> &DroneHealth {
        &self.health
    }

    /// Transform the existing motor centre into world coordinates.
    pub(crate) fn rotor_centre(&self, motor: DroneMotor) -> Vec3 {
        self.position + self.rotation * offset(motor)
    }

    /// Query the visible aim point without spending ammunition or changing health.
    pub(crate) fn aim_point(&self, arena: &Arena, aim: Aim) -> Vec3 {
        let target = self
            .nearest(aim)
            .map_or(Aim::RANGE, |(_, distance)| distance);
        let wall = arena
            .obstruction(aim.origin(), aim.point(Aim::RANGE))
            .unwrap_or(Aim::RANGE);
        aim.point(target.min(wall))
    }

    /// Spend one round, query nearest geometry, and apply at most one health transition.
    pub(crate) fn shoot(
        &mut self,
        arena: &Arena,
        pistol: &mut Pistol,
        aim: Aim,
    ) -> Result<Shot, FireError> {
        pistol.fire()?;
        let hit = self.nearest(aim);
        let distance = hit.map_or(Aim::RANGE, |(_, distance)| distance);
        // Static geometry wins ties, including rays starting inside a wall.
        if let Some(blocked) = arena
            .obstruction(aim.origin(), aim.point(Aim::RANGE))
            .filter(|blocked| *blocked <= distance)
        {
            return Ok(Shot::Wall {
                point: aim.point(blocked),
            });
        }
        let Some((part, distance)) = hit else {
            return Ok(Shot::Miss {
                point: aim.point(Aim::RANGE),
            });
        };
        let damage = match part {
            Part::Body => self.health.hit_body(),
            Part::Rotor(motor) => self.health.hit_rotor(motor),
        };
        Ok(Shot::Hit {
            part,
            point: aim.point(distance),
            damage,
        })
    }

    /// Test five fixed-size shapes in body coordinates without allocating a collection.
    fn nearest(&self, aim: Aim) -> Option<(Part, f32)> {
        if !self.health.is_alive() {
            return None;
        }
        let inverse = self.rotation.inverse();
        let origin = inverse * (aim.origin() - self.position);
        let direction = inverse * *aim.direction();
        let ray = Ray::new(
            Vector::from_array(origin.to_array()),
            Vector::from_array(direction.to_array()),
        );
        let body = Cuboid::new(Vector::new(0.17, 0.104, 0.18));
        let mut nearest = body
            .cast_local_ray(&ray, Aim::RANGE, true)
            .map(|distance| (Part::Body, distance));
        for motor in DroneMotor::ALL {
            if self.health.rotor(motor) == RotorHealth::Destroyed {
                continue;
            }
            let ray = Ray::new(
                Vector::from_array((origin - offset(motor)).to_array()),
                ray.dir,
            );
            if let Some(distance) = Ball::new(0.12).cast_local_ray(&ray, Aim::RANGE, true) {
                if nearest.is_none_or(|(_, previous)| distance < previous) {
                    nearest = Some((Part::Rotor(motor), distance));
                }
            }
        }
        nearest
    }
}

/// Motor centres in metres, matching the qualified drone's body-coordinate convention.
const fn offset(motor: DroneMotor) -> Vec3 {
    match motor {
        DroneMotor::FrontLeft => Vec3::new(-0.2505, 0.0875, -0.2606),
        DroneMotor::FrontRight => Vec3::new(0.2505, 0.0875, -0.2606),
        DroneMotor::RearRight => Vec3::new(0.2505, 0.0875, 0.2606),
        DroneMotor::RearLeft => Vec3::new(-0.2505, 0.0875, 0.2606),
    }
}
