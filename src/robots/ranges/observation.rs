//! Six optional nearest-solid readings from one physical snapshot.

use super::{DroneRangeDirection, DroneRangeDistance};

/// Body-centred rays capped at ten metres, with the drone collider excluded.
///
/// `None` means no hit within the inclusive range. `Some` may contain zero
/// when a ray starts inside another solid. A ten-metre hit remains `Some`.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneHover;
/// let unchecked_direction = DroneHover::default().ranges().distance(6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneRanges {
    /// Closest hits in `DroneRangeDirection::ALL` order.
    distances: [Option<DroneRangeDistance>; 6],
}

impl DroneRanges {
    /// Read one nearest-solid distance without exposing collision geometry.
    #[must_use]
    pub const fn distance(self, direction: DroneRangeDirection) -> Option<DroneRangeDistance> {
        let [forward, back, left, right, up, down] = self.distances;
        match direction {
            DroneRangeDirection::Forward => forward,
            DroneRangeDirection::Back => back,
            DroneRangeDirection::Left => left,
            DroneRangeDirection::Right => right,
            DroneRangeDirection::Up => up,
            DroneRangeDirection::Down => down,
        }
    }
}

/// Measure current collider poses directly, including before the first solver step.
///
/// Parry 0.31.1 uses unit ray directions to express time of impact in metres:
/// <https://docs.rs/parry3d/0.31.1/parry3d/query/trait.RayCast.html#method.cast_ray>.
/// Scanning six rays over N colliders costs O(N) time and O(1) auxiliary space.
/// No broad-phase refresh or physical step occurs during a read.
pub(in crate::robots) fn measure(
    world: &rapier3d::prelude::PhysicsWorld,
    excluded: rapier3d::prelude::ColliderHandle,
    body: crate::robots::DroneObservation,
) -> DroneRanges {
    use rapier3d::prelude::{Ray, Vector};
    let distances = DroneRangeDirection::ALL.map(|direction| {
        // A dimensionless unit direction makes ray time of impact a distance in metres.
        let unit = (body.orientation() * direction.unit()).normalize();
        let ray = Ray::new(
            Vector::from_array(body.position().to_array()),
            Vector::from_array(unit.to_array()),
        );
        world
            .colliders
            .iter()
            .filter(|(handle, collider)| *handle != excluded && !collider.is_sensor())
            .filter_map(|(_, collider)| {
                collider.shape().cast_ray(
                    collider.position(),
                    &ray,
                    DroneRangeDistance::MAX.metres(),
                    true,
                )
            })
            .min_by(f32::total_cmp)
            .map(|metres| {
                DroneRangeDistance::try_from(metres)
                    .expect("private solver returns finite bounded ray distances")
            })
    });
    DroneRanges { distances }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{Quat, Vec3};
    use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, Vector};

    /// Private geometry filters cannot expose the body or sensor volumes as solid hits.
    #[test]
    fn body_frame_rays_exclude_self_and_sensor_colliders() {
        let mut world = PhysicsWorld::default();
        let body = crate::robots::DroneHover::default().observation();
        let (_, excluded) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 2.0, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, 2.0, -1.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5).sensor(true),
        );
        let empty = measure(&world, excluded, body);
        for direction in DroneRangeDirection::ALL {
            assert!(empty.distance(direction).is_none());
        }
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(2.0, 2.0, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
        let mut rotated = body;
        // A -pi/2 yaw rotates local forward (-Z) into world +X.
        rotated.orientation = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        let reading = measure(&world, excluded, rotated);
        let forward = reading
            .distance(DroneRangeDirection::Forward)
            .expect("world +X wall");
        assert!((forward.metres() - 1.5).abs() < 0.000_01);
        assert!(reading.distance(DroneRangeDirection::Right).is_none());
        assert_eq!(body.orientation(), Quat::IDENTITY);
        assert_eq!(body.position(), Vec3::new(0.0, 2.0, 0.0));
    }
}
