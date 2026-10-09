//! Short muzzle flash, drifting smoke, and camera recoil after an accepted shot.

use super::{
    particles::{Kind, Materials},
    view::View,
    weapons::Visual,
    Game,
};
use bevy::prelude::*;
use std::time::Duration;

/// Read the current animated gun transform before normal transform propagation.
pub(super) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        project
            .after(super::weapons::project)
            .before(TransformSystems::Propagate),
    );
}

/// Rejected shots and resets cannot create another flash or camera kick.
fn project(
    mut commands: Commands<'_, '_>,
    game: Res<'_, Game>,
    materials: Res<'_, Materials>,
    guns: Query<'_, '_, (&Visual, &Transform)>,
    mut view: Option<ResMut<'_, View>>,
    mut previous: Local<'_, Option<u8>>,
) {
    let rounds = game.combat.rounds();
    let fired = game.combat.is_armed() && previous.is_some_and(|previous| rounds < previous);
    *previous = Some(rounds);
    if !fired {
        return;
    }
    if let Some(view) = view.as_mut() {
        view.recoil();
    }
    let Some((_, gun)) = guns.iter().find(|(kind, _)| **kind == Visual::Held) else {
        return;
    };
    let muzzle = super::weapon_pose::muzzle(gun);
    materials.spawn(
        &mut commands,
        muzzle,
        Vec3::ZERO,
        Kind::Flash,
        Duration::from_millis(45),
        0.045,
    );
    materials.spawn(
        &mut commands,
        muzzle,
        gun.rotation * Vec3::NEG_X * 0.15 + Vec3::Y * 0.2,
        Kind::Smoke,
        Duration::from_millis(350),
        0.015,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{firing::Action, particles};

    #[test]
    fn only_a_new_discharge_emits_at_the_current_barrel() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<Image>()
            .init_asset::<StandardMaterial>()
            .init_resource::<Game>()
            .init_resource::<View>()
            .add_systems(Startup, particles::setup);
        install(&mut app);
        let gun = Transform::from_xyz(1.0, 2.0, 3.0).with_rotation(Quat::from_rotation_y(0.5));
        app.world_mut().spawn((Visual::Held, gun));
        app.world_mut().resource_mut::<Game>().act(Action::PickUp);
        app.update();
        app.world_mut()
            .resource_mut::<Game>()
            .act(Action::Fire(Dir3::Y));
        app.update();
        let world = app.world_mut();
        let muzzle = gun.transform_point(Vec3::new(-0.372, 0.12, 0.0));
        let mut particles = world.query_filtered::<&Transform, With<particles::Particle>>();
        assert_eq!(particles.iter(world).count(), 2);
        assert!(particles.iter(world).all(|pose| pose.translation == muzzle));
        world.resource_mut::<Game>().act(Action::Fire(Dir3::Y));
        app.update();
        assert_eq!(particles.iter(app.world()).count(), 2);
        app.world_mut().resource_mut::<Game>().reset();
        app.update();
        assert_eq!(particles.iter(app.world()).count(), 2);
    }

    #[test]
    fn missing_visual_or_camera_does_not_interrupt_the_game() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<Image>()
            .init_asset::<StandardMaterial>()
            .init_resource::<Game>()
            .add_systems(Startup, particles::setup);
        install(&mut app);
        app.world_mut().resource_mut::<Game>().act(Action::PickUp);
        app.update();
        app.world_mut()
            .resource_mut::<Game>()
            .act(Action::Fire(Dir3::Y));
        app.update();
        let world = app.world_mut();
        assert_eq!(world.query::<&particles::Particle>().iter(world).count(), 0);
        assert_eq!(world.resource::<Game>().combat.rounds(), 11);
    }
}
