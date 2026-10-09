//! Project filtered sight into a body-mounted lens and a named HUD state.

use super::{
    hearing::{Bearing, Noise},
    sight::Contact,
    Game,
};
use bevy::prelude::*;

/// Keep private rendering types behind one viewer installation function.
pub(super) fn install(app: &mut App) {
    app.add_systems(Startup, setup)
        .add_systems(Update, project.after(super::hud::project));
}

/// Named presentation states; remembered coordinates never appear as current sight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Detection {
    /// Destroyed body or failed controller cannot sense.
    Offline,
    /// No current or recent sighting exists.
    None,
    /// The camera currently sees an exposed body point.
    Visible,
    /// A previous sighting remains within its finite lifetime.
    Remembered,
}

impl Detection {
    /// Derive presentation from authoritative health and the filtered observation.
    fn read(game: &Game) -> Self {
        if !game.combat.target().health().is_alive() || game.flight.error().is_some() {
            return Self::Offline;
        }
        match game.sight.contact() {
            Contact::Unknown => Self::None,
            Contact::Visible(_) => Self::Visible,
            Contact::Remembered { .. } => Self::Remembered,
        }
    }

    /// Text carries the state even when lens colours cannot be distinguished.
    const fn label(self) -> &'static str {
        match self {
            Self::Offline => "Drone sight: offline",
            Self::None => "Drone sight: none",
            Self::Visible => "Drone sight: visible",
            Self::Remembered => "Drone sight: last seen",
        }
    }
}

/// The camera lens uses the same pose as the sight query.
#[derive(Component)]
struct Lens;

/// One text node reports detection separately from weapon feedback.
#[derive(Component)]
pub(super) struct Status;

/// Reuse four materials for every observation and reset.
#[derive(Resource)]
struct Palette {
    /// Dark lens when sensing has stopped.
    offline: Handle<StandardMaterial>,
    /// Cyan lens while no sighting is available.
    none: Handle<StandardMaterial>,
    /// Red lens for a current sighting.
    visible: Handle<StandardMaterial>,
    /// Amber lens for remembered information.
    remembered: Handle<StandardMaterial>,
}

impl Palette {
    /// Borrow the existing material for this state.
    const fn material(&self, detection: Detection) -> &Handle<StandardMaterial> {
        match detection {
            Detection::Offline => &self.offline,
            Detection::None => &self.none,
            Detection::Visible => &self.visible,
            Detection::Remembered => &self.remembered,
        }
    }
}

/// Add a small camera lens without changing the licensed source model.
fn setup(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
) {
    let mut material = |color| {
        materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        })
    };
    let palette = Palette {
        offline: material(Color::srgb(0.05, 0.08, 0.08)),
        none: material(Color::srgb(0.10, 0.85, 0.90)),
        visible: material(Color::srgb(1.0, 0.10, 0.05)),
        remembered: material(Color::srgb(1.0, 0.55, 0.05)),
    };
    commands.spawn((
        Lens,
        Mesh3d(meshes.add(Cylinder::new(0.035, 0.014))),
        MeshMaterial3d(palette.none.clone()),
        Transform::default(),
        Visibility::Hidden,
    ));
    commands.insert_resource(palette);
}

/// Move the lens with the body and update text only when its named state changes.
fn project(
    game: Res<'_, Game>,
    palette: Res<'_, Palette>,
    mut lenses: Query<
        '_,
        '_,
        (
            &mut Transform,
            &mut MeshMaterial3d<StandardMaterial>,
            &mut Visibility,
        ),
        With<Lens>,
    >,
    mut labels: Query<'_, '_, &mut Text, With<Status>>,
    mut previous: Local<'_, Option<(Detection, Option<(Noise, Bearing)>)>>,
) {
    let detection = Detection::read(&game);
    let eye = game.eye();
    for (mut transform, mut material, mut visibility) in &mut lenses {
        *transform = Transform::from_translation(eye.translation.into())
            .with_rotation(eye.rotation * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
        material.0 = palette.material(detection).clone();
        *visibility = if game.combat.target().health().is_alive() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    let heard = if detection == Detection::Offline {
        None
    } else {
        game.hearing
            .latest()
            .map(|cue| (cue.noise(), cue.bearing()))
    };
    let state = (detection, heard);
    if *previous != Some(state) {
        *previous = Some(state);
        for mut text in &mut labels {
            let sight = detection.label();
            text.0 = if let Some((noise, bearing)) = heard {
                let noise = noise.label();
                let bearing = bearing.label();
                format!("{sight} · {noise} {bearing}")
            } else {
                sight.to_owned()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Project controlled sensor states without running flight or opening a window.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Font>()
            .init_resource::<Game>()
            .add_systems(Startup, crate::hud::setup)
            .add_plugins(install);
        app.update();
        app
    }

    /// Compare independent UI, lens, and camera projections after one update.
    fn assert_projection(app: &mut App, detection: Detection, label: &str, visibility: Visibility) {
        app.update();
        let world = app.world_mut();
        let eye = world.resource::<Game>().eye();
        let expected = world.resource::<Palette>().material(detection).clone();
        let (pose, material, shown) = world
            .query_filtered::<(&Transform, &MeshMaterial3d<StandardMaterial>, &Visibility), With<Lens>>()
            .single(world)
            .expect("One lens");
        assert_eq!(pose.translation, Vec3::from(eye.translation));
        assert_eq!(
            pose.rotation,
            eye.rotation * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
        );
        assert_eq!(material.0, expected);
        assert_eq!(*shown, visibility);
        let text = world
            .query_filtered::<&Text, With<Status>>()
            .single(world)
            .expect("One sight label");
        assert_eq!(text.0, label);
    }

    #[test]
    fn labels_and_lens_follow_filtered_sight_without_growing_assets() {
        let mut app = app();
        let meshes = app.world().resource::<Assets<Mesh>>().len();
        let materials = app.world().resource::<Assets<StandardMaterial>>().len();
        assert_projection(
            &mut app,
            Detection::None,
            "Drone sight: none",
            Visibility::Inherited,
        );
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let game = &mut *game;
            game.sight.sample(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Dir3::Z,
                Vec3::new(0.0, 9.4, 5.0),
                Duration::ZERO,
            );
        }
        assert_projection(
            &mut app,
            Detection::Visible,
            "Drone sight: visible",
            Visibility::Inherited,
        );
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let game = &mut *game;
            game.sight.sample(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Dir3::Z,
                Vec3::new(0.0, 9.4, -5.0),
                Duration::from_secs(1),
            );
        }
        assert_projection(
            &mut app,
            Detection::Remembered,
            "Drone sight: last seen",
            Visibility::Inherited,
        );
        app.world_mut()
            .resource_mut::<Game>()
            .flight
            .fail("Controller failed".to_owned());
        assert_projection(
            &mut app,
            Detection::Offline,
            "Drone sight: offline",
            Visibility::Inherited,
        );
        app.world_mut().resource_mut::<Game>().combat.crash();
        assert_projection(
            &mut app,
            Detection::Offline,
            "Drone sight: offline",
            Visibility::Hidden,
        );
        for _ in 0..10 {
            app.world_mut().resource_mut::<Game>().reset();
            assert_projection(
                &mut app,
                Detection::None,
                "Drone sight: none",
                Visibility::Inherited,
            );
        }
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), meshes);
        assert_eq!(
            app.world().resource::<Assets<StandardMaterial>>().len(),
            materials
        );
    }

    #[test]
    fn sound_labels_expire_independently_of_sight_and_disappear_offline() {
        let mut app = app();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let game = &mut *game;
            game.hearing.hear(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Vec3::new(0.0, 10.0, -3.0),
                Noise::Footstep,
            );
        }
        assert_projection(
            &mut app,
            Detection::None,
            "Drone sight: none · steps north",
            Visibility::Inherited,
        );
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let game = &mut *game;
            game.sight.sample(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Dir3::Z,
                Vec3::new(0.0, 9.4, 5.0),
                Duration::ZERO,
            );
            game.hearing.hear(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Vec3::new(3.0, 10.0, 0.0),
                Noise::Gunshot,
            );
        }
        assert_projection(
            &mut app,
            Detection::Visible,
            "Drone sight: visible · shot east",
            Visibility::Inherited,
        );
        app.world_mut()
            .resource_mut::<Game>()
            .hearing
            .advance(Duration::from_secs(2));
        assert_projection(
            &mut app,
            Detection::Visible,
            "Drone sight: visible",
            Visibility::Inherited,
        );
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let game = &mut *game;
            game.hearing.hear(
                &game.arena,
                Vec3::new(0.0, 10.0, 0.0),
                Vec3::new(3.0, 10.0, 0.0),
                Noise::Gunshot,
            );
            game.flight.fail("Controller failed".to_owned());
        }
        assert_projection(
            &mut app,
            Detection::Offline,
            "Drone sight: offline",
            Visibility::Inherited,
        );
    }
}
