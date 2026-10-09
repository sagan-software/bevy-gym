//! Original drone weapon geometry, warning light, projectiles, and robot health display.

use super::{enemy::gun::Phase, sight::Contact, Game};
use bevy::prelude::*;

/// Install presentation without letting it control firing or damage.
pub(super) fn install(app: &mut App) {
    app.add_systems(Startup, setup)
        .add_systems(Update, (project, project_status));
}

/// The shared HUD owns this text node's placement.
#[derive(Component)]
pub(super) struct Status;

/// Every role has one preallocated mesh; no projectile spawns render assets.
#[derive(Component)]
enum Visual {
    /// Original under-body barrel, aimed only from current sight.
    Barrel,
    /// Amber wind-up or red volley light at the muzzle.
    Warning,
    /// One of the three bounded projectile projections.
    Bolt(usize),
}

/// Named display states omit timer values that do not affect text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Alert {
    /// No weapon opportunity.
    Idle,
    /// The robot can still reach cover before firing.
    Charging,
    /// A volley is in progress.
    Firing,
    /// The drone cannot immediately shoot again.
    Cooling,
    /// The gun is disabled until reset.
    Offline,
}

impl Alert {
    /// Derive display state from the authoritative phase.
    const fn from_phase(phase: Phase) -> Self {
        match phase {
            Phase::Idle => Self::Idle,
            Phase::Charging { .. } => Self::Charging,
            Phase::Volley { .. } => Self::Firing,
            Phase::Cooling { .. } => Self::Cooling,
            Phase::Disabled => Self::Offline,
        }
    }

    /// Status names tell the player whether a shot is imminent.
    const fn label(self) -> &'static str {
        match self {
            Self::Idle => "Drone idle",
            Self::Charging => "Take cover",
            Self::Firing => "Drone firing",
            Self::Cooling => "Drone cooling",
            Self::Offline => "Drone offline",
        }
    }
}

/// Reuse warning and projectile materials across every volley and reset.
#[derive(Resource)]
struct Materials {
    /// Amber warning before any shot is permitted.
    warning: Handle<StandardMaterial>,
    /// Bright red volley and projectile surface.
    firing: Handle<StandardMaterial>,
}

/// Allocate a barrel, warning light, and exactly three projectile meshes.
fn setup(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
) {
    let metal = materials.add(Color::srgb(0.08, 0.10, 0.12));
    let warning = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.65, 0.05),
        unlit: true,
        ..default()
    });
    let firing = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.12, 0.03),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Visual::Barrel,
        Mesh3d(meshes.add(Cuboid::new(0.12, 0.07, 0.25))),
        MeshMaterial3d(metal),
        Transform::default(),
        Visibility::Hidden,
    ));
    let sphere = meshes.add(Sphere::new(1.0));
    commands.spawn((
        Visual::Warning,
        Mesh3d(sphere.clone()),
        MeshMaterial3d(warning.clone()),
        Transform::default(),
        Visibility::Hidden,
    ));
    for index in 0..3 {
        commands.spawn((
            Visual::Bolt(index),
            Mesh3d(sphere.clone()),
            MeshMaterial3d(firing.clone()),
            Transform::default(),
            Visibility::Hidden,
        ));
    }
    commands.insert_resource(Materials { warning, firing });
}

/// Project the simulation without allocating meshes or changing its observations.
fn project(
    game: Res<'_, Game>,
    materials: Res<'_, Materials>,
    mut visuals: Query<
        '_,
        '_,
        (
            &Visual,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
    >,
) {
    let phase = game.gun.phase();
    let alert = Alert::from_phase(phase);
    let muzzle = game.muzzle();
    let alive = game.combat.target().health().is_alive();
    for (role, mut transform, mut visibility, mut material) in &mut visuals {
        let shown = match role {
            Visual::Barrel => {
                let rotation = barrel_rotation(&game, muzzle);
                *transform = Transform::from_translation(muzzle + rotation * Vec3::Z * 0.125)
                    .with_rotation(rotation);
                alive
            }
            Visual::Warning => {
                let radius = if let Phase::Charging { remaining } = phase {
                    // Remaining seconds divided by the 0.8-second warning gives a unitless fraction.
                    0.04_f32.mul_add(1.0 - remaining.as_secs_f32() / 0.8, 0.04)
                } else {
                    0.08
                };
                *transform = Transform::from_translation(muzzle).with_scale(Vec3::splat(radius));
                material.0 = if alert == Alert::Charging {
                    materials.warning.clone()
                } else {
                    materials.firing.clone()
                };
                alive && matches!(alert, Alert::Charging | Alert::Firing)
            }
            Visual::Bolt(index) => game
                .projectiles
                .iter()
                .nth(*index)
                .is_some_and(|projectile| {
                    *transform = Transform::from_translation(projectile.position())
                        .with_scale(Vec3::splat(0.055));
                    true
                }),
        };
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Aim at a current sight measurement; memory cannot turn the visible barrel.
fn barrel_rotation(game: &Game, muzzle: Vec3) -> Quat {
    match game.sight.contact() {
        Contact::Visible(point) => {
            Transform::from_translation(muzzle)
                .looking_at(point, Vec3::Y)
                .rotation
        }
        Contact::Unknown | Contact::Remembered { .. } => game.combat.target().rotation(),
    }
}

/// Format status only when health or the named weapon phase changes.
fn project_status(
    game: Res<'_, Game>,
    mut labels: Query<'_, '_, &mut Text, With<Status>>,
    mut combat: Query<
        '_,
        '_,
        &mut Node,
        Or<(With<super::hud::Status>, With<super::hud::ActionButton>)>,
    >,
    mut previous: Local<'_, Option<(u8, Alert)>>,
) {
    let state = (
        game.robot_health.hits_remaining(),
        Alert::from_phase(game.gun.phase()),
    );
    if *previous != Some(state) {
        *previous = Some(state);
        // Disabled characters cannot collect or fire; reset restores these controls.
        for mut node in &mut combat {
            node.display = if state.0 == 0 {
                Display::None
            } else {
                Display::Flex
            };
        }
        for mut text in &mut labels {
            text.0 = if state.0 == 0 {
                "Robot disabled · R to restart".to_owned()
            } else {
                let health = state.0;
                let status = state.1.label();
                format!("Robot {health}/3 · {status}")
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same render projections run without a window or GPU.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Font>()
            .init_resource::<Game>()
            .add_systems(Startup, super::super::hud::setup)
            .add_plugins(install);
        app.update();
        app
    }

    /// Feed a current sighting through the actual weapon timing.
    fn tick_gun(app: &mut App) {
        let mut game = app.world_mut().resource_mut::<Game>();
        let origin = game.muzzle();
        let point = game.arena.position();
        let Game {
            gun,
            arena,
            projectiles,
            ..
        } = &mut *game;
        if let Some(projectile) = gun.advance(arena, origin, Contact::Visible(point)) {
            projectiles.launch(projectile).expect("Bounded volley");
        }
    }

    /// Check warning, live projectile count, and text against independent expected values.
    fn assert_view(app: &mut App, label: &str, warning: bool, bolts: usize) {
        app.update();
        let world = app.world_mut();
        let text = world
            .query_filtered::<&Text, With<Status>>()
            .single(world)
            .expect("Status");
        assert_eq!(text.0, label);
        let mut shown_bolts = 0;
        for (role, visibility) in world.query::<(&Visual, &Visibility)>().iter(world) {
            match role {
                Visual::Warning => assert_eq!(*visibility == Visibility::Inherited, warning),
                Visual::Bolt(_) => shown_bolts += usize::from(*visibility == Visibility::Inherited),
                Visual::Barrel => {}
            }
        }
        assert_eq!(shown_bolts, bolts);
    }

    #[test]
    fn warning_volley_cooldown_death_and_reset_reuse_render_assets() {
        let mut app = app();
        let meshes = app.world().resource::<Assets<Mesh>>().len();
        let materials = app.world().resource::<Assets<StandardMaterial>>().len();
        assert_view(&mut app, "Robot 3/3 · Drone idle", false, 0);
        tick_gun(&mut app);
        assert_view(&mut app, "Robot 3/3 · Take cover", true, 0);
        for _ in 0..40 {
            tick_gun(&mut app);
        }
        assert_view(&mut app, "Robot 3/3 · Drone firing", true, 1);
        for _ in 0..12 {
            tick_gun(&mut app);
        }
        assert_view(&mut app, "Robot 3/3 · Drone cooling", false, 3);
        app.world_mut().resource_mut::<Game>().gun.disable();
        app.world_mut().resource_mut::<Game>().combat.crash();
        assert_view(&mut app, "Robot 3/3 · Drone offline", false, 3);
        let world = app.world_mut();
        for (role, visibility) in world.query::<(&Visual, &Visibility)>().iter(world) {
            if matches!(role, Visual::Barrel) {
                assert_eq!(*visibility, Visibility::Hidden);
            }
        }
        app.world_mut().resource_mut::<Game>().reset();
        assert_view(&mut app, "Robot 3/3 · Drone idle", false, 0);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), meshes);
        assert_eq!(
            app.world().resource::<Assets<StandardMaterial>>().len(),
            materials
        );
    }

    #[test]
    fn robot_death_names_the_reset_action() {
        let mut app = app();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            for _ in 0..200 {
                game.step(super::super::Movement::Forward);
                if matches!(game.gun.phase(), Phase::Charging { .. }) {
                    break;
                }
            }
        }
        assert_view(&mut app, "Robot 3/3 · Take cover", true, 0);
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            for _ in 0..300 {
                game.step(super::super::Movement::Idle);
            }
        }
        assert_view(&mut app, "Robot disabled · R to restart", false, 0);
        assert_combat_display(&mut app, Display::None);
        app.world_mut().resource_mut::<Game>().reset();
        assert_view(&mut app, "Robot 3/3 · Drone idle", false, 0);
        assert_combat_display(&mut app, Display::Flex);
    }

    /// The pickup prompt and both weapon buttons follow the character's life state.
    fn assert_combat_display(app: &mut App, expected: Display) {
        let world = app.world_mut();
        let mut query = world.query_filtered::<&Node, Or<(
            With<super::super::hud::Status>,
            With<super::super::hud::ActionButton>,
        )>>();
        assert_eq!(query.iter(world).count(), 3);
        assert!(query.iter(world).all(|node| node.display == expected));
    }
}
