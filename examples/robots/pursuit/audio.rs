//! Original weapon sounds follow presentation events, without changing the simulation.

use super::{enemy::gun::Phase, Game};
use bevy::{
    audio::{SpatialScale, Volume},
    prelude::*,
};
use std::collections::VecDeque;

/// Bounded presentation outbox; headless simulation retains only the latest three shots.
pub(super) struct Events {
    /// Reset removes all playing sounds before projecting the new episode.
    reset: bool,
    /// World-space muzzle locations for the three most recent unrendered rounds.
    shots: VecDeque<Vec3>,
}

impl Default for Events {
    fn default() -> Self {
        Self {
            reset: true,
            shots: VecDeque::with_capacity(3),
        }
    }
}

impl Events {
    /// Record a real discharge; presentation may discard older sounds after a long frame gap.
    pub(super) fn shot(&mut self, origin: Vec3) {
        if self.shots.len() == 3 {
            self.shots.pop_front();
        }
        self.shots.push_back(origin);
    }

    /// Discard previous episode events and request playback cleanup.
    pub(super) fn reset(&mut self) {
        self.reset = true;
        self.shots.clear();
    }

    /// Read pending presentation work in game lifecycle tests.
    #[cfg(test)]
    pub(super) fn pending_shots(&self) -> usize {
        self.shots.len()
    }
}

/// Install playback; the default Bevy audio plugin owns output and decoding.
pub(super) fn install(app: &mut App) {
    app.add_systems(Startup, load).add_systems(Update, project);
}

/// Both mono WAV files are original generated assets with bounded duration and amplitude.
#[derive(Resource)]
struct Sounds {
    /// Eight hundred milliseconds of rising mechanical pitch.
    charge: Handle<AudioSource>,
    /// One hundred twenty milliseconds of a mechanical discharge.
    shot: Handle<AudioSource>,
}

/// Distinguish cancellable wind-up from already emitted shot audio.
#[derive(Component, PartialEq, Eq)]
enum Playing {
    /// Stop when a warning is interrupted or finishes.
    Charge,
    /// A finite one-shot; reset may still stop it early.
    Shot,
}

/// Load once; reset and volleys reuse the same asset handles.
fn load(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    commands.insert_resource(Sounds {
        charge: assets.load("robots/drone-charge.wav"),
        shot: assets.load("robots/drone-shot.wav"),
    });
}

/// Keep audio cleanup separate from authoritative health, timing, and projectile motion.
fn project(
    mut commands: Commands<'_, '_>,
    mut game: ResMut<'_, Game>,
    sounds: Res<'_, Sounds>,
    mut playing: Query<'_, '_, (Entity, &Playing, &mut Transform)>,
    mut was_charging: Local<'_, bool>,
) {
    let charging = matches!(game.gun.phase(), Phase::Charging { .. });
    let muzzle = game.muzzle();
    let reset = std::mem::take(&mut game.sound.reset);
    if reset {
        *was_charging = false;
    }
    for (entity, kind, mut transform) in &mut playing {
        if reset || (*kind == Playing::Charge && !charging) {
            commands.entity(entity).despawn();
        } else if *kind == Playing::Charge {
            transform.translation = muzzle;
        }
    }
    // Spatial scale maps one world metre to 0.2 audio units; physics remains in metres.
    let settings = PlaybackSettings::DESPAWN
        .with_spatial(true)
        .with_spatial_scale(SpatialScale::new(0.2))
        .with_volume(Volume::Linear(0.7));
    if charging && !*was_charging {
        commands.spawn((
            Playing::Charge,
            AudioPlayer::new(sounds.charge.clone()),
            settings,
            Transform::from_translation(muzzle),
        ));
    }
    *was_charging = charging;
    for origin in game.sound.shots.drain(..) {
        commands.spawn((
            Playing::Shot,
            AudioPlayer::new(sounds.shot.clone()),
            settings,
            Transform::from_translation(origin),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrendered_outbox_keeps_only_the_latest_volley_and_reset_clears_it() {
        let mut events = Events::default();
        for index in 0..10 {
            events.shot(Vec3::X * index as f32);
        }
        assert_eq!(events.pending_shots(), 3);
        assert_eq!(
            events.shots.iter().copied().collect::<Vec<_>>(),
            vec![Vec3::X * 7.0, Vec3::X * 8.0, Vec3::X * 9.0]
        );
        events.reset();
        assert!(events.reset);
        assert!(events.shots.is_empty());
    }
    /// Exercise playback entity ownership without opening a native audio device.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<AudioSource>()
            .init_asset_loader::<bevy::audio::AudioLoader>()
            .init_resource::<Game>()
            .add_plugins(install);
        app.update();
        app
    }

    /// Start a valid warning without advancing unrelated flight or character state.
    fn charge(app: &mut App) {
        let mut game = app.world_mut().resource_mut::<Game>();
        let origin = game.muzzle();
        let point = game.arena.position();
        let Game { gun, arena, .. } = &mut *game;
        assert!(gun
            .advance(arena, origin, super::super::sight::Contact::Visible(point))
            .is_none());
    }

    #[test]
    fn warning_cancellation_shots_and_reset_own_their_playback_entities() {
        let mut app = app();
        charge(&mut app);
        app.update();
        app.update();
        let world = app.world_mut();
        let (entity, kind, transform) = world
            .query::<(Entity, &Playing, &Transform)>()
            .single(world)
            .expect("One warning");
        assert!(*kind == Playing::Charge);
        assert_eq!(transform.translation, world.resource::<Game>().muzzle());
        // Finishing a clip cannot restart it while the same warning remains active.
        world.despawn(entity);
        app.update();
        assert_eq!(
            app.world_mut()
                .query::<&Playing>()
                .iter(app.world())
                .count(),
            0
        );
        app.world_mut().resource_mut::<Game>().gun.disable();
        app.update();
        app.world_mut().resource_mut::<Game>().reset();
        charge(&mut app);
        app.update();
        app.world_mut().resource_mut::<Game>().gun.disable();
        app.world_mut().resource_mut::<Game>().sound.shot(Vec3::X);
        app.update();
        let world = app.world_mut();
        let (kind, transform, settings) = world
            .query::<(&Playing, &Transform, &PlaybackSettings)>()
            .single(world)
            .expect("One shot; warning cancelled");
        assert!(*kind == Playing::Shot);
        assert_eq!(transform.translation, Vec3::X);
        assert!(settings.spatial);
        assert_eq!(world.resource::<Game>().sound.pending_shots(), 0);
        app.update();
        app.world_mut().resource_mut::<Game>().reset();
        app.update();
        assert_eq!(
            app.world_mut()
                .query::<&Playing>()
                .iter(app.world())
                .count(),
            0
        );
    }
    #[test]
    fn original_wave_files_decode_to_bounded_mono_samples() {
        use bevy::audio::Source;
        for (bytes, frames, milliseconds) in [
            (
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../assets/robots/drone-charge.wav"
                ))
                .as_slice(),
                17_640,
                800,
            ),
            (
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../assets/robots/drone-shot.wav"
                ))
                .as_slice(),
                2_646,
                120,
            ),
        ] {
            let source = AudioSource {
                bytes: bytes.into(),
            };
            let decoder = source.decoder();
            assert_eq!(decoder.channels(), 1);
            assert_eq!(decoder.sample_rate(), 22_050);
            assert_eq!(
                decoder.total_duration(),
                Some(std::time::Duration::from_millis(milliseconds))
            );
            let samples: Vec<_> = decoder.collect();
            assert_eq!(samples.len(), frames);
            assert!(samples.iter().any(|sample| *sample != 0));
            assert!(samples.iter().all(|sample| sample.abs() < 8_000));
        }
    }
}
