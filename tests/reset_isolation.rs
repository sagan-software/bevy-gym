//! Automatic resets remain local to their environment entity and seed schedule.

use bevy::prelude::{
    App, Entity, FixedUpdate, IntoScheduleConfigs, Messages, MinimalPlugins, ResMut, World,
};
use bevy_gym::{
    ActionRequest, ActionResponse, BevyGymPlugin, CurrentObservation, Env, EnvStats, EpisodeStatus,
    GymSet, Reset, Step, TransitionEvent,
};

/// Counter environments with different const parameters have distinct plugin types.
#[derive(Default)]
struct Counter<const END_AFTER: usize> {
    /// Steps taken in the current episode.
    steps: usize,
    /// Seed supplied by the most recent reset.
    seed: Option<u64>,
}

impl<const END_AFTER: usize> Env for Counter<END_AFTER> {
    type Action = ();
    type Observation = (usize, Option<u64>);
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.steps = 0;
        self.seed = seed;
        Reset {
            observation: (self.steps, self.seed),
            info: (),
        }
    }

    fn step(&mut self, (): Self::Action) -> Step<Self::Observation> {
        self.steps += 1;
        Step {
            observation: (self.steps, self.seed),
            reward: 1.0,
            status: if self.steps >= END_AFTER {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

/// Spawn two distinct environment types whose local environment IDs both start at zero.
fn app_with_second(second: BevyGymPlugin<Counter<2>>) -> (App, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins((
        BevyGymPlugin::new(|_| Counter::<1>::default())
            .with_reset_seeds(|_, episode| Some(100 + episode)),
        second,
    ));
    app.update();
    let first = app
        .world_mut()
        .resource_mut::<Messages<ActionRequest<Counter<1>>>>()
        .drain()
        .next()
        .expect("first plugin requested an action")
        .entity;
    let second = app
        .world_mut()
        .resource_mut::<Messages<ActionRequest<Counter<2>>>>()
        .drain()
        .next()
        .expect("second plugin requested an action")
        .entity;
    (app, first, second)
}

/// Finish only the first environment; the second receives no action.
fn finish_first(app: &mut App, entity: Entity) {
    app.world_mut()
        .write_message(ActionResponse::<Counter<1>> { entity, action: () })
        .expect("first plugin registered its message");
    app.world_mut().run_schedule(FixedUpdate);
}

#[test]
fn finishing_one_environment_type_does_not_reset_another() {
    let (mut app, first, second) = app_with_second(
        BevyGymPlugin::new(|_| Counter::<2>::default())
            .with_reset_seeds(|_, episode| Some(200 + episode)),
    );
    finish_first(&mut app, first);
    let stats = app.world().get::<EnvStats>(second).expect("second stats");
    assert_eq!(stats.total_episodes, 0);
    let observation = app
        .world()
        .get::<CurrentObservation<Counter<2>>>(second)
        .expect("second observation");
    assert_eq!(observation.observation, (0, Some(200)));
}

#[test]
fn automatic_reset_uses_the_seed_schedule_that_spawned_the_entity() {
    let (mut app, first, _) = app_with_second(
        BevyGymPlugin::new(|_| Counter::<2>::default())
            .with_reset_seeds(|_, episode| Some(200 + episode))
            .without_autoreset(),
    );
    finish_first(&mut app, first);
    let observation = app
        .world()
        .get::<CurrentObservation<Counter<1>>>(first)
        .expect("first observation");
    assert_eq!(observation.observation, (0, Some(101)));
}

#[test]
fn draining_public_transitions_cannot_interrupt_automatic_reset() {
    let (mut app, first, _) = app_with_second(BevyGymPlugin::new(|_| Counter::<2>::default()));
    app.add_systems(
        FixedUpdate,
        (|mut transitions: ResMut<Messages<TransitionEvent<Counter<1>>>>| {
            transitions.clear();
        })
        .after(GymSet::Step)
        .before(GymSet::AutoReset),
    );
    finish_first(&mut app, first);
    let observation = app
        .world()
        .get::<CurrentObservation<Counter<1>>>(first)
        .expect("first observation");
    assert_eq!(observation.observation, (0, Some(101)));
}

#[test]
fn despawning_a_finished_entity_does_not_reset_another_entity() {
    let (mut app, first, second) = app_with_second(BevyGymPlugin::new(|_| Counter::<2>::default()));
    app.add_systems(
        FixedUpdate,
        (move |world: &mut World| {
            assert!(world.despawn(first));
        })
        .after(GymSet::Step)
        .before(GymSet::AutoReset),
    );
    finish_first(&mut app, first);
    assert!(app.world().get_entity(first).is_err());
    assert_eq!(
        app.world()
            .get::<EnvStats>(second)
            .expect("second stats")
            .total_episodes,
        0
    );
    assert!(app
        .world()
        .resource::<Messages<ActionRequest<Counter<1>>>>()
        .is_empty());
}

#[test]
fn continuing_steps_do_not_trigger_automatic_reset() {
    let (mut app, _, second) = app_with_second(
        BevyGymPlugin::new(|_| Counter::<2>::default())
            .with_reset_seeds(|_, episode| Some(200 + episode)),
    );
    app.world_mut()
        .write_message(ActionResponse::<Counter<2>> {
            entity: second,
            action: (),
        })
        .expect("second plugin registered its message");
    app.world_mut().run_schedule(FixedUpdate);
    assert_eq!(
        app.world()
            .get::<CurrentObservation<Counter<2>>>(second)
            .expect("second observation")
            .observation,
        (1, Some(200))
    );
}
