/// Systems for automatic and manual environment resets.
pub mod reset;
/// System for stepping environments with action responses.
pub mod step;

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use crate::{
        ActionRequest, ActionResponse, BevyGymPlugin, CurrentObservation, Env, EpisodeEndEvent,
        EpisodeStatus, Reset, ResetRequested, Step, TransitionEvent,
    };

    #[derive(Clone, Debug)]
    struct TestEnv {
        resets: u32,
        steps: u32,
        terminal: bool,
    }

    impl TestEnv {
        fn continuing() -> Self {
            Self {
                resets: 0,
                steps: 0,
                terminal: false,
            }
        }

        fn terminal() -> Self {
            Self {
                resets: 0,
                steps: 0,
                terminal: true,
            }
        }
    }

    impl Env for TestEnv {
        type Observation = u32;
        type Action = bool;
        type Info = ();

        fn reset(&mut self, _seed: Option<u64>) -> Reset<u32> {
            self.resets += 1;
            self.steps = 0;
            Reset {
                observation: self.resets * 100,
                info: (),
            }
        }

        fn step(&mut self, action: bool) -> Step<u32> {
            assert!(action);
            self.steps += 1;
            let observation = self.resets * 100 + self.steps;
            Step {
                observation,
                reward: 1.0,
                status: if self.terminal {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Continuing
                },
                info: (),
            }
        }
    }

    fn app_with(factory: impl Fn(usize) -> TestEnv + Send + Sync + 'static) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(BevyGymPlugin::new(factory));
        app.update();
        app
    }

    fn drain_requests(app: &mut App) -> Vec<ActionRequest<TestEnv>> {
        app.world_mut()
            .resource_mut::<Messages<ActionRequest<TestEnv>>>()
            .drain()
            .collect()
    }

    fn write_response(app: &mut App, request: &ActionRequest<TestEnv>) {
        app.world_mut()
            .resource_mut::<Messages<ActionResponse<TestEnv>>>()
            .write(ActionResponse {
                entity: request.entity,
                action: true,
            });
    }

    #[test]
    fn startup_emits_initial_action_request() {
        let mut app = app_with(|_| TestEnv::continuing());
        let requests = drain_requests(&mut app);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].env_id, 0);
        assert_eq!(requests[0].observation, 100);
    }

    #[test]
    fn continuing_step_emits_transition_and_next_request() {
        let mut app = app_with(|_| TestEnv::continuing());
        let request = drain_requests(&mut app).pop().unwrap();

        write_response(&mut app, &request);
        app.world_mut().run_schedule(FixedUpdate);

        let transitions: Vec<_> = app
            .world_mut()
            .resource_mut::<Messages<TransitionEvent<TestEnv>>>()
            .drain()
            .collect();
        let requests = drain_requests(&mut app);

        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].transition.observation, 100);
        assert_eq!(transitions[0].transition.next_observation, 101);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].observation, 101);
    }

    #[test]
    fn terminal_step_auto_resets_and_emits_one_post_reset_request() {
        let mut app = app_with(|_| TestEnv::terminal());
        let request = drain_requests(&mut app).pop().unwrap();

        write_response(&mut app, &request);
        app.world_mut().run_schedule(FixedUpdate);

        let episodes: Vec<_> = app
            .world_mut()
            .resource_mut::<Messages<EpisodeEndEvent>>()
            .drain()
            .collect();
        let requests = drain_requests(&mut app);

        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].status, EpisodeStatus::Terminated);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].observation, 200);
    }

    #[test]
    fn manual_reset_emits_fresh_request_and_removes_marker() {
        let mut app = app_with(|_| TestEnv::continuing());
        let request = drain_requests(&mut app).pop().unwrap();
        app.world_mut()
            .entity_mut(request.entity)
            .insert(ResetRequested { seed: Some(42) });

        app.world_mut().run_schedule(FixedUpdate);

        let requests = drain_requests(&mut app);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].observation, 200);
        assert!(!app
            .world()
            .entity(request.entity)
            .contains::<ResetRequested>());
    }

    #[test]
    fn duplicate_action_responses_do_not_step() {
        let mut app = app_with(|_| TestEnv::continuing());
        let request = drain_requests(&mut app).pop().unwrap();

        write_response(&mut app, &request);
        write_response(&mut app, &request);
        app.world_mut().run_schedule(FixedUpdate);

        let requests = drain_requests(&mut app);
        let observation = app
            .world()
            .entity(request.entity)
            .get::<CurrentObservation<TestEnv>>()
            .unwrap()
            .observation;

        assert!(app
            .world_mut()
            .resource_mut::<Messages<TransitionEvent<TestEnv>>>()
            .drain()
            .next()
            .is_none());
        assert!(requests.is_empty());
        assert_eq!(observation, 100);
    }

    #[derive(Clone, Debug)]
    struct SeedEchoEnv;

    impl Env for SeedEchoEnv {
        type Observation = u64;
        type Action = ();
        type Info = ();

        fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
            Reset {
                observation: seed.unwrap_or(u64::MAX),
                info: (),
            }
        }

        fn step(&mut self, (): Self::Action) -> Step<Self::Observation> {
            Step {
                observation: 0,
                reward: 0.0,
                status: EpisodeStatus::Terminated,
                info: (),
            }
        }
    }

    #[test]
    fn automatic_resets_use_per_environment_episode_seed_schedule() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins).add_plugins(
            BevyGymPlugin::new(|_| SeedEchoEnv)
                .with_envs(2)
                .with_reset_seeds(|env_id, episode| Some(1_000 + env_id as u64 * 100 + episode)),
        );
        app.update();

        let initial: Vec<_> = app
            .world_mut()
            .resource_mut::<Messages<ActionRequest<SeedEchoEnv>>>()
            .drain()
            .collect();
        // Bevy does not guarantee query or message order by environment ID.
        let mut initial_observations = initial
            .iter()
            .map(|request| (request.env_id, request.observation))
            .collect::<Vec<_>>();
        initial_observations.sort_unstable();
        assert_eq!(initial_observations, vec![(0, 1_000), (1, 1_100)]);

        for request in initial {
            app.world_mut()
                .resource_mut::<Messages<ActionResponse<SeedEchoEnv>>>()
                .write(ActionResponse {
                    entity: request.entity,
                    action: (),
                });
        }
        app.world_mut().run_schedule(FixedUpdate);

        let next: Vec<_> = app
            .world_mut()
            .resource_mut::<Messages<ActionRequest<SeedEchoEnv>>>()
            .drain()
            .collect();
        let mut next_observations = next
            .iter()
            .map(|request| (request.env_id, request.observation))
            .collect::<Vec<_>>();
        next_observations.sort_unstable();
        assert_eq!(next_observations, vec![(0, 1_001), (1, 1_101)]);
    }
}
