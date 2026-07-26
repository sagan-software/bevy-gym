//! Synchronous transition collection through the Bevy ECS runner.

use std::error::Error;
use std::fmt;

use bevy::prelude::{App, FixedUpdate, Messages, MinimalPlugins};

use crate::{ActionRequest, ActionResponse, BevyGymPlugin, Env, EpisodeEndEvent, TransitionEvent};

/// One deterministic synchronous batch emitted by the Bevy runner.
#[derive(Clone)]
pub struct TransitionBatch<E: Env + Send + Sync + 'static> {
    /// One transition for every environment that stepped.
    pub transitions: Vec<TransitionEvent<E>>,

    /// Episode completions emitted during this runner tick.
    pub episode_ends: Vec<EpisodeEndEvent>,
}

impl<E: Env + Send + Sync + 'static> TransitionBatch<E> {
    /// Return the number of environment transitions in this batch.
    #[must_use]
    pub const fn environment_steps(&self) -> usize {
        self.transitions.len()
    }
}

impl<E: Env + Send + Sync + 'static> fmt::Debug for TransitionBatch<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransitionBatch")
            .field("transitions", &self.transitions.len())
            .field("episode_ends", &self.episode_ends)
            .finish()
    }
}

/// Headless synchronous facade over [`BevyGymPlugin`].
pub struct BevyTransitionCollector<E: Env + Send + Sync + 'static> {
    /// Bevy application containing the environment entities and schedules.
    app: App,

    /// Action requests that must be answered by the next [`Self::step`] call.
    requests: Vec<ActionRequest<E>>,

    /// Stable number of parallel environment entities.
    num_envs: usize,
}

impl<E: Env + Send + Sync + 'static> fmt::Debug for BevyTransitionCollector<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BevyTransitionCollector")
            .field("num_envs", &self.num_envs)
            .field("pending_requests", &self.requests.len())
            .finish_non_exhaustive()
    }
}

impl<E: Env + Send + Sync + 'static> BevyTransitionCollector<E> {
    /// Construct a headless Bevy runner and collect its initial action requests.
    ///
    /// # Errors
    ///
    /// Returns [`CollectionError::ZeroEnvironments`] for an empty pool or
    /// [`CollectionError::RequestCount`] if runner startup does not request one
    /// action per environment entity.
    pub fn new(
        factory: impl Fn(usize) -> E + Send + Sync + 'static,
        num_envs: usize,
        reset_seed: impl Fn(usize, u64) -> Option<u64> + Send + Sync + 'static,
    ) -> Result<Self, CollectionError> {
        if num_envs == 0 {
            return Err(CollectionError::ZeroEnvironments);
        }

        let mut app = App::new();
        app.add_plugins(MinimalPlugins).add_plugins(
            BevyGymPlugin::new(factory)
                .with_envs(num_envs)
                .uncapped()
                .with_reset_seeds(reset_seed),
        );
        app.update();

        let mut requests: Vec<_> = app
            .world_mut()
            .resource_mut::<Messages<ActionRequest<E>>>()
            .drain()
            .collect();
        requests.sort_by_key(|request| request.env_id);
        if requests.len() != num_envs {
            return Err(CollectionError::RequestCount {
                expected: num_envs,
                actual: requests.len(),
            });
        }

        Ok(Self {
            app,
            requests,
            num_envs,
        })
    }

    /// Return the number of parallel Bevy environment entities.
    #[must_use]
    pub const fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Borrow action requests in stable environment-id order.
    #[must_use]
    pub fn requests(&self) -> &[ActionRequest<E>] {
        &self.requests
    }

    /// Apply exactly one action per pending request and run one ECS tick.
    ///
    /// # Errors
    ///
    /// Returns [`CollectionError::ActionCount`] without stepping when the
    /// action count differs from pending requests. Returns
    /// [`CollectionError::TransitionCount`] if the runner fails to emit one
    /// transition for every accepted action.
    pub fn step(
        &mut self,
        actions: impl IntoIterator<Item = E::Action>,
    ) -> Result<TransitionBatch<E>, CollectionError> {
        let actions: Vec<_> = actions.into_iter().collect();
        if actions.len() != self.requests.len() {
            return Err(CollectionError::ActionCount {
                expected: self.requests.len(),
                actual: actions.len(),
            });
        }

        {
            let mut responses = self
                .app
                .world_mut()
                .resource_mut::<Messages<ActionResponse<E>>>();
            for (request, action) in self.requests.iter().zip(actions) {
                responses.write(ActionResponse {
                    entity: request.entity,
                    action,
                });
            }
        }

        self.app.world_mut().run_schedule(FixedUpdate);
        self.app
            .world_mut()
            .resource_mut::<Messages<ActionResponse<E>>>()
            .clear();

        let mut transitions: Vec<_> = self
            .app
            .world_mut()
            .resource_mut::<Messages<TransitionEvent<E>>>()
            .drain()
            .collect();
        transitions.sort_by_key(|event| event.env_id);

        let mut episode_ends: Vec<_> = self
            .app
            .world_mut()
            .resource_mut::<Messages<EpisodeEndEvent>>()
            .drain()
            .collect();
        episode_ends.sort_by_key(|event| event.env_id);

        let mut requests: Vec<_> = self
            .app
            .world_mut()
            .resource_mut::<Messages<ActionRequest<E>>>()
            .drain()
            .collect();
        requests.sort_by_key(|request| request.env_id);
        self.requests = requests;

        if transitions.len() != self.num_envs {
            return Err(CollectionError::TransitionCount {
                expected: self.num_envs,
                actual: transitions.len(),
            });
        }
        if self.requests.len() != self.num_envs {
            return Err(CollectionError::RequestCount {
                expected: self.num_envs,
                actual: self.requests.len(),
            });
        }

        Ok(TransitionBatch {
            transitions,
            episode_ends,
        })
    }
}

/// Invalid runner collection request or invariant violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionError {
    /// A qualifying parallel collector needs at least one environment.
    ZeroEnvironments,

    /// Caller did not provide exactly one action per pending request.
    ActionCount {
        /// Required action count.
        expected: usize,

        /// Supplied action count.
        actual: usize,
    },

    /// Runner did not expose exactly one next request per environment.
    RequestCount {
        /// Required request count.
        expected: usize,

        /// Observed request count.
        actual: usize,
    },

    /// Runner did not emit exactly one transition per accepted action.
    TransitionCount {
        /// Required transition count.
        expected: usize,

        /// Observed transition count.
        actual: usize,
    },
}

impl fmt::Display for CollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEnvironments => {
                formatter.write_str("transition collector requires at least one environment")
            }
            Self::ActionCount { expected, actual } => write!(
                formatter,
                "transition collector expected {expected} actions, received {actual}"
            ),
            Self::RequestCount { expected, actual } => write!(
                formatter,
                "Bevy runner expected {expected} action requests, emitted {actual}"
            ),
            Self::TransitionCount { expected, actual } => write!(
                formatter,
                "Bevy runner expected {expected} transitions, emitted {actual}"
            ),
        }
    }
}

impl Error for CollectionError {}

#[cfg(test)]
mod tests {
    use crate::{EpisodeStatus, Reset, Step};

    use super::*;

    #[derive(Clone, Debug)]
    struct CounterEnv {
        value: u32,
    }

    impl Env for CounterEnv {
        type Observation = u32;
        type Action = u32;
        type Info = ();

        fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
            self.value = seed.unwrap_or_default() as u32;
            Reset {
                observation: self.value,
                info: (),
            }
        }

        fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
            self.value += action;
            Step {
                observation: self.value,
                reward: f64::from(action),
                status: if self.value >= 20 {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Continuing
                },
                info: (),
            }
        }
    }

    #[test]
    fn collector_steps_multiple_entities_through_typed_runner_transitions() {
        let mut collector = BevyTransitionCollector::new(
            |_| CounterEnv { value: 0 },
            3,
            |env_id, episode| Some(10 + env_id as u64 + episode * 100),
        )
        .expect("positive environment count");

        assert_eq!(collector.num_envs(), 3);
        assert_eq!(
            collector
                .requests()
                .iter()
                .map(|request| request.observation)
                .collect::<Vec<_>>(),
            vec![10, 11, 12]
        );

        let batch = collector.step([1, 2, 3]).expect("one action per entity");
        assert_eq!(batch.transitions.len(), 3);
        assert_eq!(batch.episode_ends.len(), 0);
        assert_eq!(batch.environment_steps(), 3);
        assert_eq!(
            batch
                .transitions
                .iter()
                .map(|event| event.transition.next_observation)
                .collect::<Vec<_>>(),
            vec![11, 13, 15]
        );
        assert_eq!(collector.requests().len(), 3);
    }

    #[test]
    fn collector_rejects_missing_or_extra_actions_without_stepping() {
        let mut collector = BevyTransitionCollector::new(
            |_| CounterEnv { value: 0 },
            2,
            |env_id, _episode| Some(env_id as u64),
        )
        .expect("positive environment count");

        assert!(matches!(
            collector.step([1]),
            Err(CollectionError::ActionCount {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(collector.requests().len(), 2);
        assert!(matches!(
            BevyTransitionCollector::new(|_| CounterEnv { value: 0 }, 0, |_env_id, _episode| None,),
            Err(CollectionError::ZeroEnvironments)
        ));
    }

    #[test]
    fn collector_exposes_episode_end_and_seeded_autoreset_request() {
        let mut collector = BevyTransitionCollector::new(
            |_| CounterEnv { value: 0 },
            1,
            |_env_id, episode| Some(19 + episode * 100),
        )
        .expect("positive environment count");

        let batch = collector.step([1]).expect("one action");
        assert_eq!(batch.episode_ends.len(), 1);
        assert_eq!(batch.episode_ends[0].status, EpisodeStatus::Terminated);
        assert_eq!(collector.requests()[0].observation, 119);
    }
}
