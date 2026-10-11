//! Native scoped workers over the shared serial lane contract.
use bevy_gym::{
    training::{RecurrentPpoPolicy, RecurrentPpoSequence},
    Env,
};
use std::{error::Error, num::NonZeroU16};

/// Native collection over eight scoped workers with owned, transferable errors.
pub(crate) struct ParallelRecoveryBatch<E: Env, const N: usize> {
    /// Each lane retains its original environment, sampler and recurrent memory.
    inner: crate::learning::RecoveryBatch<E, N, Box<dyn Error + Send + Sync>>,
}
impl<E: Env + Send, const N: usize> ParallelRecoveryBatch<E, N>
where
    E::Observation: Copy + Send,
{
    /// Initialize the same eight lanes without supplying policy decisions.
    pub(crate) fn with_parallel_actions(
        seed: u64,
        policy: &RecurrentPpoPolicy,
        make_environment: fn() -> E,
        encode: fn(&E::Observation) -> [f32; N],
        decode: fn(&[f32]) -> Result<E::Action, Box<dyn Error + Send + Sync>>,
        limit: NonZeroU16,
    ) -> Self {
        Self {
            inner: crate::learning::RecoveryBatch::with_decoder(
                seed,
                policy,
                make_environment,
                encode,
                decode,
                limit,
            ),
        }
    }
    /// Collect exactly 512 transitions and retain original lane and sequence order.
    ///
    /// All workers finish before returning, including failures. A failed lane cannot
    /// supply another action after its error. Other valid lanes may finish their 64
    /// actions. Errors propagate in lane order; a worker panic returns an I/O error.
    /// Work and retained transition storage have the same bounds as serial collection.
    pub(crate) fn collect_parallel(
        &mut self,
        policy: &RecurrentPpoPolicy,
    ) -> Result<Vec<RecurrentPpoSequence>, Box<dyn Error>> {
        std::thread::scope(|scope| {
            // Array mapping starts every worker before any join can block.
            let handles = self.inner.collectors(policy).map(|collector| {
                scope.spawn(move || {
                    let mut sequences = Vec::new();
                    collector.collect_into(&mut sequences)?;
                    Ok(sequences)
                })
            });
            // Join all eight workers before resolving errors; no dropped handle can unwind the scope.
            let mut index = 0;
            let results = handles.map(|handle| {
                let lane_index = index;
                index += 1;
                handle
                    .join()
                    .map_err(|_panic_payload| {
                        Box::<dyn Error + Send + Sync>::from(std::io::Error::other(format!(
                            "rollout worker {lane_index} panicked"
                        )))
                    })
                    .and_then(|result| result)
            });
            let mut sequences = Vec::with_capacity(8);
            for result in results {
                sequences.extend(result?);
            }
            Ok::<_, Box<dyn Error + Send + Sync>>(sequences)
        })
        .map_err(|error| -> Box<dyn Error> { error })
    }
}
