//! Bounded on-policy standing training with explicit budget failure.

use super::{
    batch_with_recipe,
    checkpoint::{from_bytes, Record},
    evaluation, model,
    reward::{Recipe, POSTURE_SCALE},
};
use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoUpdate};
use serde::Serialize;
use std::{
    error::Error,
    io::Write,
    num::{NonZeroU32, NonZeroU64},
    path::Path,
};
use tokio::{fs, io::AsyncWriteExt};

/// Mutually exclusive weight origins; every run creates fresh optimization and episode state.
enum Start<'a> {
    /// Seeded random initialization.
    Random,
    /// Validated actor/critic import from an existing checkpoint.
    Checkpoint(&'a Path),
}

/// Fixed emitted audit profile; no parser or alternate spelling is provided.
#[derive(Serialize)]
enum WarmStartSchema {
    /// Fresh optimizer, minibatch RNG, environments, samplers and recurrent memory.
    #[serde(rename = "droid-standing-warm-start-v1")]
    V1,
}

/// Run-local seed and source identity, independent of the unchanged checkpoint sidecar.
#[derive(Serialize)]
struct WarmStartRecord<'a> {
    /// Closed warm-start semantics and format version.
    schema: WarmStartSchema,
    /// New run's seed; it does not claim to initialize the imported parameters.
    seed: u64,
    /// Validated source counters and SHA-256 of the retained initial.mpk bytes.
    source: &'a Record,
}

/// Train from random parameters; every selection uses a reloaded frozen policy.
pub(crate) async fn train(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    Box::pin(train_from(
        seed,
        updates,
        output,
        Start::Random,
        Recipe::Original,
    ))
    .await
}

/// Continue RL weights with fresh optimizer state and a separately recorded source identity.
pub(crate) async fn warm_start(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
    checkpoint: &Path,
) -> Result<(), Box<dyn Error>> {
    Box::pin(train_from(
        seed,
        updates,
        output,
        Start::Checkpoint(checkpoint),
        Recipe::Original,
    ))
    .await
}

/// Train a selected reward profile from random or validated RL checkpoint parameters.
pub(crate) async fn train_profiled(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
    checkpoint: Option<&Path>,
    recipe: Recipe,
) -> Result<(), Box<dyn Error>> {
    let start = checkpoint.map_or(Start::Random, Start::Checkpoint);
    Box::pin(train_from(seed, updates, output, start, recipe)).await
}

/// Validated import bytes and their source identity, retained before optimization.
struct InitialCheckpoint {
    /// Exact imported actor/critic container bytes.
    bytes: Vec<u8>,
    /// Existing source counters and weight digest.
    record: Record,
}

/// Ready learner and optional source evidence; optimizer and collector state are fresh.
struct Initialization {
    /// Configured recurrent learner with fresh Adam state.
    agent: RecurrentPpoAgent,
    /// None for seeded random initialization; otherwise the validated import.
    initial: Option<InitialCheckpoint>,
}

/// Validate the selected initialization before creating output or collecting any transitions.
async fn train_from(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
    start: Start<'_>,
    recipe: Recipe,
) -> Result<(), Box<dyn Error>> {
    let Initialization { mut agent, initial } = initialize(seed, start).await?;
    create_output(output, seed, initial).await?;
    record_reward_profile(output, seed, recipe).await?;
    let mut progress = fs::File::create(output.join("optimization.jsonl")).await?;
    let mut row = Vec::with_capacity(512);
    let mut rollout = batch_with_recipe(seed, &agent.policy(), recipe);
    for update in 1..=updates.get() {
        // One immutable policy snapshot supplies all eight lanes before either optimizer changes.
        let sequences = rollout.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        let update = NonZeroU32::new(update).expect("positive update loop");
        row.clear();
        record_update(&mut row, update, &metrics)?;
        progress.write_all(&row).await?;
        progress.flush().await?;
        if !update.get().is_multiple_of(20) && update != updates {
            continue;
        }
        if save_and_evaluate(output, seed, update, &metrics, agent.policy().to_bytes()?).await? {
            return Ok(());
        }
    }
    Err("Standing exhausted its update budget without passing; checkpoint is not qualified.".into())
}

/// Validate the original checkpoint shape before a new output directory exists.
async fn initialize(seed: u64, start: Start<'_>) -> Result<Initialization, Box<dyn Error>> {
    match start {
        Start::Random => Ok(Initialization {
            agent: model::new_agent(seed)?,
            initial: None,
        }),
        Start::Checkpoint(path) => {
            let bytes = fs::read(path).await?;
            let metadata = fs::read(path.with_extension("json")).await?;
            let record = Record::for_bytes(&bytes, &metadata)?;
            // One cold copy retains exact source evidence while Burn consumes its decoder input.
            let agent = model::warm_start(seed, bytes.clone())?;
            Ok(Initialization {
                agent,
                initial: Some(InitialCheckpoint { bytes, record }),
            })
        }
    }
}

/// Reject existing trials and retain validated source evidence before collection.
async fn create_output(
    output: &Path,
    seed: u64,
    initial: Option<InitialCheckpoint>,
) -> Result<(), Box<dyn Error>> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).await?;
    fs::create_dir(output).await?;
    if let Some(InitialCheckpoint { bytes, record }) = initial {
        fs::write(output.join("initial.mpk"), &bytes).await?;
        fs::write(
            output.join("initial.json"),
            serde_json::to_vec_pretty(&record)?,
        )
        .await?;
        let origin = WarmStartRecord {
            schema: WarmStartSchema::V1,
            seed,
            source: &record,
        };
        fs::write(
            output.join("warm-start.json"),
            serde_json::to_vec_pretty(&origin)?,
        )
        .await?;
    }
    Ok(())
}

/// Emit the complete local profile before collection; the original recipe adds no artifact.
async fn record_reward_profile(
    output: &Path,
    seed: u64,
    recipe: Recipe,
) -> Result<(), Box<dyn Error>> {
    if recipe == Recipe::Original {
        return Ok(());
    }
    let record = serde_json::json!({"schema":recipe, "seed":seed,
        "posture_up_projection_scale":POSTURE_SCALE, "contact_both":1.0,
        "contact_one":0.5, "contact_none":0.25, "evaluation":"original-standing-v1"});
    fs::write(output.join("training-reward.json"), record.to_string()).await?;
    Ok(())
}

/// Preserve exact weights and provenance before reporting independent selection outcomes.
async fn save_and_evaluate(
    output: &Path,
    seed: u64,
    update: NonZeroU32,
    metrics: &RecurrentPpoUpdate,
    bytes: Vec<u8>,
) -> Result<bool, Box<dyn Error>> {
    let steps =
        NonZeroU64::new(metrics.optimizer_steps).ok_or("PPO reported zero optimizer steps")?;
    let record = Record::new(seed, update, steps, &bytes);
    let metadata = serde_json::to_vec_pretty(&record)?;
    let number = update.get();
    let checkpoint = output.join(format!("standing-{number}.mpk"));
    fs::write(&checkpoint, &bytes).await?;
    fs::write(checkpoint.with_extension("json"), &metadata).await?;
    let (policy, _) = from_bytes(bytes, &metadata)?;
    let episodes = evaluation::evaluate(&policy, &evaluation::SELECTION_SEEDS)?;
    let passed = evaluation::passes(&episodes, &evaluation::SELECTION_SEEDS);
    let report = serde_json::json!({"mode":"selection", "provenance":record,
        "episodes":episodes, "passed":passed, "qualification":"requires held-out evaluation and independent run evidence"});
    fs::write(
        output.join(format!("standing-{number}.evaluation.json")),
        report.to_string(),
    )
    .await?;
    println!("{report}");
    Ok(passed)
}

/// Reject non-finite measurements before JSON could silently replace them with null.
fn record_update(
    writer: &mut impl Write,
    update: NonZeroU32,
    metrics: &RecurrentPpoUpdate,
) -> std::io::Result<()> {
    for (name, value) in [
        ("actor_loss", metrics.actor_loss),
        ("critic_loss", metrics.critic_loss),
        ("entropy", metrics.entropy),
        ("approximate_kl", metrics.approximate_kl),
        ("actor_learning_rate", metrics.actor_learning_rate),
        ("critic_learning_rate", metrics.critic_learning_rate),
    ] {
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("non-finite PPO metric: {name}"),
            ));
        }
    }
    let record = serde_json::json!({"lesson":"droid-standing-v1", "update":update,
        "optimizer_steps":metrics.optimizer_steps, "optimizer_updates":metrics.optimizer_updates,
        "valid_samples":metrics.valid_samples, "actor_loss":metrics.actor_loss,
        "critic_loss":metrics.critic_loss, "entropy":metrics.entropy,
        "approximate_kl":metrics.approximate_kl, "actor_learning_rate":metrics.actor_learning_rate,
        "critic_learning_rate":metrics.critic_learning_rate});
    writeln!(writer, "{record}")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blocked profile artifact propagates I/O failure before any collector is created.
    #[test]
    fn reward_profile_write_failure_is_not_a_fallback() {
        let process = std::process::id();
        let directory = std::env::temp_dir().join(format!("standing-profile-write-{process}"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            fs::create_dir(&directory)
                .await
                .expect("new isolated directory");
            let artifact = directory.join("training-reward.json");
            fs::create_dir(&artifact).await.expect("blocked artifact");
            record_reward_profile(&directory, 17, Recipe::Original)
                .await
                .expect("original emits no profile");
            record_reward_profile(&directory, 17, Recipe::PostureV1)
                .await
                .expect_err("profile cannot replace a directory");
            assert!(!directory.join("optimization.jsonl").exists());
            fs::remove_dir(&artifact)
                .await
                .expect("remove test artifact");
            fs::remove_dir(&directory)
                .await
                .expect("remove isolated test directory");
        });
    }

    /// Finite metrics with signed losses for journal tests.
    fn metrics() -> RecurrentPpoUpdate {
        RecurrentPpoUpdate {
            optimizer_steps: 4,
            optimizer_updates: 4,
            valid_samples: 512,
            actor_loss: -0.1,
            critic_loss: 2.0,
            entropy: -2.0,
            approximate_kl: -0.01,
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.001,
        }
    }

    /// The journal appends complete finite rows and rejects every non-finite measurement first.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn journal_records_finite_metrics_and_rejects_nonfinite_values() {
        let mut bytes = Vec::new();
        record_update(&mut bytes, NonZeroU32::MIN, &metrics()).expect("finite row");
        record_update(
            &mut bytes,
            NonZeroU32::new(2).expect("positive"),
            &metrics(),
        )
        .expect("second row");
        let text = String::from_utf8(bytes).expect("JSON UTF-8");
        assert!(text.ends_with('\n'));
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("JSON row"))
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.first().expect("first")["actor_loss"], -0.1);
        assert_eq!(rows.last().expect("last")["update"], 2);
        for field in 0..6 {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut measurements = metrics();
                match field {
                    0 => measurements.actor_loss = value,
                    1 => measurements.critic_loss = value,
                    2 => measurements.entropy = value,
                    3 => measurements.approximate_kl = value,
                    4 => measurements.actor_learning_rate = value,
                    _ => measurements.critic_learning_rate = value,
                }
                let mut bytes = Vec::new();
                assert_eq!(
                    record_update(&mut bytes, NonZeroU32::MIN, &measurements)
                        .expect_err("non-finite metric")
                        .kind(),
                    std::io::ErrorKind::InvalidData
                );
                assert!(bytes.is_empty());
            }
        }
    }

    /// The two possible I/O failure stages.
    enum Failure {
        Write,
        Flush,
    }
    /// A deterministic failing journal sink used only in tests.
    struct FailedWriter(Failure);
    impl Write for FailedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            match self.0 {
                Failure::Write => Err(std::io::Error::other("write failure")),
                Failure::Flush => Ok(bytes.len()),
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("flush failure"))
        }
    }

    /// A failed write or flush reaches the caller instead of allowing another rollout.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn journal_propagates_write_and_flush_errors() {
        for (failure, expected) in [
            (Failure::Write, "write failure"),
            (Failure::Flush, "flush failure"),
        ] {
            let error = record_update(&mut FailedWriter(failure), NonZeroU32::MIN, &metrics())
                .expect_err("failed journal");
            assert_eq!(error.to_string(), expected);
        }
    }
}
