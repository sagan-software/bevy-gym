---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - README.md
  - Cargo.toml
  - docs/examples/cartpole.md
  - docs/plugins/bevy_gym_plugin.md
  - docs/plugins/gym_render.md
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Reinforcement learning in the Rust ecosystem with Burn, Bevy, and Godot'
research_goals: 'Find prior art, notable examples, toy projects, papers, relevant crates, and transferable patterns for curriculum learning, continual learning, transfer learning, physics-based locomotion training, and imitation learning.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Reinforcement Learning in Rust, Burn, Bevy, and Godot

**Date:** 2026-06-12
**Author:** Sagan
**Research Type:** Technical research
**Local Anchor:** `bevy-gym`

---

## Executive Summary

Rust now has enough reinforcement learning pieces to build real experiments, but it does not yet have a mature Rust-native equivalent to Unity ML-Agents, Godot RL Agents, Isaac Lab, MuJoCo Playground, or the legged-robotics Python stacks. The strongest Rust-native path for this repository is still:

1. Keep `rl-traits` as the environment and policy contract.
2. Use `bevy-gym` as the parallel Bevy ECS environment substrate.
3. Use Burn-based training through `ember-rl`, `burn-rl`, or a narrow local trainer where the public crates are still too immature.
4. Borrow curriculum, imitation, locomotion, transfer, and continual-learning patterns from the broader robotics/game-engine world.

The local `bevy-gym` project is well-positioned as the Rust environment runtime: it bridges `rl-traits` environments into Bevy ECS, runs many environment instances as Bevy entities, steps them in `FixedUpdate`, supports headless `ScheduleRunnerPlugin` operation, and has an optional render feature. The missing pieces are not basic CartPole-style RL. The gaps are continuous-control training, physics stability gates, locomotion morphology conventions, demonstration/replay data, curriculum APIs, transfer/evaluation harnesses, and a clear route for imitation or adversarial motion-prior training.

The best practical recommendation is a hybrid architecture: make `bevy-gym` the Rust-native environment and data collection layer, keep Burn as the preferred Rust model/training backend, and maintain an optional Python/SB3/Godot/Unity/Isaac-style bridge as a baseline oracle for algorithms that Rust crates have not yet proven at scale.

---

## Methodology And Source Boundary

This report combines local repository inspection with current external source review. Local files inspected:

- `Cargo.toml`
- `README.md`
- `docs/examples/cartpole.md`
- `docs/plugins/bevy_gym_plugin.md`
- `docs/plugins/gym_render.md`

External sources included current docs.rs/crates.io metadata, official framework docs, GitHub project pages, and primary or near-primary research paper pages. Current crate metadata was checked on 2026-06-12, so version and download counts should be treated as point-in-time evidence.

The Rust ecosystem does not have a single canonical RL index. The Rust section below is an evidence-backed catalog of discoverable relevant crates and notable examples from crates.io, docs.rs, GitHub, and local ecosystem references, not a mathematical proof that no obscure repository exists.

## Local Project Baseline

`bevy-gym` is already aimed at the right architectural layer. It is not an algorithm crate; it is an environment runtime:

- Package: `bevy-gym` 0.3.3.
- Core dependency: `rl-traits` 0.2.2.
- Engine dependency: `bevy` 0.18 with minimal default features.
- Dev training examples: `ember-rl` 0.3.4 and `burn` 0.20.1 with `ndarray` and `autodiff`.
- README ecosystem model: `rl-traits` for shared interfaces, `ember-rl` for Burn-based algorithms, `bevy-gym` for parallelized environment simulation.
- CartPole example: 4 parallel environments, DQN through `ember-rl::TrainingSession`, JSONL logging, checkpointing, `best.mpk`, and optional 2D rendering.

That means the local project should be treated as an ECS/vectorized simulation substrate, not as the place to hide algorithm-specific logic. Curriculum, imitation, and transfer machinery should be expressed as environment metadata, reset distributions, action/observation schemas, replay/demo records, and evaluation harnesses that trainers can consume.

## Rust Ecosystem Map

### Burn And Burn-Native RL

| Artifact | Status | What It Offers | Relevance |
| --- | --- | --- | --- |
| [Burn](https://docs.rs/burn/latest/burn/) / [crates.io](https://crates.io/crates/burn) | Current crate metadata found 0.21.0 | Rust deep learning framework with multiple backends and autodiff support. | Best Rust-native deep learning candidate for this repo's preferred stack. |
| [burn-rl](https://docs.rs/burn-rl/latest/burn_rl/) / [crates.io](https://crates.io/crates/burn-rl) | Current crate metadata found 0.21.0 | Burn's RL module. Docs expose examples/modules around Q-learning, DQN, REINFORCE, and actor-critic style learning. | Useful reference and possible future dependency, but currently much thinner than mature Python RL libraries. |
| [ember-rl](https://crates.io/crates/ember-rl) | Current crate metadata found 0.3.4 | Burn-based DQN, PPO, SAC implementations; used by local CartPole docs. | Closest match to local `bevy-gym` architecture. Treat as the first Rust-native trainer to integrate. |
| [rl4burn](https://docs.rs/rl4burn/latest/rl4burn/) | Current crate metadata found 0.1.1 | Burn RL algorithms including DQN/PPO-style pieces, GAE, replay buffer, agent/env modules. | Worth reading for implementation patterns; adoption risk is higher than `ember-rl` because it is early. |
| [rwrd](https://docs.rs/rwrd/latest/rwrd/) | Current crate metadata found 0.4.0 | Burn-powered framework with A2C, PPO, SAC, DQN modules. | Interesting algorithm prior art; needs code quality and maintenance review before dependency use. |
| [r2l-burn](https://docs.rs/r2l-burn/latest/r2l_burn/) / [r2l-gym](https://crates.io/crates/r2l-gym) | Current crate metadata found 0.0.2-rc2 | Rust-first RL with Burn backend, CPU training note, examples including CartPole and Montezuma's Revenge, PPO/RND references. | Relevant for PPO/RND design and environment split, but release-candidate maturity. |
| [sb3-burn](https://github.com/will-maclean/sb3-burn) | GitHub project | Rust/Burn implementation inspired by Stable-Baselines3. | Strong conceptual fit for a Burn-first ecosystem; evaluate as a project reference rather than assuming production readiness. |
| [rlevo](https://crates.io/crates/rlevo) / [GitHub](https://github.com/anthonytorlucci/burn-evorl) | Current crate metadata found 0.2.0 | Burn-based evolutionary RL / population optimization. | Useful for open-ended/curriculum experiments where population search may beat gradient-only local optima. |
| [reinforcex](https://crates.io/crates/reinforcex) | Search result indicates DQN/PPO framework | Deep RL framework built in Rust. | Candidate to inspect for API ideas; not yet validated deeply in this pass. |

### Rust RL Libraries Not Centered On Burn

| Artifact | Status | What It Offers | Relevance |
| --- | --- | --- | --- |
| [border](https://docs.rs/border/latest/border/) | Current crate metadata found 0.0.8 | Experimental Rust RL library using `tch`; docs list DQN, QR-DQN, IQN, PPO, TD3, SAC, REDQ, Gymnasium/Atari examples. | Best non-Burn algorithm reference in Rust. More mature algorithm spread than most Burn crates, but depends on PyTorch/libtorch stack. |
| [rurel](https://docs.rs/rurel/latest/rurel/) | Current crate metadata found 0.6.0 | Reusable RL traits and tabular Q-learning/SARSA style training. | Good historical Rust RL prior art. Less relevant for deep locomotion. |
| [gym-rs](https://docs.rs/gym-rs/latest/gym_rs/) | Current crate metadata found 0.3.0 | Pure Rust OpenAI Gym-like environments: FrozenLake, Blackjack, CartPole, MountainCar, Acrobot, Pendulum, LunarLander, BipedalWalker. | Useful environment reference and source of classic-control test targets. |
| [scivex-rl](https://docs.rs/scivex_rl/latest/scivex_rl/) | Current crate metadata found 0.1.1 | RL environments and algorithms including DQN, PPO, A2C. | Early but relevant for comparing trait/API choices. |
| [train-station](https://crates.io/crates/train-station) | Search result shows PPO continuous example | Zero-dependency PyTorch-inspired Rust ML library with RL examples. | Interesting if dependency minimization matters, but outside Burn. |
| [tch-rs](https://crates.io/crates/tch) | Current crate metadata found 0.24.0 | Rust wrappers for PyTorch C++ API. | Pragmatic route for algorithm parity with Python/PyTorch, but less Rust-native than Burn. |
| [Candle](https://crates.io/crates/candle-core) | Current crate metadata found 0.10.2 | Hugging Face minimalist ML framework in Rust. | Strong inference-oriented ecosystem; less directly tied to RL training in discovered examples. |
| [dfdx](https://crates.io/crates/dfdx) | Current crate metadata found 0.13.0 | Rust autodiff/deep learning library. | Useful prior art for type-driven tensor APIs, but not the main RL route here. |

### Bevy-Specific RL And Simulation Prior Art

| Artifact | Stack | What It Proves | Use For `bevy-gym` |
| --- | --- | --- | --- |
| Local [`bevy-gym`](https://crates.io/crates/bevy-gym) | Bevy ECS + `rl-traits` | Parallel RL environments as Bevy entities, headless schedule runner, optional rendering. | Keep as the core Rust-native env substrate. |
| [bevy_rl](https://docs.rs/bevy_rl/latest/bevy_rl/) / [GitHub](https://github.com/stillonearth/bevy_rl) | Bevy + Gym-like API + REST/pixels | Exposes Bevy worlds to Gym-style control, multi-agent support, pixel observations, REST API. | Study for external API, multi-agent, and pixel-observation design. |
| [bevy_rl_shooter](https://github.com/stillonearth/bevy_rl_shooter) | Bevy + `bevy_rl` | Multi-agent FPS/deathmatch toy environment. | Useful for multi-agent reward/event schema ideas. |
| [bevy_quadruped_neural_control](https://github.com/stillonearth/bevy_quadruped_neural_control) | Bevy + MuJoCo + `bevy_rl` + SB3/ONNX | Unitree A1 quadruped simulation wrapped as RL environment, with Python policy export path. | Most directly relevant Bevy locomotion prior art found. Study this before designing a Rust locomotion stack. |
| [lunar-lander-tch](https://github.com/Robsutar/lunar-lander-tch) | Rust + Bevy + Rapier2D + `tch-rs` | LunarLander implementation with DQN-family training in Rust. | Good toy project for Bevy physics plus RL training loop patterns. |
| [bevy-snake-ai](https://github.com/cswinter/bevy-snake-ai) | Bevy game + trained opponents | Uses trained agents as varying opponent skill levels. | Lightweight example of packaging learned policies into a game. |
| [bevy-tnua](https://docs.rs/bevy-tnua) | Bevy character controller | Floating character controller integrations for Rapier/Avian. | Not an RL system, but useful for non-ragdoll locomotion tasks and baselines. |
| [Avian](https://docs.rs/avian3d/latest/avian3d/) | Bevy-native ECS physics | ECS-driven 2D/3D physics engine. | Strong candidate for Bevy-native physics training, especially if ECS integration matters. |
| [bevy_rapier3d](https://crates.io/crates/bevy_rapier3d) | Rapier physics plugin | Mature Bevy integration for Rapier 3D physics. | Candidate when physics maturity matters more than pure ECS-native design. |

### Godot RL Prior Art

| Artifact | Stack | What It Offers | Relevance |
| --- | --- | --- | --- |
| [Godot RL Agents](https://edbeeching.github.io/papers/gdrl.html) | Godot + Python training | Interfaces Godot-created environments with ML algorithms; supports 2D/3D, image observations, memory agents, multi-agent, and robust communication. | Mature Godot path, but training is Python-first. Best baseline for engine bridge design. |
| [godot-rust/gdext](https://godot-rust.github.io/) / [godot crate](https://crates.io/crates/godot) | Godot 4 Rust bindings | Rust GDExtension bindings for Godot. | Useful for Rust environment code inside Godot, but not an RL framework. |
| No `godot-rl` crate found | crates.io search | No obvious Rust-native Godot RL crate surfaced. | For Godot, expect to bridge to Python/SB3/RLlib or use local Rust bindings plus a custom protocol. |

Godot's strongest RL story is currently not Rust-native. If the project needs Godot scenes, use Godot RL Agents as the reference architecture: engine process emits observations/rewards/done signals, trainer lives outside, and policies come back over a protocol or exported artifact.

### Broader Game-Engine Benchmarks

| Artifact | Stack | Why It Matters |
| --- | --- | --- |
| [Unity ML-Agents](https://unity-technologies.github.io/ml-agents/) | Unity C# SDK + Python package | The most mature game-engine RL/IL workflow: Unity scenes as environments, Python trainers, executables/editor as envs, PPO/SAC/behavioral cloning/GAIL-style imitation support across versions. |
| [Unreal Learning Agents](https://dev.epicgames.com/community/learning/tutorials/8OWY/unreal-engine-learning-agents-introduction-5-3) | Unreal Engine plugin | Official Unreal direction for training AI characters with reinforcement and imitation learning, including physics-based animation and QA-bot use cases. |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) | Python + PyTorch | Baseline algorithm reference for PPO/SAC/TD3/DQN-family training, experiment conventions, and evaluation harnesses. |

These systems validate the same integration pattern: the engine is the environment and data-production layer; mature algorithm libraries often live out-of-process; model artifacts are imported back into the runtime.

## Robotics, Locomotion, And Imitation Prior Art

### Mature Platforms

| Artifact | Stack | What It Proves | Transferable Pattern |
| --- | --- | --- | --- |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | NVIDIA Isaac Sim | Unified robot-learning framework with RL, imitation learning, motion planning, environments, controllers, and RL-library interfaces. | Treat locomotion as environment families plus manager/config layers, not one-off scenes. |
| [legged_gym](https://github.com/leggedrobotics/legged_gym) | Isaac Gym + PyTorch | GPU-based legged robot RL environments, rough-terrain locomotion, robot variants; now points users toward Isaac Lab. | Terrain curriculum, domain randomization, and vectorized simulation are central to locomotion success. |
| [MuJoCo Playground](https://playground.mujoco.org/) | MuJoCo/MJX + JAX | Environments for training sim-to-real transferable policies, including locomotion and manipulation. | Use batched simulation, strict environment definitions, and cross-task evaluation suites. |
| [Brax](https://arxiv.org/abs/2106.13281) | JAX differentiable physics | Large-scale rigid-body simulation for RL and gradient-based optimization. | Differentiable/vectorized physics is the high-throughput target for research velocity, even if Bevy physics is the product runtime. |

### Physics-Based Locomotion And Character Control

| Artifact | Key Idea | Why It Matters |
| --- | --- | --- |
| [DeepMimic](https://arxiv.org/abs/1804.02717) / [project page](https://xbpeng.github.io/projects/DeepMimic/index.html) | Combine motion-imitation objectives with RL task objectives to train physics-based characters from motion clips. | Canonical prior art for humanoid/creature locomotion, acrobatics, martial arts, and multi-skill motion imitation. |
| [AMP](https://arxiv.org/abs/2104.02180) | Adversarial Motion Priors learn style rewards from unstructured motion clips, avoiding manual clip selection and reward engineering. | Directly relevant if the goal is natural-looking active ragdoll or character locomotion from animation data. |
| [RMA](https://arxiv.org/abs/2107.04034) | Base policy plus adaptation module for real-time quadruped adaptation to terrain/payload/wear. | Useful transfer/continual-learning architecture: split policy from online adaptation context. |
| [Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652) | Train in simulation, transfer to ANYmal robot, recover from falls and follow velocity commands. | Strong evidence for sim-to-real via fast simulation, domain randomization, and robust control policies. |
| [OpenAI Rubik's Cube / ADR](https://arxiv.org/abs/1910.07113) / [OpenAI post](https://openai.com/index/solving-rubiks-cube/) | Automatic domain randomization grows randomized environment difficulty for sim-to-real transfer. | Transferable to physics training as automatic environment-parameter widening. |

## Technique-Specific Findings

### Curriculum Learning

Important sources:

- [Curriculum Learning, Bengio et al. 2009](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf)
- [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)
- [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300)
- [POET](https://arxiv.org/abs/1901.01753)

Useful patterns for `bevy-gym`:

- Treat curriculum as environment-parameter scheduling, not as trainer-specific hidden state.
- Store curriculum state per environment instance: difficulty, terrain seed, reset distribution, task family, and recent success metrics.
- Support teacher-driven task selection: trainer asks for environment variants with high learning progress, not just uniform random sampling.
- Implement reverse curriculum for sparse-goal tasks: reset near successful states first, then expand initial-state distance.
- Treat POET-style open-ended training as a future extension: generate environment variants, archive policies, and allow policy transfer between variants.

Suggested local API direction:

```rust
pub trait CurriculumEnv {
    type CurriculumState;
    type TaskSpec;

    fn task_spec(&self) -> Self::TaskSpec;
    fn curriculum_state(&self) -> Self::CurriculumState;
    fn set_task_spec(&mut self, task: Self::TaskSpec);
}
```

The exact trait should be shaped by `rl-traits`, but the core idea is stable: environment difficulty belongs in typed task/reset state that can be logged, replayed, and evaluated.

### Continual Learning

Important sources:

- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)
- Teacher-student curriculum learning's attention to forgetting across subtasks
- RMA-style adaptation modules for changing terrain/context

Rust-specific finding: no mature Rust/Burn continual-learning framework surfaced for RL. Continual learning will likely need local implementation around:

- Evaluation across old task distributions after every training phase.
- Replay buffers that preserve older tasks or demonstrations.
- Regularization methods such as EWC if catastrophic forgetting becomes measurable.
- Modular policies or adaptation modules for terrain/context variation.

For `bevy-gym`, the first continual-learning step should not be EWC. It should be a regression harness: freeze checkpoints and evaluate them across a fixed matrix of environment variants, seeds, and curriculum levels.

### Transfer Learning And Sim-To-Real

Important sources:

- [RMA](https://arxiv.org/abs/2107.04034)
- [OpenAI ADR / Rubik's Cube](https://arxiv.org/abs/1910.07113)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

Transfer patterns:

- Train in fast/headless simulation, evaluate in slower/rendered/product simulation.
- Randomize dynamics, morphology, friction, mass, motor strength, observation noise, and terrain.
- Separate base policy from adaptation context when the environment changes online.
- Export policies through stable artifacts: Burn `.mpk`, ONNX, or an explicit local checkpoint format.
- Keep observation/action schemas versioned. Transfer failures often come from silent schema drift.

For Rust/Bevy, the near-term transfer target is not real robots. It is transfer between:

1. Simple classic-control Rust envs.
2. Bevy ECS headless physics envs.
3. Rendered Bevy scenes.
4. Optional Godot/Python/SB3/ONNX reference policies.

### Imitation Learning

Important sources:

- [DAgger](https://arxiv.org/abs/1011.0686)
- [PMLR DAgger page](https://proceedings.mlr.press/v15/ross11a.html)
- [GAIL](https://arxiv.org/abs/1606.03476)
- [DeepMimic](https://arxiv.org/abs/1804.02717)
- [AMP](https://arxiv.org/abs/2104.02180)
- [Unity ML-Agents](https://unity-technologies.github.io/ml-agents/)

Rust-specific finding: no mature Burn-native imitation learning crate surfaced in this pass. The likely implementation path is staged:

1. Behavioral cloning from recorded observation-action pairs.
2. DAgger-style dataset aggregation if an expert policy or scripted controller can label states.
3. GAIL/AMP later, once discriminator training and trajectory datasets are stable.

For `bevy-gym`, the highest-leverage prerequisite is a demo/replay data model:

```text
episode_id
env_id
task_spec
seed
timestep
observation
action
reward
terminated
truncated
info
expert_action optional
source_policy optional
schema_version
```

Do not start with adversarial imitation. Start by making demonstrations reproducible, typed, and versioned.

### Physics-Based Locomotion

The broader state of the art has converged on a pattern:

- PPO/SAC-style continuous control.
- Massive vectorized simulation.
- Terrain and dynamics randomization.
- Curriculum over terrain/task difficulty.
- Strict reset and termination design.
- Reward terms for velocity tracking, energy, contacts, posture, foot slip, and survival.
- For character motion quality: imitation objectives, DeepMimic-style motion tracking, or AMP-style learned motion priors.

Rust/Bevy gap:

- Bevy has physics engines and RL environment surfaces.
- Rust has RL algorithm crates.
- The public Rust ecosystem does not yet show a mature, maintained, Burn-native active-ragdoll or quadruped locomotion training stack comparable to Isaac Lab or legged_gym.

Practical sequence:

1. CartPole or Pendulum in current `bevy-gym`.
2. Continuous MountainCar/Pendulum with continuous action support.
3. 2D hopper or walker with Rapier/Avian.
4. 3D simple rigid-body walker.
5. Ragdoll/active character with scripted-controller demonstrations.
6. DeepMimic/AMP-style imitation after the replay data path exists.

## Integration Patterns

### Pattern A: Fully Rust-Native

```text
Bevy ECS envs -> rl-traits -> bevy-gym -> ember-rl / burn-rl / local Burn trainer -> .mpk checkpoint
```

Pros:

- One language and one process.
- Strong typing around observations/actions/task specs.
- Good fit for local headless Bevy schedules.
- No Python bridge overhead.

Cons:

- Rust RL algorithm maturity is limited.
- Imitation/curriculum/locomotion examples are sparse.
- GPU/backend maturity must be validated per Burn backend.

Use this for the main project direction.

### Pattern B: Engine Environment With Python Trainer

```text
Bevy/Godot executable -> socket/REST/shared memory protocol -> SB3/RLlib/PyTorch trainer -> ONNX/Burn import or runtime policy bridge
```

Pros:

- Mature algorithms and debugging tools.
- Easier comparison against published baselines.
- Godot RL Agents and Unity ML-Agents validate the pattern.

Cons:

- Schema and process synchronization costs.
- More moving parts.
- Less Rust-native.

Use this as an oracle/baseline path, especially for PPO/SAC locomotion and imitation methods that Burn crates do not yet cover.

### Pattern C: Offline Dataset And Checkpoint Boundary

```text
Environment rollouts -> versioned trajectory dataset -> BC/DAgger/GAIL/AMP training -> policy artifact -> evaluation matrix
```

Pros:

- Best for imitation and transfer.
- Decouples environment collection from trainer iteration.
- Makes reproducibility and regression testing easier.

Cons:

- Requires careful schema design.
- Less interactive than direct online training.

Use this before any serious imitation-learning work.

## Recommended Architecture For This Repo

### Keep

- `rl-traits` as the shared vocabulary.
- `bevy-gym` as the parallel ECS environment plugin.
- Headless-first training through Bevy schedules.
- Optional rendering as inspection/debug tooling, not a training requirement.
- `TrainingSession`/checkpoint/logging conventions from the current CartPole example where they fit.

### Add Next

1. Continuous action-space support and examples.
2. A typed task/curriculum spec that can be attached to each environment entity.
3. A rollout/demo writer and reader.
4. A deterministic evaluation harness over seeds, tasks, and checkpoints.
5. A passive physics stability probe before any learning gate for locomotion.
6. A Python/SB3 bridge or ONNX import spike as an external baseline.

### Avoid For Now

- Starting with full active-ragdoll humanoid training.
- Treating imitation learning as only a reward-function problem.
- Coupling curriculum logic directly into one algorithm implementation.
- Assuming Burn RL crates have parity with Stable-Baselines3 or Isaac Lab.
- Training from rendered pixels before state-vector environments are stable.

## Implementation Roadmap

### Phase 1: Rust-Native Control Baseline

Goal: prove `bevy-gym` can support more than discrete CartPole.

- Add or port Pendulum and MountainCarContinuous.
- Represent continuous actions explicitly in `rl-traits` or local extension traits.
- Train with `ember-rl` SAC/PPO if available; otherwise add a narrow local continuous-control spike.
- Verify evaluation with fixed seeds and checkpoint reload.

Exit criteria:

- Reproducible learning curve.
- Saved checkpoint beats random policy.
- Observation/action schema documented.

### Phase 2: Curriculum Substrate

Goal: make task difficulty observable and controllable.

- Add per-env `TaskSpec`/difficulty metadata.
- Log task spec with every episode.
- Add hand-authored curriculum schedules.
- Add simple teacher selection based on recent success or learning progress.

Exit criteria:

- Same trainer can run fixed distribution, linear curriculum, and adaptive curriculum.
- Evaluation reports performance by difficulty bin.

### Phase 3: Bevy Physics Control

Goal: train a low-dimensional physics agent before active ragdolls.

- Choose Avian or Rapier for the first environment.
- Start with 2D hopper/walker or simple 3D body, not humanoid.
- Add passive simulation health checks: no NaN transforms, stable contacts, bounded velocities, reset validity.
- Keep rendering optional.

Exit criteria:

- Random policy simulation is stable for long runs.
- PPO/SAC policy improves over random.
- Reset, termination, and reward terms are documented.

### Phase 4: Demonstrations And Imitation

Goal: support BC/DAgger before GAIL/AMP.

- Record scripted-controller or human/expert trajectories.
- Train behavioral cloning on observation-action records.
- Run policy rollouts and aggregate expert labels for DAgger if an expert is available.
- Keep adversarial methods behind a later research spike.

Exit criteria:

- Demo dataset can be replayed and validated against schema.
- BC policy outperforms random and approximates expert on held-out episodes.

### Phase 5: Transfer And Motion Quality

Goal: move from task success to robust motion.

- Add environment randomization.
- Add policy evaluation across held-out parameters.
- Explore adaptation-module designs inspired by RMA.
- Explore DeepMimic/AMP only after motion datasets and discriminator training are feasible.

Exit criteria:

- Checkpoint ranking is based on held-out distributions, not only training reward.
- Transfer failures are attributable to environment parameters or schema mismatches.

## Risk Register

| Risk | Severity | Evidence | Mitigation |
| --- | --- | --- | --- |
| Burn RL ecosystem maturity | High | Burn and Burn-native RL crates exist, but broad locomotion/imitation examples are sparse. | Keep algorithms modular; preserve Python/SB3 baseline option. |
| Physics simulation instability | High | Locomotion training amplifies small contact/reset defects. | Add passive physics probes before training gates. |
| Active ragdoll complexity | High | Mature examples are mostly Isaac/MuJoCo/Python or Unity/Unreal, not Rust/Burn. | Stage through simple continuous-control and low-dimensional walkers. |
| Curriculum hidden state | Medium | Curriculum can become algorithm-specific and unreproducible. | Store task specs and difficulty in env metadata and logs. |
| Imitation data drift | Medium | Demonstrations require stable observation/action schemas. | Version demo schemas and include env seed/task/checkpoint metadata. |
| Transfer false positives | Medium | A policy can overfit one simulator or one seed range. | Evaluation matrix over seeds, task variants, and randomized parameters. |
| Crate churn | Medium | Bevy, Burn, and RL crates are moving quickly. | Pin versions and isolate adapters. |

## Prior-Art Index

### Rust And Burn

- [Burn](https://docs.rs/burn/latest/burn/) - Rust deep learning framework; current crate metadata found 0.21.0.
- [burn-rl](https://docs.rs/burn-rl/latest/burn_rl/) - official Burn RL module; current crate metadata found 0.21.0.
- [ember-rl](https://crates.io/crates/ember-rl) - Burn DQN/PPO/SAC implementations; current crate metadata found 0.3.4.
- [rl-traits](https://crates.io/crates/rl-traits) - shared RL environment/policy/agent traits; current crate metadata found 0.2.2.
- [bevy-gym](https://crates.io/crates/bevy-gym) - Bevy ECS plugin for parallel RL environment simulation; current crate metadata found 0.3.3.
- [rl4burn](https://docs.rs/rl4burn/latest/rl4burn/) - Burn RL algorithms and utilities; current crate metadata found 0.1.1.
- [rwrd](https://docs.rs/rwrd/latest/rwrd/) - Burn-powered RL framework; current crate metadata found 0.4.0.
- [r2l-burn](https://docs.rs/r2l-burn/latest/r2l_burn/) and [r2l-gym](https://crates.io/crates/r2l-gym) - Rust-first RL crates with Burn/Gym split; current metadata found 0.0.2-rc2.
- [rlevo](https://crates.io/crates/rlevo) - Burn-based evolutionary RL; current crate metadata found 0.2.0.
- [sb3-burn](https://github.com/will-maclean/sb3-burn) - Stable-Baselines3-inspired Burn project.
- [border](https://docs.rs/border/latest/border/) - `tch`-based Rust RL library with broader algorithm set.
- [rurel](https://docs.rs/rurel/latest/rurel/) - older reusable tabular RL library.
- [gym-rs](https://docs.rs/gym-rs/latest/gym_rs/) - pure Rust Gym-like environments.
- [scivex-rl](https://docs.rs/scivex_rl/latest/scivex_rl/) - early RL envs and algorithms.
- [reinforcex](https://crates.io/crates/reinforcex) - Rust deep RL framework.
- [train-station](https://crates.io/crates/train-station) - Rust ML library with PPO continuous examples.
- [rlox](https://github.com/wojciechkpl/rlox) - Rust-accelerated RL project claiming PPO/SAC/DQN/TD3/A2C and many more algorithms, Gymnasium-compatible environments, vectorized envs, dashboards, and mixed NN backends. Treat as a high-interest project to audit, not as a proven dependency yet.

### Bevy And Rust Game Examples

- [bevy_rl](https://github.com/stillonearth/bevy_rl) - Gym-style Bevy RL API, multi-agent, pixel observations, REST.
- [bevy_rl_shooter](https://github.com/stillonearth/bevy_rl_shooter) - multi-agent FPS environment.
- [bevy_quadruped_neural_control](https://github.com/stillonearth/bevy_quadruped_neural_control) - Unitree A1 / MuJoCo / Bevy RL example.
- [lunar-lander-tch](https://github.com/Robsutar/lunar-lander-tch) - LunarLander in Rust with Bevy/Rapier/tch.
- [rusty_RL](https://github.com/leesg0107/rusty_RL) - Bevy drone simulation with a `tch` PPO implementation and ECS integration; explicitly work-in-progress.
- [bevy-3d-dodge](https://github.com/ggand0/bevy-3d-dodge) - Bevy 3D dodge game with a Gymnasium-compatible Python training interface, gRPC server, Stable-Baselines3, SAC/PPO/DQN, and headless high-FPS training mode.
- [bevy-snake-ai](https://github.com/cswinter/bevy-snake-ai) - Bevy game with trained RL opponents.
- [trictrac](https://github.com/mmai/trictrac) - game project referencing Burn DQN/PPO/SAC training scripts.
- [Avian](https://docs.rs/avian3d/latest/avian3d/) - Bevy-native ECS physics.
- [bevy_rapier3d](https://crates.io/crates/bevy_rapier3d) - Bevy plugin for Rapier 3D physics.
- [bevy-tnua](https://docs.rs/bevy-tnua) - Bevy floating character controller with Rapier/Avian integrations.

### Godot, Unity, Unreal, And Training Frameworks

- [Godot RL Agents](https://edbeeching.github.io/papers/gdrl.html) - Godot environments with Python training interfaces.
- [godot-rust/gdext](https://godot-rust.github.io/) - Rust bindings for Godot 4.
- [Unity ML-Agents](https://unity-technologies.github.io/ml-agents/) - mature game-engine RL/IL system.
- [Unreal Learning Agents](https://dev.epicgames.com/community/learning/tutorials/8OWY/unreal-engine-learning-agents-introduction-5-3) - Unreal plugin for reinforcement/imitation learning.
- [Gymnasium](https://gymnasium.farama.org/) - maintained Gym successor and single-agent environment API standard.
- [PettingZoo](https://pettingzoo.farama.org/) - multi-agent API standard with AEC and Parallel APIs.
- [RLlib MultiAgentEnv](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html) - scalable multi-agent environment and policy-mapping model.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) - common algorithm baseline in Python/PyTorch.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - readable single-file reference implementations for PPO, DQN, C51, DDPG, TD3, SAC, PPG, and variants.
- [CORL](https://corl-team.github.io/CORL/) - clean offline RL reference implementations including BC, TD3+BC, Decision Transformer, CQL, AWAC, IQL, and related methods.
- [RLtools](https://github.com/rl-tools/rl-tools) - C++ high-performance RL project with PPO/SAC/TD3 examples and quadrotor/continuous-control emphasis; useful as a performance and API benchmark, not a Rust dependency.
- [Minari](https://minari.farama.org/) - Farama offline RL dataset API.
- [RLDS](https://github.com/google-research/rlds) - dataset ecosystem for episodic sequential-decision data, demonstrations, offline RL, and imitation learning.
- [D4RL](https://arxiv.org/abs/2004.07219) - offline RL benchmark suite spanning diverse data sources and tasks.

### Papers And Robotics Platforms

- [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf)
- [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)
- [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300)
- [POET](https://arxiv.org/abs/1901.01753)
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)
- [DAgger](https://arxiv.org/abs/1011.0686)
- [GAIL](https://arxiv.org/abs/1606.03476)
- [DeepMimic](https://arxiv.org/abs/1804.02717)
- [AMP](https://arxiv.org/abs/2104.02180)
- [RMA](https://arxiv.org/abs/2107.04034)
- [OpenAI ADR / Rubik's Cube](https://arxiv.org/abs/1910.07113)
- [Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [Brax](https://arxiv.org/abs/2106.13281)

## Second-Pass Gaps And Additions

The first pass correctly identified the main Rust-native gap, but a second pass adds three important themes: Bevy/Python bridge examples are more common than pure Burn examples, dataset standards matter earlier than adversarial imitation, and multi-agent API design should be planned before it is needed.

### Additional Rust And Bevy Examples

`sb3-burn` is worth treating carefully. It is strongly aligned with this repo's Burn direction, but its README currently describes DQN as implemented, SAC as in progress, and PPO as planned, with Rust Gridworld, CartPole, Pendulum, MountainCar, and probe environments. That makes it a good API and design reference, but not yet the stable PPO/SAC replacement for SB3.

The Bevy examples reinforce the hybrid recommendation:

- `rusty_RL` uses Bevy for a drone environment and `tch` for PPO. Its own README calls the RL implementation work-in-progress, but it is directly relevant for drone-control loop shape.
- `bevy-3d-dodge` uses Bevy as a high-FPS game/server and trains through a Gymnasium-compatible Python interface with Stable-Baselines3. This is close to the recommended "engine as environment, Python as baseline oracle" bridge.
- `bevy_quadruped_neural_control` remains the most relevant found Bevy locomotion prior art because it ties Bevy, a quadruped task, MuJoCo, SB3, and ONNX-style deployment ideas together.
- `rlox` looks like an ambitious Rust-accelerated RL stack with many claimed algorithms and environment features. It should be audited, but its breadth makes it especially important to verify code maturity before taking dependency risk.

Implication for `bevy-gym`: keep the Rust-native path, but design the environment boundary so that a Gymnasium/SB3 process bridge is not a bolt-on afterthought. The bridge can be experimental, but the observation/action/reward/termination schema should be shared with the Rust trainer.

### Environment API Standards To Mirror

Gymnasium's current `Env.step()` returns observation, reward, `terminated`, `truncated`, and `info`, explicitly separating MDP termination from outside-the-MDP truncation. The local `bevy-gym` documentation already cares about bootstrapping signals through `EpisodeStatus`; preserve that distinction as a compatibility invariant.

For future multi-agent work:

- PettingZoo distinguishes sequential Agent Environment Cycle APIs from simultaneous Parallel APIs.
- RLlib maps many agents to one or more policies through a user-defined mapping function and returns dictionaries keyed by agent ID for observations, rewards, terminations, truncations, and infos.

Implication for `bevy-gym`: do not encode "one environment entity equals one agent forever" too deeply. A future multi-agent extension should allow one Bevy world/env instance to contain multiple agent IDs, each with its own action request and observation stream, while still supporting vectorized batches of independent worlds.

### Dataset Standards For Imitation And Offline RL

The first-pass demo schema is directionally right, but Minari, RLDS, and D4RL sharpen it:

- Minari is the maintained Farama dataset API for offline RL datasets generated from Gymnasium-style environments.
- RLDS is an episodic dataset ecosystem for sequential decision making, including RL, demonstrations, offline RL, and imitation learning.
- D4RL is still important conceptually because it stressed datasets from hand-designed controllers, human demonstrators, multitask runs, and mixtures of policies, not only online agents at different training checkpoints.
- CORL and CleanRL are valuable because they keep algorithm implementations readable and benchmarkable, even when they are not modular libraries.

Implication for `bevy-gym`: make rollout data a first-class artifact early. The schema should include:

```text
dataset_id
schema_version
environment_id
environment_version
task_spec
seed
episode_id
timestep
agent_id optional
policy_id optional
observation
action
reward
terminated
truncated
info
expert_action optional
source enum: random | scripted | human | trained_policy | imported
```

This is enough to support behavioral cloning, DAgger-style aggregation, offline evaluation, curriculum diagnostics, and transfer tests without committing to GAIL/AMP immediately.

### Additional Recommendation

Add a "compatibility lane" to the roadmap:

1. Keep the canonical Rust `rl-traits`/`bevy-gym` environment contract.
2. Add a Gymnasium-compatible process adapter for at least one simple task.
3. Compare Burn/`ember-rl` against SB3/CleanRL on the same environment and seeds.
4. Write rollouts in a Minari/RLDS-inspired local format.
5. Only then decide whether to port more algorithms into Rust or keep the bridge as the baseline oracle.

This lane directly reduces algorithm-risk while preserving the Rust-native destination.

## Recommendation

For `bevy-gym`, the best near-term research bet is not to chase a full Rust-native clone of Unity ML-Agents or Isaac Lab immediately. The stronger path is to make `bevy-gym` excellent at typed, parallel, reproducible environment execution, then layer curricula, demos, and evaluation on top.

Concrete next technical step:

1. Add a continuous-control example and action-space abstraction.
2. Add task/curriculum metadata to environment instances and logs.
3. Add a rollout/demo dataset schema.
4. Spike one Bevy physics control environment with passive stability checks.
5. Keep `ember-rl`/Burn as the preferred trainer, but compare at least one task against SB3 or another mature baseline.

That sequence gives the project a defensible Rust-native core while still borrowing the parts of the mature robotics/game-development world that are already known to work.
