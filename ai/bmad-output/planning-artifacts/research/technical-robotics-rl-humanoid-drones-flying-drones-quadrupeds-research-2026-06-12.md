---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - ../../../README.md
  - ../../../docs/plugins/bevy_gym_plugin.md
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Robotics reinforcement learning for humanoids, flying drones, and quadrupeds'
research_goals: 'Research simulated environments, open-source projects and examples, curricula, training environments, reward policies, optimization algorithms, best practices, lessons learned, state-of-the-art models, strategies, techniques, papers, projects, and reusable open-source 3D assets for custom training environments.'
user_name: 'Sagan'
date: '2026-06-12'
web_research_enabled: true
source_verification: true
---

# Robotics Reinforcement Learning for Humanoids, Flying Drones, and Quadrupeds

**Date:** 2026-06-12  
**Author:** Sagan  
**Research Type:** Technical

---

## Research Overview

This report surveys reinforcement learning for robotics with a focus on humanoid robots, flying drones, and quadrupeds. It emphasizes practical simulated environments, open-source projects, curriculum design, reward policy design, optimization algorithms, sim-to-real practice, reusable 3D assets, and implications for a custom Bevy-based training environment.

The main conclusion is straightforward: if the target is real robot control, use Bevy-gym as a fast custom environment harness, visualization layer, or game-like embodied AI sandbox, but validate low-level robot dynamics in a robotics simulator with proven articulation, contact, actuator, and sensor models. For current serious sim-to-real work, Isaac Lab plus RSL-RL/PPO is the most common production-style baseline for legged robots; MuJoCo Playground is the cleanest open GPU-accelerated alternative for rapid robot-learning iteration; drone work is split between high-throughput specialist stacks such as Aerial Gym, OmniDrones, Flightmare, and gym-pybullet-drones plus PX4/Gazebo for flight-stack validation.

Confidence is high for the broad stack and best-practice recommendations because the claims are grounded in official docs, project repositories, and primary papers. Confidence is medium for "state of the art" ranking because humanoid and drone RL are moving quickly, and many leading systems are demonstrated in papers before stable open-source releases.

---

## Executive Summary

Robotics RL is no longer a single "choose an algorithm" problem. The strongest systems combine massively parallel simulation, carefully limited observations, curriculum learning, domain randomization, reward shaping or motion priors, and deployment-aware policy interfaces. The recurring pattern across quadrupeds, humanoids, and drones is: train with much more variation than the nominal task; give the critic or teacher privileged information only during training; keep the deployed actor restricted to realistic sensors; validate in another simulator or flight stack; then deploy with conservative safety wrappers.

**Best starting points by embodiment:**

| Embodiment | Best first research stack | Why |
| --- | --- | --- |
| Quadruped locomotion | Isaac Lab + RSL-RL/PPO, or MuJoCo Playground | Most open examples, proven terrain curricula, strong sim-to-real history |
| Humanoid locomotion | Isaac Lab/Humanoid-Gym/Unitree RL Lab, then MuJoCo sim-to-sim | Humanoid policies are more sensitive to dynamics mismatch, so cross-simulator validation matters |
| Flying drones | Aerial Gym, OmniDrones, Flightmare, gym-pybullet-drones, PX4/Gazebo | Drone work needs different layers: fast RL, vision sensors, and real autopilot/SITL validation |
| Game-like fair embodied AI | Bevy-gym + rendered sensor wrappers | Good fit for true pixel/audio-style perception and procedural environments, weaker fit for real robot dynamics |

**Technical recommendation for this repo:** build Bevy-gym examples around the Gymnasium-style environment contract and fast parallel rollouts, but keep a boundary between "custom training environment for behavior" and "validated robot dynamics simulator." If the long-term goal is humanoid/quadruped/drone behavior for a game or embodied-agent demo, Bevy-gym can own the environment, sensors, curriculum, and reward decomposition. If the goal is real hardware deployment, Bevy-gym should interoperate with assets, traces, or policies from Isaac Lab, MuJoCo, or PX4/Gazebo rather than becoming the sole physics truth.

---

## Table of Contents

1. Scope Confirmation
2. Simulator and Environment Landscape
3. Open-Source Projects and Examples
4. Curriculum and Training Environment Design
5. Reward Policies
6. Optimization Algorithms and Policy Architectures
7. Sim-to-Real Best Practices and Lessons
8. State-of-the-Art Papers and Strategies
9. Reusable 3D Assets
10. Bevy-gym Implications and Roadmap
11. Source Verification
12. Implementation Addendum: First Bevy-gym Spike

---

## 1. Scope Confirmation

**Research topic:** robotics reinforcement learning for humanoids, flying drones, and quadrupeds.

**Research goals:** identify simulated environments, open-source projects, curriculum patterns, reward designs, optimization algorithms, best practices, lessons learned, state-of-the-art strategies, and reusable open-source 3D assets for a custom training environment.

**Terminology note:** "Embodied RL" and "sim-to-real robot learning" are the standard terms for this space. "Humanoid drones" is interpreted here as autonomous humanoid robots or humanoid embodied agents. If the intended meaning is a flying humanoid-shaped vehicle, the closest reusable evidence is from drone and custom multirotor work, not mainstream humanoid locomotion.

---

## 2. Simulator and Environment Landscape

### High-Level Comparison

| Stack | Best use | Strengths | Risks |
| --- | --- | --- | --- |
| Isaac Lab | GPU-scale robot learning, sim-to-real, sensors, USD assets | Active successor to Isaac Gym/Orbit/OmniIsaacGymEnvs; supports SKRL, RSL-RL, RL-Games, SB3 | Heavy NVIDIA/Omniverse stack; migration details matter |
| legged_gym | Reading proven quadruped environment/reward/curriculum patterns | Influential ANYmal rough-terrain code and sim-to-real components | Built on deprecated Isaac Gym; maintainers point users to Isaac Lab |
| MuJoCo Playground | Open GPU-accelerated robot-learning iteration | Apache-2.0, MJX/Warp, quadruped and biped environments | Younger ecosystem than Isaac Lab for full sensor/rendered robotics workflows |
| Genesis World | Emerging general physical-AI simulator | Multi-physics, Pythonic interface, asset parsing, sensors, renderer | New and fast-moving; verify stability per task before committing |
| Aerial Gym Simulator | Massive parallel multirotor RL and sensor tasks | Isaac Gym based; GPU controllers; depth/segmentation/LiDAR-style ray casting | Isaac Lab support still under development at the source checked |
| OmniDrones | Multi-rotor RL on Isaac Sim | Benchmark tasks, sensor/control modes, Isaac Sim integration | Smaller project surface; check Isaac Sim version compatibility |
| Flightmare | Vision-rich quadrotor sim | Unity renderer plus decoupled physics; hundreds of quadrotors; sensor suite | Older stack; likely more maintenance risk than active Isaac/MuJoCo paths |
| gym-pybullet-drones | Lightweight quadrotor RL examples | Gymnasium/PyBullet, SB3 PPO examples, MIT license, easy local experiments | Lower fidelity than Isaac/Gazebo/PX4 for final control validation |
| Gazebo/PX4 | Flight stack validation and SITL | Autopilot integration, open models/worlds, quadrotor/VTOL/plane support | Not the fastest path for massive RL training |
| Bevy-gym | Custom ECS simulation and game-style embodied AI | Headless parallel Bevy rollouts, FixedUpdate, message-driven policy integration | Not a robotics-grade dynamics stack by itself |

### Isaac Lab and Isaac Gym Lineage

Isaac Lab is the current NVIDIA robot-learning framework to treat as primary for new work. Its RL documentation lists SKRL, RSL-RL, RL-Games, and Stable-Baselines3 as supported libraries and compares vectorized/distributed support across them. Source: [Isaac Lab RL framework comparison](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html).

Isaac Gym Preview and IsaacGymEnvs are deprecated in favor of Isaac Lab. NVIDIA's migration docs explicitly describe IsaacGymEnvs and Isaac Gym Preview as deprecated and document API differences such as quaternion convention, joint ordering, and simulation parameter defaults. Source: [Isaac Lab migration from IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html).

**Implication:** use older Isaac Gym projects such as legged_gym, Humanoid-Gym, Unitree RL Gym, and Aerial Gym as technical references, but budget migration work or choose Isaac Lab-native examples for new code.

### MuJoCo Playground

MuJoCo Playground is an open-source GPU-accelerated robot-learning framework built with MuJoCo MJX. Its repository advertises classic control, quadruped and bipedal locomotion, manipulation, and vision-based support through MJWarp Batch Renderer. It is Apache-2.0 licensed. Source: [MuJoCo Playground GitHub](https://github.com/google-deepmind/mujoco_playground).

**Implication:** this is the strongest open alternative when you want a lighter, more inspectable stack than Isaac Lab and can live inside JAX/MJX/Warp conventions.

### Genesis World

Genesis World is an emerging physical-AI simulation platform with a unified multi-physics engine, photo-realistic renderer, cross-platform compiler, asset parsing for URDF/MJCF/OBJ/GLB/USD, sensors, parallel environments, and a Pythonic interface. Source: [Genesis World GitHub](https://github.com/Genesis-Embodied-AI/genesis-world).

**Implication:** promising for custom robotics and embodied-AI environments, but treat it as an evaluation candidate until task-specific stability, asset import, and training throughput are proven.

### Drone-Specific Simulators

Aerial Gym Simulator is designed for multirotor RL at scale, including standard quadrotors, fully actuated platforms, arbitrary multirotor configurations, GPU-resident geometric controllers, and fast depth/segmentation/LiDAR-style sensors. Source: [Aerial Gym Simulator GitHub](https://github.com/ntnu-arl/aerial_gym_simulator).

Flightmare provides a large multi-modal sensor suite, 3D point-cloud extraction, an RL API that simulates hundreds of quadrotors in parallel, and Unity-based rendering decoupled from physics. Source: [Flightmare GitHub](https://github.com/uzh-rpg/flightmare).

gym-pybullet-drones provides Gymnasium/PyBullet environments for single and multi-agent quadrotor control with SB3 PPO examples and an MIT license. Source: [gym-pybullet-drones GitHub](https://github.com/learnsyslab/gym-pybullet-drones).

OmniDrones is an open-source drone RL platform on NVIDIA Isaac Sim with benchmark tasks, multiple drone models, sensor modalities, control modes, and RL baselines. Sources: [OmniDrones docs](https://omnidrones.readthedocs.io/) and [OmniDrones arXiv](https://arxiv.org/abs/2309.12825).

PX4/Gazebo models are useful for SITL and flight-stack validation. The PX4 model repository includes models, worlds, and a script that starts Gazebo simulation with PX4. Source: [PX4 Gazebo Models](https://github.com/PX4/PX4-gazebo-models).

---

## 3. Open-Source Projects and Examples

### Quadrupeds

**legged_gym** remains one of the most important reference codebases for quadruped locomotion. It includes the environment used to train ANYmal and other robots on rough terrain, with sim-to-real components such as actuator networks, friction and mass randomization, noisy observations, and random pushes. The maintainers state that the work has moved to Isaac Lab and that legged_gym receives limited updates. Source: [legged_gym GitHub](https://github.com/leggedrobotics/legged_gym).

**RSL-RL** is a robotics-first learning library with PPO, student-teacher distillation, multi-GPU support, and integrations with Isaac Lab, Legged Gym, MuJoCo Playground, and related stacks. Source: [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl).

**MuJoCo Playground** includes quadruped locomotion tasks and should be treated as a current open benchmark environment for quadruped policies. Source: [MuJoCo Playground GitHub](https://github.com/google-deepmind/mujoco_playground).

### Humanoids

**Humanoid-Gym** is an Isaac Gym based RL framework for humanoid locomotion with emphasis on zero-shot sim-to-real transfer and sim-to-sim validation in MuJoCo. It reports validation on RobotEra XBot-S and XBot-L hardware. Source: [Humanoid-Gym project page](https://sites.google.com/view/humanoid-gym/).

**Unitree RL Lab** is Isaac Lab based and supports Unitree Go2, H1, and G1-29dof robots. Source: [unitree_rl_lab GitHub](https://github.com/unitreerobotics/unitree_rl_lab).

**Unitree RL Gym** is an older Unitree RL implementation supporting Go2, H1, H1_2, and G1 with an explicit workflow of Train -> Play -> Sim2Sim -> Sim2Real. Source: [unitree_rl_gym GitHub](https://github.com/unitreerobotics/unitree_rl_gym).

**Berkeley Humanoid** provides a learning-focused humanoid research platform and reports robust outdoor locomotion with a simple RL controller and light domain randomization. Source: [Berkeley Humanoid project page](https://berkeley-humanoid.com/).

**HOVER** is a recent humanoid whole-body controller that uses full-body kinematic motion imitation as a common abstraction and distills multiple control modes into one policy. Source: [HOVER project page](https://hover-versatile-humanoid.github.io/).

**GR00T** is adjacent to low-level locomotion RL rather than a replacement for it. NVIDIA's GR00T N1.7 repository describes an open vision-language-action model for generalized humanoid manipulation skills, with fine-tuning and inference workflows. Source: [NVIDIA Isaac GR00T GitHub](https://github.com/Nvidia/Isaac-GR00T).

### Flying Drones

**Aerial Gym Simulator** is the best match for high-throughput multirotor RL with state and sensor tasks. Source: [Aerial Gym Simulator GitHub](https://github.com/ntnu-arl/aerial_gym_simulator).

**Flightmare** is valuable when visual perception and photorealistic-ish rendering matter more than newest framework support. Source: [Flightmare GitHub](https://github.com/uzh-rpg/flightmare).

**gym-pybullet-drones** is the simplest practical starter for SB3/PyBullet/Gymnasium quadrotor RL. Source: [gym-pybullet-drones GitHub](https://github.com/learnsyslab/gym-pybullet-drones).

**Agilicious** is not just a simulator; it is an open-source and open-hardware agile quadrotor stack that supports model-based and neural-network-based controllers and has been used across many agile-flight papers. Source: [Agilicious GitHub](https://github.com/uzh-rpg/agilicious).

---

## 4. Curriculum and Training Environment Design

### General Curriculum Pattern

Use a staged curriculum with explicit gates:

1. **Stabilization:** standing, hovering, attitude control, or simple velocity tracking.
2. **Command tracking:** expand target velocity, yaw, height, target path, or gait command range.
3. **Perturbations:** pushes, payload changes, wind, latency, mass/friction randomization, damaged actuator masks.
4. **Environment complexity:** slopes, stairs, rubble, gates, clutter, dynamic obstacles, occlusions.
5. **Sensor realism:** noise, latency, missing data, lighting variation, camera blur, depth dropout, IMU bias.
6. **Deployment constraints:** action limits, rate limits, torque/thrust saturation, controller frequency, onboard compute.
7. **Out-of-distribution evaluation:** new terrain, new track layouts, unseen masses, altered lighting, cross-simulator validation.

### Quadruped Curriculum

Start with flat-ground velocity tracking, then expand command ranges and terrain. Add terrain curriculum, randomized friction/mass, noisy observations, actuator delay, and random pushes before trying rich exteroception. Prior work shows proprioception-only policies can generalize surprisingly far if trained against sufficient variability. Source: [Learning Quadrupedal Locomotion over Challenging Terrain](https://leggedrobotics.github.io/rl-blindloco/).

Useful progression:

- Flat standing and slow walking.
- Omnidirectional velocity tracking.
- Slopes, stairs, gaps, stepping stones, rubble, deformable-looking terrain.
- Disturbance recovery and fall prevention.
- Payload and morphology randomization.
- Vision/LiDAR only after a strong blind/proprioceptive baseline exists.

### Humanoid Curriculum

Humanoids need narrower early gates because falls are frequent and dynamics mismatch is larger than quadrupeds. Start with upright stabilization and low-speed walking, then expand to turning, push recovery, uneven terrain, and whole-body tasks. Use motion imitation when the target behavior should look human-like, and distill multiple command modes when the same body must navigate, manipulate, and switch tasks.

Useful progression:

- Pose stabilization and fall recovery.
- Root velocity tracking on flat ground.
- Commanded turns and heading changes.
- Terrain and push recovery.
- Motion tracking from retargeted mocap or scripted references.
- Multi-mode control: root velocity, head, arms, hands, upper-body task modes.
- Sim-to-sim and residual dynamics compensation before real deployment.

Sources: [Humanoid-Gym](https://sites.google.com/view/humanoid-gym/), [HOVER](https://hover-versatile-humanoid.github.io/), [ASAP](https://arxiv.org/abs/2502.01143).

### Flying Drone Curriculum

Drones should separate low-level flight control from perception-heavy navigation. Start with hover and attitude control, then trajectory tracking, gates, clutter, wind, and sensor degradation. Racing and cluttered navigation should train on randomized short segments before full tracks so the policy sees many local maneuvers.

Useful progression:

- Hover at fixed height and yaw.
- Position/velocity/attitude tracking.
- Waypoint tracking and smooth trajectory following.
- Gates and narrow passages.
- Wind, payload, rotor constants, camera/IMU noise.
- Visual-only or visual-inertial input.
- Full track or cluttered course.
- PX4/Gazebo SITL validation for autopilot integration.

Sources: [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator), [OmniDrones](https://arxiv.org/abs/2309.12825), [Champion-level drone racing using deep RL](https://www.nature.com/articles/s41586-023-06419-4).

### Environment Design Rules

- Keep training observations, critic observations, and logging observations separate.
- Let the critic/teacher see privileged state; keep the deployable actor restricted to realistic sensors.
- Make reset distributions explicit and versioned.
- Log reward components separately from total reward.
- Record failed trajectories and termination reasons.
- Use deterministic evaluation separately from exploration rollouts.
- Build a "no privileged information" validation mode early.
- For game-like fair AI, use rendered pixels, audio waveforms, proprioception, and IMU-like channels instead of direct target vectors.

---

## 5. Reward Policies

### Reward Components by Embodiment

| Embodiment | Positive rewards | Penalties | Termination signals |
| --- | --- | --- | --- |
| Quadruped | command velocity tracking, body height, heading alignment, foot clearance, terrain progress | torque, action rate, joint limit, foot slip, collision, roll/pitch error, undesired contacts | fall, base height too low, joint limit, timeout |
| Humanoid | upright balance, root velocity, foot placement, imitation pose, contact timing, task progress | energy, jerk, self-collision, arm flailing, knee collapse, head/torso deviation, foot scuffing | fall, unsafe contact, body height, joint limits |
| Flying drone | position/velocity tracking, gate progress, time reduction, yaw alignment, hover stability | collision, control effort, jerk/snap, thrust saturation, attitude error, distance from corridor | crash, missed gate, out-of-bounds, timeout |

### Reward Design Lessons

**Dense rewards are useful, but make them auditable.** Dense reward shaping accelerates early locomotion and hover learning, but can hide reward hacking. Record every component, chart each term, and inspect videos of high-reward failures.

**Use curricula before over-shaping.** If the agent never sees successful trajectories, expand the curriculum or initialize from demonstrations before adding brittle rewards.

**Motion priors reduce hand reward complexity.** AMP replaces hand-designed low-level style objectives with adversarial style rewards learned from motion clips, while still allowing simple task rewards. Source: [AMP paper](https://arxiv.org/abs/2104.02180).

**LLM-generated rewards are now a serious design tool, not an oracle.** Eureka uses LLM-generated reward code, evolutionary improvement, and GPU-accelerated RL evaluation; its project page reports improvements over human-engineered rewards across many open RL environments. Use this as a reward-proposal and ablation tool, not as unsupervised production truth. Source: [Eureka project](https://eureka-research.github.io/).

**For sim-to-real, penalize unphysical behavior.** Penalize high action rates, high torque/thrust, ground slamming, foot slip, lateral foot motion during stance, and actuator saturation. These terms often matter more for transfer than for simulated score.

---

## 6. Optimization Algorithms and Policy Architectures

### PPO as the Baseline

PPO remains the default baseline for legged locomotion and many drone RL tasks because it works well with vectorized environments, continuous actions, and large-scale on-policy simulation. CleanRL describes PPO as popular, reasonably fast with vector environments, and broadly compatible with action spaces. Source: [CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/).

RSL-RL is the most robotics-specific PPO-oriented library in this survey, with student-teacher distillation and high-throughput training. Source: [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl).

### Algorithm Selection

| Need | Recommended starting point | Notes |
| --- | --- | --- |
| Legged locomotion baseline | PPO in RSL-RL or RL-Games | Common in Isaac Lab/legged_gym lineage |
| Broad algorithm comparison | SB3 or skrl | SB3 is easy and well documented; skrl supports broader Isaac-style integrations |
| Minimal reproducible algorithms | CleanRL | Good for understanding implementation details |
| Sample efficiency with expensive simulation | SAC/TD3/model-based methods | Worth testing when rollouts are not cheap |
| Multi-agent drone swarms | MAPPO/IPPO through skrl/RLlib-style interfaces | Keep centralized training vs decentralized execution explicit |
| Humanoid natural motion | PPO plus motion imitation, AMP, or distillation | Pure task reward often looks unnatural |
| Vision-based policies | PPO/SAC with CNN/RNN, or imitation distillation from state teacher | Avoid training from pixels too early unless sim throughput is high |

### Policy Architectures

**Asymmetric actor-critic:** critic sees privileged terrain/contact/state during training; actor sees deployable observations. This is a strong pattern for legged robots and should be considered the default for sim-to-real.

**History encoders and adaptation modules:** RMA uses a base policy plus adaptation module for online adaptation to terrain, payload, and wear. Source: [RMA project page](https://ashish-kmr.github.io/rma-legged-robots/).

**Teacher-student distillation:** train a state-rich teacher, distill to a sensor-limited student. This is useful for visual drone navigation and legged locomotion over rough terrain.

**Motion priors:** use mocap, retargeted trajectories, or adversarial style rewards to avoid unnatural humanoid behavior.

**VLA/foundation models:** use models such as GR00T for language-conditioned manipulation or high-level skill priors, not as a direct substitute for low-level balance, thrust, and contact control. Source: [NVIDIA Isaac GR00T GitHub](https://github.com/Nvidia/Isaac-GR00T).

### RL Experiment Hygiene

Stable-Baselines3 emphasizes multiple runs because RL results vary with seeds, tuned hyperparameters, and input normalization for custom problems. Source: [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

Minimum experiment standard:

- At least 3 seeds for any claimed improvement.
- Fixed deterministic evaluation protocol.
- Separate train/eval environments.
- Saved config, git revision, simulator version, asset version, randomization ranges, policy checkpoint, and videos.
- Reward component time series.
- Termination reason histogram.
- Out-of-distribution evaluation set.

---

## 7. Sim-to-Real Best Practices and Lessons

### Core Lessons

1. **Do not trust a single simulator.** Humanoid-Gym explicitly includes Isaac Gym to MuJoCo sim-to-sim validation to test robustness and generalization. Source: [Humanoid-Gym project page](https://sites.google.com/view/humanoid-gym/).
2. **Randomize physics and sensors.** legged_gym includes actuator networks, friction/mass randomization, noisy observations, and random pushes. Source: [legged_gym GitHub](https://github.com/leggedrobotics/legged_gym).
3. **Model actuator limits.** Deployment failures often come from latency, motor saturation, gear dynamics, and controller frequency, not just policy quality.
4. **Restrict deployed observations.** If the real robot cannot sense terrain normals, perfect gate pose, wind vectors, or exact contact forces, the actor should not depend on them.
5. **Train with disturbances.** Pushes, wind, contact perturbations, lighting changes, and sensor dropout should be part of the normal training distribution.
6. **Use sim-to-sim before sim-to-real.** Move Isaac-trained policies into MuJoCo, Genesis, Gazebo, or another stack to expose assumptions.
7. **Keep a safety controller.** Real robots need action clipping, rate limiting, emergency stop, fall detection, and conservative state estimation.

### Deployment Checklist

- Policy input/output schema frozen and documented.
- Observation normalization exported with the model.
- Action scaling and joint/thrust command semantics verified.
- Control frequency and latency measured.
- Domain randomization ranges match measured hardware uncertainty.
- Simulator contact/drive parameters audited after asset import.
- Cross-simulator smoke test passed.
- Hardware-in-the-loop or SITL test passed when available.
- Recovery/fall/crash behavior tested before performance optimization.

### Drone-Specific Lessons

Champion-level drone racing shows a hybrid pattern: deep RL in simulation plus real-world data to model sensing/dynamics discrepancies, with onboard sensors and compute. Swift separates perception from control: visual-inertial inputs are converted to a compact representation, and the control policy maps that representation to thrust/body-rate commands. Source: [Nature Swift paper](https://www.nature.com/articles/s41586-023-06419-4).

For drones, validate at three layers:

- Fast RL simulator for policy search.
- Visual/sensor simulator for perception stress.
- PX4/Gazebo or real SITL/HITL for flight-stack compatibility.

---

## 8. State-of-the-Art Papers and Strategies

| Work | Area | Main lesson | Source |
| --- | --- | --- | --- |
| Learning Quadrupedal Locomotion over Challenging Terrain | Quadrupeds | Proprioceptive RL policies can zero-shot transfer to difficult natural terrain when trained robustly | [Project](https://leggedrobotics.github.io/rl-blindloco/) |
| RMA | Quadrupeds | Online adaptation module lets a policy adapt to terrain, payload, and wear from recent history | [Project](https://ashish-kmr.github.io/rma-legged-robots/) |
| legged_gym | Quadrupeds | Rough-terrain sim-to-real needs actuator models, randomization, noisy observations, and pushes | [GitHub](https://github.com/leggedrobotics/legged_gym) |
| Humanoid-Gym | Humanoids | Humanoid sim-to-real benefits from Isaac training plus MuJoCo sim-to-sim validation | [Project](https://sites.google.com/view/humanoid-gym/) |
| Berkeley Humanoid | Humanoids | Hardware designed for learning can narrow the sim-to-real gap enough for simple RL plus light randomization | [Project](https://berkeley-humanoid.com/) |
| ASAP | Humanoids | Residual/delta action models trained from real data can align simulation and real dynamics for agile skills | [arXiv](https://arxiv.org/abs/2502.01143) |
| HOVER | Humanoids | Multi-mode policy distillation can unify different humanoid control modes | [Project](https://hover-versatile-humanoid.github.io/) |
| AMP | Humanoid/character control | Adversarial motion priors replace brittle hand-designed imitation objectives | [arXiv](https://arxiv.org/abs/2104.02180) |
| Swift drone racing | Drones | World-champion-level racing used sim RL, real data, onboard sensing, and compact perception-control interfaces | [Nature](https://www.nature.com/articles/s41586-023-06419-4) |
| Aerial Gym | Drones | Massive multirotor parallelism plus custom sensors enables fast aerial RL | [GitHub](https://github.com/ntnu-arl/aerial_gym_simulator) |
| OmniDrones | Drones | Isaac Sim-based benchmark suite for multi-rotor RL tasks, sensors, control modes, and baselines | [arXiv](https://arxiv.org/abs/2309.12825) |
| Eureka | Reward design | LLMs can generate reward code that is optimized through RL evaluation and reflection | [Project](https://eureka-research.github.io/) |
| GR00T N1/N1.7 | Humanoid foundation models | VLA/diffusion action models are useful for manipulation and high-level skills, especially with post-training | [GitHub](https://github.com/Nvidia/Isaac-GR00T) |

---

## 9. Reusable 3D Assets

### Robot and Drone Assets

| Asset source | Contents | License notes | Best use |
| --- | --- | --- | --- |
| MuJoCo Menagerie | Curated MJCF robots including Unitree H1/G1, Berkeley Humanoid, Cassie, Crazyflie, Skydio X2, arms, hands | Per-model licenses vary; repo requires checking each model directory | High-quality robot models for MuJoCo/MJX/MuJoCo Playground |
| Unitree RL Lab | Unitree Go2, H1, G1-29dof environments on Isaac Lab | Check repo license and asset subdirectories | Unitree-specific Isaac Lab training |
| Unitree RL Gym | Go2, H1, H1_2, G1 RL workflow for Isaac Gym/MuJoCo/physical | Check repo license and model terms | Older Unitree RL reference |
| PX4 Gazebo Models | Drone models and worlds for PX4/Gazebo | Check model/world licenses | PX4 SITL and flight-stack validation |
| gym-pybullet-drones | Crazyflie-style drone assets and PyBullet environments | MIT license on repo | Lightweight RL examples |

Sources: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab), [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym), [PX4 Gazebo Models](https://github.com/PX4/PX4-gazebo-models), [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones).

### Worlds, Props, and Object Assets

| Asset source | Contents | License notes | Best use |
| --- | --- | --- | --- |
| Gazebo Fuel | Hundreds of SDF models insertable into Gazebo worlds | License varies by model | Robotics worlds, obstacles, props |
| Google Scanned Objects | 3D-scanned household objects usable with SDF, Gazebo, and PyBullet | Google says open-source / Creative Commons in paper context; verify item packaging | Grasping, obstacle clutter, household props |
| ReplicaCAD | Indoor apartments and objects for Habitat-sim | CC BY 4.0 | Indoor embodied AI, navigation, rearrangement |
| AI2-THOR / RoboTHOR | Unity-based rooms, objects, humanoid/drone/multi-agent support | Check project and dataset terms before extracting assets | Embodied AI tasks more than raw asset reuse |
| Objaverse / Objaverse-XL | Large-scale Creative Commons 3D object corpus | Per-object Creative Commons licenses vary | Domain randomization and visual clutter |

Sources: [Gazebo Fuel docs](https://gazebosim.org/docs/latest/fuel_insert/), [Google Scanned Objects blog](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/), [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/), [AI2-THOR](https://ai2thor.allenai.org/), [Objaverse-XL GitHub](https://github.com/allenai/objaverse-xl).

### Asset Reuse Rules

- Treat every robot model as a physics asset, not just a mesh. Verify mass, inertia, joint limits, drive modes, damping, collision geometry, and contact parameters.
- Prefer MJCF/URDF/USD/SDF assets with explicit articulation metadata over raw visual meshes.
- For Bevy, consider importing visual meshes for rendered sensors and maintaining a separate simplified dynamics model unless physics fidelity has been validated.
- Keep license metadata beside every imported asset.
- Version imported assets; small inertia or joint-limit changes can invalidate training results.

---

## 10. Bevy-gym Implications and Roadmap

### Local Context

This repo's README describes Bevy-gym as a Bevy ECS plugin for parallelized RL environment simulation. It bridges `rl-traits` environments into Bevy ECS, steps N parallel environment entities in `FixedUpdate`, supports headless training, and emits action/experience/episode events. Source: [local README](../../../README.md) and [BevyGymPlugin docs](../../../docs/plugins/bevy_gym_plugin.md).

That is a good fit for:

- Parallel custom environments.
- Procedural curricula.
- Game-like sensor fairness: rendered pixels, audio-derived observations, simple proprioception.
- Fast iteration around reward functions, environment events, and policy integration.
- Benchmarking custom RL interface quality in Rust.

It is not yet a complete fit for:

- High-fidelity robot contact and actuator dynamics.
- Validated drone aerodynamics.
- Real sim-to-real deployment without external simulator validation.
- Large robot asset import and physics calibration without additional tooling.

### Recommended Architecture for Custom Environments

Use Bevy-gym as the environment harness:

- `ScenarioConfig`: terrain, props, lighting, wind, obstacles, curriculum level.
- `RobotConfig`: morphology, action scaling, sensor set, randomization ranges.
- `SensorBundle`: rendered camera, depth/raycast, IMU-like, proprioception, audio waveform features.
- `RewardTerms`: named reward components with per-term logging.
- `TerminationTerms`: fall, collision, out-of-bounds, timeout, unsafe attitude.
- `DomainRandomization`: physics, visual, sensor, latency, mass, friction, actuator constants.
- `PolicyIO`: normalized observation schema, action schema, and export metadata.
- `EvaluationSuite`: fixed seeds, OOD levels, videos, failure reports.

Keep three observation tiers:

1. **Actor observation:** only realistic deployable sensors.
2. **Critic/teacher observation:** privileged information allowed only during training.
3. **Debug observation:** all state for logging and diagnostics, never consumed by the policy.

### Suggested First Three Bevy-gym Spikes

1. **Flying drone hover/tracking toy task.**  
   Use simplified rigid-body dynamics first. Reward height, position, attitude, and action smoothness. Add wind/noise curriculum. Stop when PPO learns hover and simple waypoint tracking across seeds.

2. **Quadruped or biped procedural balance toy task.**  
   Do not attempt full sim-to-real locomotion first. Start with a simple articulated or reduced-order body, prove reward decomposition, reset logic, vectorized stepping, and telemetry.

3. **Fair sensor navigation environment.**  
   Build a corridor/gate/clutter task where the actor receives rendered camera/depth/raycast and audio-like channels instead of direct target vectors. This aligns with the prior project preference for fair sensor-based AI and is where Bevy-gym is most differentiated from robotics-only stacks.

### Integration Strategy

- Use Isaac Lab or MuJoCo Playground as reference outputs for real robot dynamics tasks.
- Import MuJoCo Menagerie or Unitree models for visual inspection and eventually physics experiments, but do not assume raw import quality.
- Export policies through ONNX/TorchScript or a simple IPC boundary rather than binding Bevy-gym to one trainer too early.
- Add Gymnasium-compatible wrappers if Python trainers remain the fastest path for experiment velocity.
- Keep Rust-native trainer support for headless, reproducible examples once the environment API stabilizes.

### Stop Conditions

Stop investing in Bevy-native robot dynamics if:

- Contact-rich locomotion cannot reproduce basic simulator traces from MuJoCo/Isaac.
- Asset import requires hand-fixing every joint/inertia.
- Training success depends on privileged observations that the intended agent should not have.
- Rollout throughput is worse than Isaac/MuJoCo while physics fidelity is also worse.

Continue investing if:

- The task is game-like embodied AI rather than hardware sim-to-real.
- Sensor fairness, procedural worlds, and visual/audio perception are the differentiators.
- The environment needs Bevy ECS/game systems more than robotics-grade contact fidelity.

---

## 11. Source Verification

### Primary Sources Used

- [Isaac Lab RL framework comparison](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html)
- [Isaac Lab migration from IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [legged_gym GitHub](https://github.com/leggedrobotics/legged_gym)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [Humanoid-Gym project page](https://sites.google.com/view/humanoid-gym/)
- [Unitree RL Lab GitHub](https://github.com/unitreerobotics/unitree_rl_lab)
- [Unitree RL Gym GitHub](https://github.com/unitreerobotics/unitree_rl_gym)
- [MuJoCo Playground GitHub](https://github.com/google-deepmind/mujoco_playground)
- [Genesis World GitHub](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Aerial Gym Simulator GitHub](https://github.com/ntnu-arl/aerial_gym_simulator)
- [Flightmare GitHub](https://github.com/uzh-rpg/flightmare)
- [gym-pybullet-drones GitHub](https://github.com/learnsyslab/gym-pybullet-drones)
- [OmniDrones arXiv](https://arxiv.org/abs/2309.12825)
- [PX4 Gazebo Models GitHub](https://github.com/PX4/PX4-gazebo-models)
- [RMA project page](https://ashish-kmr.github.io/rma-legged-robots/)
- [Learning Quadrupedal Locomotion over Challenging Terrain](https://leggedrobotics.github.io/rl-blindloco/)
- [Champion-level drone racing using deep RL](https://www.nature.com/articles/s41586-023-06419-4)
- [ASAP arXiv](https://arxiv.org/abs/2502.01143)
- [HOVER project page](https://hover-versatile-humanoid.github.io/)
- [AMP arXiv](https://arxiv.org/abs/2104.02180)
- [Eureka project page](https://eureka-research.github.io/)
- [NVIDIA Isaac GR00T GitHub](https://github.com/Nvidia/Isaac-GR00T)
- [MuJoCo Menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- [Gazebo Fuel docs](https://gazebosim.org/docs/latest/fuel_insert/)
- [Google Scanned Objects blog](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)
- [ReplicaCAD dataset](https://aihabitat.org/datasets/replica_cad/)
- [AI2-THOR](https://ai2thor.allenai.org/)
- [Objaverse-XL GitHub](https://github.com/allenai/objaverse-xl)
- [Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/)

### Confidence Notes

- **High confidence:** Isaac Gym deprecation path, Isaac Lab RL library support, legged_gym reference value, RSL-RL role, MuJoCo Playground availability, main drone simulator capabilities, and asset repository existence.
- **Medium confidence:** exact current best "state of the art" ordering, because recent humanoid/drone projects change quickly and open-source quality varies by release.
- **Medium confidence:** Genesis suitability for a production training pipeline; it is promising and current, but should be validated with a focused spike.
- **High confidence for Bevy-gym local implications:** based on local README and plugin docs, but physics capability depends on future integration choices.

### Next Research Spikes

1. Reproduce one Isaac Lab/RSL-RL locomotion example and inspect its reward config.
2. Reproduce one MuJoCo Playground quadruped or humanoid example and compare training throughput.
3. Run gym-pybullet-drones SB3 PPO hover and inspect action/observation schemas.
4. Import one MuJoCo Menagerie robot and one drone model into the intended custom environment asset pipeline.
5. Build a Bevy-gym sensor-fair navigation toy task and compare state-based vs pixel/raycast/audio-like observations.

---

## 12. Implementation Addendum: First Bevy-gym Spike

### Selected Spike

**Build a Bevy-gym sensor-fair drone navigation toy task.**

This is the best continuation from the research because it tests Bevy-gym's differentiating strengths without pretending Bevy is already a robotics-grade dynamics simulator. The spike should prove parallel environments, curriculum progression, reward decomposition, and realistic sensor boundaries before attempting high-fidelity humanoid, quadruped, or drone dynamics.

### Why This Spike Comes First

- It aligns with the research conclusion that Bevy-gym is strongest as a custom environment harness, procedural curriculum surface, and fair sensor sandbox.
- It exercises the user's stated interest in drone locomotion, AI training curricula, and realistic sensors such as rendered pixels and audio waveform-style signals.
- It avoids a premature dependency on Isaac Lab, MuJoCo, or PX4 installation while still keeping those tools as validation references.
- It produces a reusable pattern for later humanoid and quadruped examples: state baseline first, sensor-limited actor next, privileged critic/debug state separate.

### Task Definition

**Environment:** a short 3D corridor or open test volume with gates, obstacles, and goal beacons.

**Agent:** simplified drone body with position, velocity, orientation, angular velocity, and thrust-like action controls.

**Goal:** navigate from start to goal through ordered gates while avoiding collisions and minimizing excessive control.

**Initial action space:**

- `thrust_delta`
- `roll_rate_cmd`
- `pitch_rate_cmd`
- `yaw_rate_cmd`

This mirrors common drone-control abstractions without requiring a full autopilot stack on day one. A later version can add motor RPM commands or PX4-style setpoints.

### Observation Tiers

Keep three explicit observation surfaces:

| Tier | Contents | Allowed consumer |
| --- | --- | --- |
| Actor observation | rendered/raycast/depth-like forward sensor, IMU-like state, last action, audio-like event channels | deployable policy |
| Critic/teacher observation | exact pose, gate vector, obstacle distances, wind, hidden curriculum parameters | critic or teacher only |
| Debug observation | full ECS state, reward terms, collision normals, reset reason | logs and tests only |

The first implementation can start with a low-dimensional state actor to prove reward and reset semantics, but the spike should not be considered complete until a sensor-limited actor path exists.

### Curriculum Phases

1. **Hover and attitude stabilization**
   - Empty volume.
   - Reward stable height, low attitude error, low velocity, smooth actions.
   - Pass condition: multi-seed policy maintains hover for full episode.

2. **Single waypoint tracking**
   - Static goal in front of the agent.
   - Reward progress and final distance.
   - Pass condition: reaches goal without collision across randomized starts.

3. **Single gate traversal**
   - One gate, no clutter.
   - Reward gate-plane crossing in correct direction and centerline alignment.
   - Pass condition: crosses gate reliably without using direct gate vector in actor observation.

4. **Short gate chain**
   - Three to five gates with mild turns.
   - Reward ordered gate progress and time efficiency.
   - Pass condition: completes full chain across held-out layouts.

5. **Sensor degradation and clutter**
   - Add visual noise, raycast dropout, simple wind, obstacles, and distracting props.
   - Pass condition: policy remains above threshold on randomized evaluation layouts.

### Reward Terms

Use named reward components and log them every episode:

| Term | Purpose |
| --- | --- |
| `progress_to_next_gate` | dense forward progress |
| `gate_crossed` | sparse milestone reward |
| `goal_completed` | terminal success reward |
| `centerline_alignment` | discourages clipping gate edges |
| `attitude_stability` | prevents tumbling |
| `control_effort_penalty` | discourages saturation |
| `action_rate_penalty` | discourages oscillation |
| `collision_penalty` | enforces obstacle safety |
| `out_of_bounds_penalty` | keeps behavior inside training volume |
| `timeout_penalty` | discourages hovering in place |

Do not hide success inside one scalar. Every run should be inspectable by reward component, termination reason, and video or trace.

### Validation Gates

Minimum completion gates:

- `cargo check` passes.
- Headless rollout smoke test runs multiple parallel environments.
- Deterministic evaluation runs at least 3 seeds.
- Reward component summary is written for every evaluation run.
- Termination reasons are reported.
- A state-based baseline and a sensor-limited policy are compared.
- Held-out gate layouts are evaluated separately from training layouts.
- The final report names whether dynamics are Bevy-only, cross-checked in gym-pybullet-drones, or cross-checked in another robotics simulator.

### External Reference Gates

Use these only when the local toy task is stable:

- **gym-pybullet-drones:** compare hover and simple waypoint action/observation semantics.
- **Aerial Gym or OmniDrones:** compare multirotor curriculum structure and sensor assumptions.
- **PX4/Gazebo:** validate flight-stack integration only if the project moves toward real autopilot behavior.

### Initial Bevy-gym Artifact Shape

Target files for a later implementation pass:

```text
examples/robotics/drone_gate_navigation/
  README.md
  drone_gate_navigation.rs
  assets/
  configs/
    hover.toml
    single_gate.toml
    gate_chain.toml
  reports/
```

The example should be small enough to understand in one sitting. Prefer a simple test volume, procedural gate placement, and explicit reward logs over a visually rich scene at the start.

### Stop Rules

Stop and reassess if:

- The policy only learns with direct privileged target vectors.
- Reward improves while visible behavior is collision-prone or unstable.
- Time-scale changes alter control dynamics enough to change pass/fail behavior.
- Bevy physics details dominate the project before the curriculum and sensor contract are proven.

Continue if:

- The same environment can run state-based and sensor-limited actor modes.
- The actor observation remains realistic and does not leak exact target state.
- Reward components explain failures without needing manual video inspection every time.
- A simple held-out layout pass is reproducible across seeds.
