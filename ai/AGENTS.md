# AI Workspace

This directory owns repo-local AI tool configuration and skills.

- Canonical skill source lives in `ai/skills`.
- BMad install files live in `ai/bmad`.
- BMad generated artifacts live in `ai/bmad-output`.
- Root discovery paths such as `.agents/skills` and `.claude/skills` are
  generated compatibility symlinks.
- Do not add root `_bmad` or `_bmad-output` compatibility links; update BMad
  config and skill instructions to use `ai/bmad` and `ai/bmad-output`.

## Reinforcement learning integrity

Every autonomous agent shown in a project example must act through a policy trained
with reinforcement learning. This applies to all examples, including curriculum
lessons, demonstrations, opponents, teammates, and final competitive environments.

Prohibit programmed agent decisions: scripted routes, navigation planners, automatic
target selection or aiming, automatic firing, behaviour trees, hand-written alert
transitions, scripted locomotion, and heuristic or constant-action fallback controllers.
Do not disguise these as learned behaviour, including behind a neural-network wrapper.
Imitation-only weights and scripted teachers do not satisfy reinforcement training.
Do not substitute them when RL training fails or takes longer than expected.

Policies must choose movement, aim, trigger requests, and behavioural state. Physical
locomotion examples must use policy-driven actuators and authoritative simulated bodies.
Animation playback or direct transform movement cannot substitute for learned locomotion.
A hierarchy is permitted only when every decision-making controller is RL-trained.

Implement physics, sensors, collisions, actuator bounds, ammunition, cooldowns, damage,
death, reset, rewards, and lesson promotion as explicit environment rules. These rules
may constrain an action but must not choose an agent's action. Keep reward and critic
information unavailable to inference actors unless it is part of their valid observation.
Spectator controls affect presentation only. Manual control must be an explicit user mode.

Use deterministic action fixtures only in isolated tests. They are not example agents,
training substitutes, or qualification evidence. Missing or incompatible checkpoints
and invalid inference must stop execution with a visible error, never activate a fallback.

Before publishing any agent example, trace every action to its policy checkpoint.
Record training method, source and checkpoint hashes, observation/action contracts,
seeds, curriculum stages, sample counts, rewards, and independent evaluation results.
A successful build, neural-network inference, or selected video does not prove RL.
Untrained and failed candidates remain unfinished; do not claim learned competence.

When modifying existing examples, audit inherited controllers against this rule.
Remove nonconforming controllers from published agent demonstrations as they are replaced;
a historical hybrid implementation is not an exception allowing new scripted agents.
Preserve historical evidence, and record remaining violations without claiming compliance.

For drone/droid implementation, training, or review, read
`docs/DRONES_VS_DROIDS.md`, `docs/ROBOT_SKILL_CURRICULUM.md`, and
`docs/EXAMPLE_STATUS.md` before choosing work. Keep curriculum skills independently
runnable and reusable in the final shared 3v3 environment. Promote only after independent
evaluation passes; an exhausted training budget is a failed lesson, not a promotion.
