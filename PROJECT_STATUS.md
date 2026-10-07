# Project status and next steps

Assessment date: 2026-10-07.

Finish one Gymnasium environment at a time, including browser playback and
browser training and measured policy quality, before expanding the catalog.
CartPole is deployed. MountainCar passed offline browser qualification; its
deployment and visual review remain pending.
Keep ecosystem and PettingZoo expansion behind the Gymnasium milestones.

## Gymnasium port policy

The Gymnasium catalog must port the original environment logic to Rust and
retain its original simulation model. Avian is not a dependency requirement
for these ports.

- Classic Control and Toy Text use direct Rust translations of the upstream
  equations, transition rules, and task logic. They need no replacement physics
  engine.
- LunarLander, BipedalWalker, and CarRacing retain Box2D physics, as specified by
  the [official Box2D family](https://gymnasium.farama.org/environments/box2d/).
  Port the Python environment logic to Rust around the compatible engine.
- MuJoCo tasks retain MuJoCo and the original MJCF models, as specified by the
  [official MuJoCo family](https://gymnasium.farama.org/environments/mujoco/).
  Port the environment logic to Rust around that engine. The three existing
  native MuJoCo examples are starting points, pending conformance verification.
- Bevy can render and schedule these environments without controlling their
  physics. Native evaluation and browser execution must use the same task
  semantics. Browser delivery must not substitute simplified dynamics.
- Separate Avian examples may demonstrate custom physics and ecosystem tasks.
  Label them separately and exclude their scores from Gymnasium qualification.

Rust environment code may call the original C/C++ physics engine through
bindings. This policy does not require rewriting Box2D or MuJoCo in Rust.
Select and pin engine versions against the upstream reference, then prove
native and browser compatibility before choosing a binding. A newer engine
with the same name is not evidence of equivalent dynamics.

Before dependency cleanup, `examples/avian2d` and `examples/avian3d` imported Avian largely
through unused-dependency markers. No Avian physics API use was found in those
directories during this assessment. Inspected LunarLander and InvertedPendulum
paths integrate handwritten dynamics. Removing imports or renaming directories
does not turn those implementations into faithful ports.

Replace those approximations one environment at a time after oracle tests
establish the replacement's behavior. Keep existing work until its replacement
is verified.

Preserve any useful approximation under an explicit experimental
identity; do not describe it as an Avian simulation without actual Avian use.
The approximations now live under `examples/box2d-approximation` and
`examples/mujoco-approximation`. Shared Avian development dependencies and the
render feature dependency were removed.

Ecosystem examples require explicit
`--features ecosystem-inference`. Conditional unused-dependency imports remain
only to support builds that explicitly combine Gymnasium and ecosystem features.
Reevaluate or retrain its policies because old scores and checkpoints do not
establish performance under the restored physics.

## Repository state

The public repository is <https://github.com/sagan-software/bevy-gym>.
The `github` remote points there; `origin` still points to the existing GitLab
repository. GitHub `master` contains baseline commit
`915db888df025ec76c78bb24edc0d663ea6ccc3b`.

Substantial staged, unstaged, and untracked work remains in this checkout.
That includes the browser implementation and many completed-looking examples.
The GitHub baseline does not contain those changes. Preserve the index and
working tree while reviewing and committing this work in coherent groups.
Do not include `runs/`, `target/`, `dist/`, or `test-results/` as source changes.

The CI workflow now targets `main`, runs tests and strict Clippy, and does not
publish crates. A separate browser-preview workflow builds and tests the site
before deploying GitHub Pages. Qualification remains incomplete until all
per-environment gates pass.
The local `AGENTS.md` symlink is broken on this host; the session-supplied
instructions governed this assessment.

## What exists locally

- Five Classic Control examples have native train, evaluation, and visual
  workflows. Their README records strong results for CartPole, both Mountain
  Car tasks, and Acrobot. Pendulum's recorded mean of `-653.65` misses the
  fixture catalog's project gate of `-200`. These historical scores were not
  reproduced during this assessment.
- Four Toy Text examples have native workflows and historical evaluation
  results. Registry variants and constructor options need explicit acceptance
  coverage as well as the default tasks.
- Three Box2D task approximations and eleven MuJoCo task approximations exist.
  Their own documentation says
  they are approximations. Their scores cannot establish success on the
  original Box2D or MuJoCo environments.
- Three native MuJoCo examples use original Gymnasium models. Their native
  runtime dependency has no demonstrated browser integration here.
- `web/` implements six ecosystem routes with checkpoint loading, uploads,
  pause, restart, and playback speed. It has no Gymnasium route catalog.
  Its bundled policies are labelled `best-compatible-available`, not qualified.
- The ecosystem handoff records unresolved conservation/value-state issues,
  competition retention failures, and unfinished later-stage qualification.
  Keep that work available, but do not make it the next delivery milestone.

The July reference audit predates much of the current implementation. Its
placeholder inventory is historical, not the current completion checklist.

## Delivery order

1. Preserve and review the local implementation, repair CI, and establish a
   reproducible source baseline on GitHub.
2. Finish CartPole-v1 with a shared native/browser environment, a bundled
   qualified policy, conformance tests, and browser acceptance.
3. Finish MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, then Acrobot-v1.
   Complete each environment before starting the next.
4. Finish CliffWalking-v1, FrozenLake-v1, Taxi-v4, then Blackjack-v1. Include
   registered variants and document supported constructor options.
5. Replace the LunarLander, BipedalWalker, and CarRacing approximations with
   Rust environment logic using the original Box2D physics. Prove compatible
   native and browser engine execution before implementing the first port.
6. Finish MuJoCo tasks, starting with InvertedPendulum, InvertedDoublePendulum,
   and Reacher. Then cover Pusher, Swimmer, Hopper, HalfCheetah, Walker2D, Ant,
   HumanoidStandup, and Humanoid. Replace the corresponding approximations with
   original MuJoCo models and Rust task logic. Prove browser engine execution first.
7. Extend to Atari, multi-agent tasks, and the custom ecosystem curriculum only
   after the preceding examples meet their acceptance gates.

The initial browser catalog covers nine default Classic Control/Toy Text tasks.
The four core families provide 23 base tasks before configuration variants.
This count excludes Atari and external environments. Official family catalogs:
[Classic Control](https://gymnasium.farama.org/environments/classic_control/),
[Toy Text](https://gymnasium.farama.org/environments/toy_text/),
[Box2D](https://gymnasium.farama.org/environments/box2d/), and
[MuJoCo](https://gymnasium.farama.org/environments/mujoco/).

## Acceptance for every environment

- Pin the upstream registry ID, source commit, options, spaces, reset rules,
  transition rules, reward, termination, and truncation behavior. Record supported
  variants and deliberate differences separately. Keep the existing oracle pin
  `7a1191388aa4aa973d3a5e4b039899cd99cc991f` until a reviewed upgrade.
- Generate oracle results from pinned Python dependencies and compare the Rust
  implementation against them. Cover boundary states and time limits. Test
  stochastic distributions separately from deterministic transitions; matching
  numeric seeds does not imply matching NumPy random streams.
- Use the same environment implementation in native evaluation and WASM.
  Separate native training, filesystem, and process code from browser inference.
- Qualify policies on held-out evaluation seeds across three training seeds.
  Keep selection seeds separate from final test seeds. Report environment
  rewards, training-only shaping, baselines, failures, and per-seed results.
  Use the catalog's project gates without calling local gates upstream rules.
- Bundle versioned policy artifacts with hashes, configuration, source commit,
  and evaluation evidence. A fresh clone must reproduce evaluation without a
  private `runs/` directory. Label unqualified policies explicitly.
- Provide a browser URL that loads a trained policy automatically. Support
  reset, pause, single-step, speed, and seed selection. Show return, episode
  length, termination reason, and policy identity. Include a random baseline
  for comparison. Every delivered environment must support both browser training
  and browser inference. See [the implementation plan](GYMNASIUM_BROWSER_PLAN.md)
  for learning curves, 1x through 16x pacing, and per-environment gates.
- Verify sustained motion, episode completion, repeated resets, route changes,
  checkpoint failures, and native/WASM action parity. Inspect desktop and mobile
  rendering. Run Chromium, Firefox, and WebKit checks and record unsupported
  browser capabilities explicitly.
- Measure release download size, startup time, and frame time. Publish the
  tested static artifact through GitHub Pages. The user authorized public
  visibility for Sagan-software/bevy-gym.
- Pass the required Rust, personal-lint, Nix, WASM, and browser gates. Measure
  changed-code coverage and explain any remaining uncovered path. Do not count
  a moving canvas, passing build, or recorded video as policy qualification.

## Delivered CartPole slice

[GitHub Pages](https://sagan-software.github.io/bevy-gym/) serves CartPole
training and inference from the public repository. The shared Rust environment
uses the pinned Gymnasium Euler equations and float64 internal state. Reset
uses SplitMix64, so its numeric seeds do not reproduce NumPy PCG64 sequences.

Three native training seeds passed qualification within 90,000 transitions.
The selected bundled policy scored 500 across 200 held-out episodes. A separate
Rust WebAssembly training run reached 500 across 200 held-out episodes after
70,000 transitions, from an initial validation mean of 9.45.

The page supports pause, step, restart, 1× through 16× pacing, return and
learning-rate charts, and policy download/upload. Inference has no optimizer.
Desktop and mobile views were inspected through the collaborative browser.
See [the browser package](gymnasium-web/README.md) for commands and coverage gaps.

## Remaining verification and environments

MountainCar-v0 now has shared Rust dynamics, browser training and inference,
and a bundled policy selected by validation score. All three native training
seeds passed the -110/95% gate. Browser functional checks passed in Chromium;
its full offline learning gate and deployment verification remain pending.
MountainCarContinuous-v0 follows after that slice passes. Other Gymnasium
environments have not passed browser acceptance. Box2D and MuJoCo browser
engine compatibility still needs proof before their faithful ports proceed.

Sustained resource use and the provisional latency budgets remain unmeasured.
Chromium, Firefox, and WebKit passed the browser controls and real optimizer
checks. Chromium also passed full offline learning qualification.
Fixture integrity alone does not prove environment conformance;
Blackjack hidden-hand and Taxi fickle-state enumeration remain unfinished.
