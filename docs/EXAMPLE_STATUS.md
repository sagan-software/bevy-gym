# Examples execution status

Updated: 2026-10-08. Goal status: active.

## Resume here

Read [the roadmap](EXAMPLE_ROADMAP.md), [initial audit](QUALITY_AUDIT.md), and
[reference research](EXAMPLE_RESEARCH.md). Continue P0 reference collection, then
P1 drone hover. Do not start later port families before the custom robot milestones.

Working checkout: `/home/sagan/Code/github.com/sagan-software/bevy-gym-quality`.
Branch: `quality-roadmap-20261008`. Baseline: `232e801`.
Original checkout: `/home/sagan/Code/github.com/sagan-software/bevy-gym`.
It has 1,304 staged deletions, four additions, and three untracked example folders.
Preserve its index and working files. Do not use `git add -A` there.

GitHub remote is `github`; `origin` is a separate GitLab repository. Publish
authorized milestones to GitHub `main`. Configured identity is Bill Curry,
`bill@sagan.software`. Inspect author, committer, message, and trailers before
finalizing each commit. Never add AI attribution.

## Completed in this run

- Created a persistent goal covering the complete request.
- Created an isolated worktree and inspected tracked reviewer instructions.
  No matching reviewer-instruction file was found outside the `ai` instructions.
  Root `AGENTS.md` is an ignored broken Nix-store symlink; use session instructions.
- Inventoried 23 existing Gymnasium reference sheets and their manifest.
- Located Unity source scenes, official image inventory, and source revision.
- Located all eleven Godot README videos, source folders, and per-example notices.
- Verified AI Warehouse's public README says example code is private.
- Identified existing `yt-dlp` version 2026.08.19 and FFmpeg.
- Confirmed `yt-dlp` is already committed in NixOS dotfiles at
  `modules/dev/default.nix:156`, with the development module enabled on T490.
  The user meant NixOS dotfiles. Removed the proposed project-shell additions;
  no dotfiles edit or activation is needed.
- Wrote the complete roadmap and initial audit with explicit verification limits.

## In progress

- Official ARC Raiders documentary downloaded; aerial, damage, locomotion, and
  training sheets inspected. Dense drone and Leaper sheets are also inspected.
  A continuous jump clip is still needed for measured motion timing.
- All 17 Unity images and eleven Godot videos downloaded. Godot 20-frame
  sheets generated and individually inspected. All 17 Unity images are inspected.
- Select a licensed four-thruster drone model after visual and geometry review.
- Establish baseline Rust/personal lint outcomes and changed-code coverage tools.
- Removed dependency-only imports from the five Classic Control examples.
  Formatting and the direct import check pass. Compilation verification remains
  incomplete; cleanup is uncommitted. Drone regression follows reference review.
- Dedicated Leaper gameplay downloaded and inspected at overview and dense
  24-frame intervals. Native Computer Use also reports no available browser.

## Verification

- `nix develop --command cargo fmt --all -- --check`: passed on baseline.
- Full test and strict Clippy sessions returned exit 143 during initial
  validation. Tests never reported a result. Clippy logged a finished build,
  but its process outcome is not a pass. Both exact gates require a rerun.
- Free disk fell below 200 MB. Permission to remove older unused Cargo build
  artifacts is pending. Preserve existing caches until the user responds.
- The full test rerun uses the cached Nix shell environment with both
  `CARGO_TARGET_DIR` and `CARGO_BUILD_BUILD_DIR` under `/dev/shm`.
  Debug information is disabled and build jobs are limited to two. This avoids
  filling the disk or deleting existing caches. Compilation remains in progress.
  The cached environment is
  `/nix/store/lgpxrzzh30dabqqay3mdz83waq388xy8-nix-shell-env.drv`.
  Logs are under `/dev/shm/bevy-gym-quality-logs/` and are volatile across reboots.
- `cargo llvm-cov` 0.9.0 is available. Coverage has not run.
- Personal lints and Nix checks remain pending.
- Roadmap, research, status, and audit Markdown: repository `rumdl` check passed.
- Personal-lint ecosystem inventory completed; this is not a passed lint gate.
- Source media hashes verified for all 28 Unity/Godot files and both ARC videos.
- The latest five-document Markdown check passes. `HANDOFF.md` still reports ten
  `MD060` table-alignment diagnostics; the same ten diagnostics occur in its
  committed baseline. The added handoff notice does not change those tables.
- T3 `preview_status` and `preview_open`: unavailable; no desktop automation host
  is connected to this environment. No live visual acceptance is claimed.
- Historical browser qualification exists in repository documents. It has not
  been rerun or promoted to current verification by this audit.
- No robot implementation, newly qualified policy, or new deployment is complete.
- Planning checkpoint `2843952` pushed to GitHub `main`. Its CI run is
  <https://github.com/sagan-software/bevy-gym/actions/runs/37809258024>.
  CI completed successfully. No new deployment is claimed.

Research downloads and command logs are under ignored `runs/quality-research/`.
Keep durable source URLs and conclusions in committed documents. Do not depend on
ignored files as the only record of a completed acceptance check.
The documentary path now links to
`/dev/shm/bevy-gym-quality-references/arc-documentary.mp4` to recover disk space.
Its hash was verified before moving this run's download. RAM files disappear on
reboot; use the committed media inventory to download them again.

## Pending audit areas

Core/reset lifecycle; public configuration validation; multi-plugin isolation;
asynchronous response correlation; all environment transition implementations;
trainer numerical correctness; recurrence and multi-agent accounting; checkpoint
validation; worker cancellation and stale results; rendering fidelity; asset
integrity; all tutorial paths; full CI and deployed gallery.

## Handoff update rule

After each checkpoint, record its commit, pushed revision, CI/deployment outcome,
tests and exact coverage gaps, visual artifacts, learning evidence, blockers, and
one concrete next action. Keep the goal active until the entire roadmap is done.
