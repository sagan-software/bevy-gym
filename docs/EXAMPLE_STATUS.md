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
- Wrote the complete roadmap and initial audit with explicit verification limits.

## In progress

- Official ARC Raiders documentary downloaded; aerial, damage, and locomotion
  sheets inspected. Dedicated jumping footage and detailed motion analysis remain.
- All 17 Unity images and eleven Godot videos downloaded. Godot 20-frame
  sheets generated; their visual inspection remains pending.
- Select a licensed four-thruster drone model after visual and geometry review.
- Establish baseline Rust/personal lint outcomes and changed-code coverage tools.
- Removed dependency-only imports from the five Classic Control examples.
  Formatting and compilation verification are in progress; drone regression is next.

## Verification

- `nix develop --command cargo fmt --all -- --check`: passed on baseline.
- Full tests: compiling dependencies. Strict Clippy, personal lints, coverage,
  and Nix checks remain pending.
- Roadmap, research, status, and audit Markdown: repository `rumdl` check passed.
- Personal-lint ecosystem inventory completed; this is not a passed lint gate.
- T3 `preview_status` and `preview_open`: unavailable; no desktop automation host
  is connected to this environment. No live visual acceptance is claimed.
- Historical browser qualification exists in repository documents. It has not
  been rerun or promoted to current verification by this audit.
- No robot implementation, newly qualified policy, or new deployment is complete.
- No checkpoint from this run has been pushed yet.

Research downloads and command logs are under ignored `runs/quality-research/`.
Keep durable source URLs and conclusions in committed documents. Do not depend on
ignored files as the only record of a completed acceptance check.

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
