## Deferred from: code review of 1-1-bg-train-001-required-burn-dependency-and-trainer-skeleton (2026-06-13)

- Restore or formally migrate removed render feature/module compatibility. Blind review flagged removed `render`, `winit`, `x11`, `wayland`, and `render` module surface in the baseline diff; this appears to be pre-existing API-redesign work outside BG-TRAIN-001.
- Restore or formally migrate removed legacy public re-export compatibility. Blind review flagged removed or renamed items such as `ActionRequestEvent`, `ExperienceEvent`, `PendingAction`, and `EnvironmentComponent`; this appears to be pre-existing API-redesign work outside BG-TRAIN-001.
