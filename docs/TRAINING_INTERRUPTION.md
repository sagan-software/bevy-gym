# Training outcomes and storage restart

The seed-29 drone rehearsal exhausted its near-travel budget.
The seed-23 standing trial stopped when the filesystem filled.
Standing restarted from its original parent with fresh Adam.
Both lessons remain unqualified.

## Drone rehearsal

The [trial record](progress/travel-rehearsed-seed29-final/summary.json) retains
source `4300eaa`, its immutable executable hash and the launch record.
Training completed 6,860 endurance updates and 24,000 near updates,
using 15,800,320 transitions. Near selection failed and the process exited 1.
Far and fast travel were not promoted.

All five final near selection episodes survived 1,000 actions.
Seed 0 first arrived at action 767, after the action-500 deadline.
Its final settled streak was 234 actions. The other four cases never arrived,
had zero settled actions and ended with heading errors above 1.29 radians.

The frozen near checkpoint also failed the separate retention evaluator.
Twenty-eight of 32 endurance cases and all 32 near cases failed physical gates.
Every case survived the horizon. The endurance mean-return gate passed;
the near mean-return gate failed. These held-out roots did not enter training
or checkpoint selection.

Replay the frozen report:

```sh
nix develop --command cargo run --features robots --example drone-curriculum -- \
  --lesson travel \
  --evaluate-promotion docs/progress/travel-rehearsed-seed29-final/checkpoint.mpk \
  --travel-stage travel-near
```

Exit 1 is the recorded qualification failure. The archive preserves the
checkpoint, all 64 retention cases, all 1,543 selection reports and all
30,860 optimizer records. Hashes identify the original and compressed journals.

## Standing interruption and restart

The [interrupted trial](progress/standing-posture-storage-restart/summary.json)
used source `2343336`. Its last complete selection was update 5,720,
with zero stable actions in all five cases. The optimizer journal contains
5,728 complete records and a 176-byte unfinished final record.
The process ended at `Sat 2026-10-10 10:55:35 MDT` with
`StorageFull: No space left on device`. It did not exhaust its training budget.

The compressed interrupted journal preserves its original bytes, including
the unfinished final record. The archive also retains the update-5,720
checkpoint, provenance, selection, reward profile and parent transfer record.

Checkpoints store actor and critic weights. Adam, samplers and episode memory
are not saved. The new run imports the same original update-22,940 parent
and initializes fresh Adam and episode state. It uses the same seed 23,
posture reward, executable and 24,000-update limit.
It does not resume the interrupted optimizer.

The [restart launch](progress/standing-posture-storage-restart/restart-launch.json)
records its command and unit `bevy-gym-standing-posture-seed23-retry-20261010.service`.
At the archived observation, the restart was active and its first 5,728 complete
optimizer records matched the entire completed interrupted prefix.
Original selection gates, the 32 held-out cases and independent repeat evidence
remain required before qualification.
