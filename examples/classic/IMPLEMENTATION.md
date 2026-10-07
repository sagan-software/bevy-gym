# PettingZoo classic implementation ledger

This ledger records the source contract, comparison protocol, targets, and reproducible evidence for the nine classic ports.

The pinned PettingZoo source is commit `38e73889c04cedf7b92eb65d74bb6484f69c8c33`. Environment code and GIF references live under `ref/pettingzoo`.

## Comparison protocol

The DQN examples reproduce `tutorials/SB3/connect_four/sb3_connect_four_action_mask.py`:

- Training uses one shared policy for every AEC agent and one transition per AEC action.
- Evaluation uses seeded uniform-random actions for `possible_agents[0]` and the greedy checkpoint for `possible_agents[1]`.
- Every policy action uses the source legal-action mask.
- Alternating zero-sum games negate the opponent-perspective bootstrap value.
- `comparison_win_rate` is the trained agent's share of absolute decisive reward. This matches PettingZoo's reward-weighted score calculation.
- Reported comparison results use seeds `0..99` and the final checkpoint at the source training budget.

Tic-Tac-Toe reproduces `tutorials/Tianshou/2_training_agents.py`: player one is random, player two learns, and the stop target is mean return `0.6`. RPS remains a learned player-zero policy against a seeded random player-one policy because PettingZoo explicitly excludes it from the self-play benchmark.

No classic training or evaluation path uses demonstrations, arranged deals, fixed replies, tactical action replacement, or a separate deployment policy. Fixed card orders and Scholar's Mate appear only in raw rules tests.

## Results

| Environment | Source budget and target | Rust final-checkpoint result | Comparison |
| --- | --- | --- | --- |
| RPS v2 | No official learning target; PettingZoo says self-play does not learn reliably | 500 episodes; mean return `0.090`, positive-return rate `0.440` | Consistent with no exploitable uniform-random opponent strategy |
| Tic-Tac-Toe v3 | Tianshou stop at mean return `0.6` | Stopped after 3,000 episodes; mean return `0.630`, success rate `0.770` | Reached the official stop target |
| Connect Four v3 | 20,480 transitions; about `0.80` win rate, with `0.86` noted near 20,000 transitions | 20,480 transitions; win rate `0.910`, mean return `0.820`, final Huber loss `0.009990` | Above the published reference range |
| Chess v6 | 8,192 transitions; difficult and expected below `0.75` | 8,192 transitions; 9 wins, 10 losses, 81 draws; reward share `0.474`, mean return `-0.010`, final loss `0.000020` | Matches the official difficult-environment expectation |
| No-Limit Hold'em v6 | 32,768 transitions; easy environment expected above `0.50` | 32,768 transitions; reward share `0.549`, success rate `0.660`, mean payoff `3.170`, final loss `12.666074` | Reached the official threshold |
| Limit Hold'em v4 | 8,192 transitions; medium environment expected below `0.75` | 8,192 transitions; reward share `0.707`, success rate `0.750`, mean payoff `1.310`, final loss `0.231744` | Matches the official expected range |
| Leduc Hold'em v4 | 32,768 transitions; easy environment expected above `0.50` | 32,768 transitions; reward share `0.846`, success rate `0.830`, mean payoff `1.260`, final loss `0.254631` | Reached the official threshold |
| Go v5 | 8,192 transitions; hard environment, with `0%` noted in the source test | 8,192 transitions; 100-game final-checkpoint win rate `0.000`, mean return `-1.000`, final loss `0.000813` | Matches the official hard-environment result |
| Hanabi v5 | 8,192 transitions; source test notes `0%` win rate and tied score | 8,192 transitions; comparison win rate `0.000`, mean score delta `0.000`, final loss `0.033525` | Matches the official result |

The DQN loss values are optimizer diagnostics. PettingZoo does not define universal loss targets. Reward and win-rate targets control the comparison.

The 62 focused environment tests cover 80.03% of lines across the nine classic modules. Individual module coverage ranges from 74.41% to 85.64%.

## Environment contracts

### RPS v2

- Source: `pettingzoo/classic/rps/rps.py`.
- Default rules: three actions, two players, 15 simultaneous cycles.
- Rewards: cycle winner `+1`, loser `-1`, draw `0`.
- Tests cover the payoff matrix, observations, history, configuration, truncation, and seeded random opponent.

### Tic-Tac-Toe v3

- Source: `pettingzoo/classic/tictactoe/tictactoe.py` and `board.py`.
- Rewards: winner `+1`, loser `-1`, draw `0`.
- The adapter keeps source board ordering and masks while assigning the official random and learned seats.
- Tests cover all winning lines, draws, perspectives, masks, raw errors, illegal-action rewards, and seeded role assignment.

### Connect Four v3

- Source: `pettingzoo/classic/connect_four/connect_four.py`.
- Rewards: winner `+1`, loser `-1`, draw `0`.
- Tests cover gravity, every win direction, perspectives, masks, raw errors, illegal-action rewards, AEC turn preservation, and random-first evaluation.

### Chess v6

- Source: `pettingzoo/classic/chess/chess.py` and `chess_utils.py`.
- Observation: `8 x 8 x 111`; action space: `8 x 8 x 73`.
- `shakmaty` supplies legal chess transitions. Local history reproduces repetition and 50-move claims.
- Tests cover action mapping, black mirroring, en passant, history planes, illegal moves, checkmate, AEC turns, and evaluation roles.

### No-Limit Hold'em v6

- Source: `pettingzoo/classic/rlcard_envs/texas_holdem_no_limit.py` and `rlcard_base.py`.
- Observation length: 54; action count: 5; starting stack: 100.
- Tests cover blinds, legal actions, all-in restrictions, chip payoff, observations, hand ordering, shuffled deals, AEC turns, and evaluation roles.

### Limit Hold'em v4

- Source: `pettingzoo/classic/rlcard_envs/texas_holdem.py` and `rlcard_base.py`.
- Observation length: 72; action count: 4; rewards use big-blind-scaled net chips.
- Tests cover blinds, raises, folds, observations, showdown ordering, shuffled deals, AEC turns, and evaluation roles.

### Leduc Hold'em v4

- Source: `pettingzoo/classic/rlcard_envs/leduc_holdem.py` and `rlcard_base.py`.
- Observation length: 36; action count: 4; rewards use big-blind-scaled net chips.
- Tests cover blinds, raise caps, the public-card transition, fold and showdown payoffs, observations, shuffled deals, AEC turns, and evaluation roles.

### Go v5

- Source: `pettingzoo/classic/go/go.py`, `go_base.py`, and `coords.py`.
- Default profile: 19 by 19 board, 7.5 komi, 17 observation planes, and 362 actions.
- Tests cover captures, liberties, suicide, ko, history planes, masks, two-pass scoring, AEC turns, and evaluation roles.

### Hanabi v5

- Source: `pettingzoo/classic/hanabi/hanabi.py` and `rendering.py`.
- Default profile: two players, five colors, five ranks, 50 cards, 658 observation values, and 20 actions.
- Tests cover deck multiplicities, action masks, hints, token rules, failed plays, all observation sections, final turns, shuffled deals, AEC turns, and evaluation roles.

## Visual and artifact contract

Each Bevy renderer uses the source GIF dimensions, playback rate, copied images, copied font assets, and source placement formulas. Watch and GIF modes use the same checkpoint and random-first comparison roles as evaluation.

Reproducible comparison runs are stored under `runs/classic-comparison`. Each run contains configuration, seeds, metrics, step checkpoints, `latest.mpk`, `best.mpk`, and a summary.
