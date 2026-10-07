# Continual ecosystem curriculum and reward design

Research date: 2026-07-30

## Conclusion

This project needs curriculum transfer with retention, not a sequence of isolated
tasks. Keep one policy interface and one healthy-survival objective. Change the
distribution of worlds and opponents as competence increases. Continue sampling
earlier lessons so later PPO updates cannot erase them.

Reward should define the desired outcome. It should not imitate neurotransmitter
concentrations. Dopamine activity has a strong relationship to reward-prediction
error, which is closer to the temporal-difference learning signal than to a
designer assigning dopamine points for every biological event
([Schultz, Dayan, and Montague, 1997](https://pubmed.ncbi.nlm.nih.gov/9054347/)).
Homeostatic reinforcement learning provides a useful biological abstraction:
outcomes earn value when they reduce deviation from physiological setpoints
([Keramati and Gutkin, 2014](https://elifesciences.org/articles/04811)). It does
not require reproducing brain chemistry.

## Curriculum learning and continual learning

Curriculum learning sequences tasks or data so experience from one task helps
learning in a harder target task. Continual reinforcement learning adds the
requirement that an agent adapts to a changing task stream while retaining useful
past behavior. Sequential fine-tuning alone does not provide that guarantee
([Narvekar et al., 2020](https://jmlr.org/papers/v21/20-212.html),
[Khetarpal et al., 2022](https://arxiv.org/abs/2012.13490)).

For this project:

- Long-term memory is stored in policy parameters. The LSTM state is only
  episode-scoped working memory.
- A checkpoint transfers learned parameters. It does not transfer an abstract
  named skill.
- The actor transfers most cleanly when observations, actions, normalization,
  and physiology retain the same meanings.
- The critic predicts return under the old task distribution. Reset or rapidly
  adapt it when reward meaning changes materially.
- PPO is on-policy. Retention rehearsal should collect fresh rollouts from old
  lessons with the current policy. It should not train PPO repeatedly on stale
  transitions.

A practical initial task mixture is 70% current lesson, 20% direct predecessor,
and 10% all earlier lessons. Treat those numbers as an experiment baseline. Raise
the earlier-lesson fraction when a retention gate fails. Prioritized Level Replay
supports sampling levels by current learning potential instead of using a fixed
uniform distribution
([Jiang, Grefenstette, and Rocktaschel, 2021](https://proceedings.mlr.press/v139/jiang21b.html)).

Promote difficulty only after several consecutive evaluations pass. Automatic
Domain Randomization used performance thresholds to expand the environment
distribution, and then trained on that expanded distribution
([OpenAI et al., 2019](https://arxiv.org/abs/1910.07113)). A hard switch from one
lesson to the next invites forgetting.

## Define the objective before calculating reward

There are two defensible objectives. They are not equivalent.

1. Pure survival: maximize expected lifetime.
2. Healthy survival: maximize time alive while keeping physiological variables
   near safe setpoints.

If eating and drinking bonuses are added to a pure-survival return, they are
reward shaping. If resource acquisition is part of the declared goal, they are
task reward. A run cannot be called unshaped until the task objective is stated.

Healthy survival is the better fit for this ecosystem. Define normalized
physiological drive from deficits:

```text
D(s) = w_food * food_deficit(s)^2
     + w_water * water_deficit(s)^2
     + w_hp * injury_fraction(s)^2
     + w_exposure * exposure_deficit(s)^2

sum(weights) = 1
0 <= D(s) <= 1
```

One stable task reward is:

```text
r_task(s, s') = alive(s') * (dt / reference_lifetime) * (1 - rho * D(s'))
0 <= rho <= 1
```

This rewards healthy time alive. Keeping the live reward nonnegative prevents
early death from becoming a way to escape accumulating physiology penalties.
Death ends future survival reward. Use one reward path for HP and needs instead
of separately rewarding survival, HP, and the same physiological improvement
several times.

If direct resource feedback is needed, derive it from actual absorbed benefit:

```text
r_resource = eta * max(0, D(before_consumption) - D(after_consumption))
```

This gives no reward for eating when full, handles food and water with the same
rule, and extends to prey consumption and shelter recovery. It is part of the
task objective if retained permanently. Otherwise label it shaping and remove it
before qualification.

Homeostatic drive reduction is biologically motivated, but its weights still
encode application policy. Measure the resulting tradeoffs. Do not assume the
formula makes them correct.

## Audit of the current reward

The current default has a 20-second horizon, food reward 8, drink reward 10, and
a maximum need multiplier of 1.25. Terminal survival is:

```text
survived_seconds * remaining_hp / maximum_hp
```

Its maximum is 20. One urgent food event pays 10 and one urgent drink pays 12.5.
Resource collection followed by death can therefore score 22.5, more than a
full-health 20-second survival episode before any other resource events. The
current profile defines resource collection as a stronger objective than
survival. That may teach the tutorial, but it does not express pure survival.

The current held-out run reached 100% eating-and-drinking success and the
20-second horizon. This establishes the tutorial mechanism. It does not establish
long-horizon survival, retention under scarcity, or reward robustness. Increase
the horizon and vary physiology before selecting weights for later lessons.

Use an explicit counterfactual test for every reward revision:

- Compare full survival with no unnecessary consumption.
- Compare urgent consumption followed by immediate death.
- Compare repeated consumption with stable homeostasis.
- Compare cautious survival with risky resource hoarding.
- Verify the intended behavior has the largest return in each case.

## Distance to food

Do not add `-distance_to_food` on every step. It changes the objective and can
reward camping near food. A raw progress term can also reward oscillation when
targets respawn or the selected nearest target changes.

Prefer changing the lesson distribution:

- Spawn one compatible food item inside the initial field of view.
- Start hungry enough that eating has immediate physiological value.
- Keep the initial bearing narrow and distance short.
- Expand bearing, distance, occlusion, map size, and distractors after success.
- Randomize left and right placement from the first lesson that permits turning.

If this still fails, use temporary potential-based shaping:

```text
r'(s, a, s') = r_task(s, a, s') + beta * (gamma * Phi(s') - Phi(s))
Phi(s) = -normalized_path_distance_to_current_compatible_resource(s)
Phi(terminal) = 0
```

Potential-based shaping is the standard policy-invariant form under its stated
MDP conditions
([Ng, Harada, and Russell, 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)).
Finite episodic tasks require correct terminal handling. Deep function
approximation, partial observation, target switching, and an incorrect potential
can still make training fail. Use traversable path distance if obstacles exist;
Euclidean distance can reward movement into a wall.

Treat distance shaping as nursery assistance. Anneal `beta` to zero and qualify
the resulting policy under the stable task reward. OpenAI's competitive
self-play experiments used an exploration reward early and annealed it to zero
before optimizing the competition objective
([Bansal et al., 2018](https://openai.com/index/competitive-self-play/)).

## Recommended curriculum

### Lesson 0: mechanics and exploration checks

Verify that a stochastic policy can move, turn, perceive, contact, eat, drink,
take damage, and terminate. This is an environment test, not a learned stage.
Check exploration scale because a policy that never discovers an event has no
useful policy-gradient signal from that event.

### Lesson 1: eat

Use one bunny, one visible food item, no water pressure, and a short map. Keep the
final observation and action shapes. Make food placement random within a narrow
achievable range. Promote when held-out episodes show eating across left, right,
near, and far placement without overeating or a fixed open-loop route.

### Lesson 2: food and water homeostasis

Activate thirst. Require both resources to survive the evaluation horizon. Vary
their order and sides independently. Select checkpoints with a constrained rule:
maximize healthy survival subject to both eating and drinking gates. An average
return can hide a water-only or food-only policy.

### Lesson 3: shelter and exposure

Define shelter's causal function first. For example, exposure rises outdoors and
falls under shelter. Exposure joins the same drive function. Do not add reward
for entering shelter if shelter already improves physiological state. Randomize
weather timing and shelter placement.

### Lesson 4: robust solo survival

Mix food, water, shelter, occlusion, longer horizons, and procedural layouts.
Increase one difficulty axis at a time until the agent succeeds, then train on a
range containing easier and harder values. Evaluate on unseen layout seeds.

### Lesson 5: scarcity competition

Start with several bunnies and enough resources for all. Reduce food-per-agent,
well capacity, and refill rate only after every identity succeeds. Use one shared
bunny policy with individual healthy-survival reward. Randomize spawn assignment
and resolve simultaneous contacts fairly. Do not reward collision, blocking,
pushing, winning, or movement unless those are independently desired goals.
Scarcity will create competition through the resource dynamics.

Measure the whole population: lower-tail lifetime, number of resource consumers,
consumption by identity, resource share inequality, starvation causes, and
retention on solo survival. A high mean can hide one policy role sacrificing
some spawn positions.

### Lesson 6: predator and prey

Initialize bunnies from the qualifying competition actor. Initialize the fox
against slow, scripted, or frozen prey so predation is discoverable. Then
alternate species updates or freeze one species for training windows. Maintain
an opponent pool containing current, early, middle, and best historical policies.
Evaluate a cross-play matrix.

Use separate shared policies for the two species. Each species keeps the same
healthy-survival objective:

- Bunny food reduces bunny hunger.
- Fox predation reduces fox hunger.
- Water and exposure use the same homeostatic rules for both species.
- Predation terminates prey and removes its future survival reward.

A separate kill bonus is unnecessary if prey is food and healthy survival is the
goal. If foxes never discover predation, first simplify prey and geometry. A
small temporary predation shaping signal is a later fallback and must be
annealed before qualification.

Learning opponents make a multi-agent environment non-stationary. Centralized
training with decentralized actors and historical opponent policies are standard
responses
([Lowe et al., 2017](https://papers.nips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html),
[Yu et al., 2022](https://arxiv.org/abs/2103.01955)). Self-play can create an
automatic curriculum, but it can also cycle or overfit to the current opponent.
OpenAI's hide-and-seek work used simple competitive rewards and observed both
emergent strategies and physics exploits at very large training scale
([Baker et al., 2019](https://openai.com/index/emergent-tool-use/)).

### Lesson 7: full ecosystem

Add obstacles, thorns, weather, and richer shelter while retaining all earlier
lessons in the rollout mixture. A final policy qualifies only if it passes the
new ecosystem gates and the frozen evaluation suites for eating, food-water,
shelter, solo survival, competition, and predator-prey.

## Transfer and retention protocol

Keep observation meanings, action meanings, timestep, physiology units, reward
scale, actor architecture, and normalization stable across stages. Include task
context only when two worlds require different actions from indistinguishable
observations.

At a stage transition:

1. Save the qualifying predecessor and its fixed evaluation suites.
2. Load the actor, encoder, and LSTM.
3. Reset optimizer state when the task distribution changes sharply.
4. Reset the critic when reward semantics or global-state meaning changes.
5. Warm the critic with the actor fixed for a short measured period if unstable
   value errors damage the transferred actor.
6. Gradually change the rollout mixture instead of switching environments at
   one update boundary.
7. Select checkpoints by new-stage performance subject to prior-stage retention
   floors.

If one network still forgets, first increase fresh old-task rehearsal. More
complex options include EWC, policy distillation, modular skills, or progressive
networks. Progressive networks avoid forgetting by retaining prior columns, but
their parameter cost grows with tasks
([Rusu et al., 2016](https://arxiv.org/abs/1606.04671)). EWC reduced sequential
forgetting in Atari by constraining parameters important to old tasks
([Kirkpatrick et al., 2017](https://arxiv.org/abs/1612.00796)). These mechanisms
should follow a measured rehearsal failure, not precede it.

## Evidence and experiment order

Keep separate training, validation, and test seeds. Use validation only for
curriculum promotion and checkpoint selection. Reload the selected checkpoint in
a fresh process for test evaluation. Run multiple training seeds.

Report at least:

- healthy-survival return and lifetime;
- food, water, shelter, and predation success separately;
- conjunction rates such as `ate_and_drank`;
- each death cause;
- time to first required resource;
- performance on every previous lesson;
- population-tail and identity metrics for competition;
- cross-play results for predator and prey;
- shaped and unshaped results when shaping was used;
- transferred and from-scratch results at the same environment-step budget.

Use this experiment order:

1. Train the stable task reward with an easy environment distribution.
2. If discovery fails, raise exploration and simplify placement.
3. If discovery still fails, add potential-based shaping as a labeled ablation.
4. Anneal shaping to zero.
5. Verify the unshaped policy on held-out seeds.
6. Advance one difficulty axis.
7. Add prior lessons to every later PPO collection mixture.
8. Reject a stage when any required mechanism or retention gate fails, even if
   mean return rises.

Reward curves show optimization of the implemented scalar. Behavioral gates
show whether that scalar produced the intended ecosystem behavior.
