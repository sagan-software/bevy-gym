# Navigation from sightings

This research baseline combines programmed navigation with the existing learned
motor pilot. It receives its own flight observation and filtered sight. It cannot
read the target's scripted route or hidden position. The initial research results
below predate integration into the playable scene.

The [earlier learned navigator](DRONE_SEARCH.md) could imitate cover approaches but
failed moving-target pursuit. This baseline provides a separate control reference
and prospective training demonstrations. It does not replace learned-search,
damaged-flight, adversarial-training, or browser requirements.

## Geometry and observations

The planner copies the authored static boxes into a private Rapier 0.36.0 world.
The world contains no moving character. The versioned
[PhysicsWorld documentation](https://docs.rs/rapier3d/0.36.0/rapier3d/pipeline/struct.PhysicsWorld.html#method.cast_shape)
and installed crate source define the shape queries used here.

The graph has 39 by 39 cells at 0.5-metre spacing, with X/Z coordinates from
-9.5 through 9.5 metres and Y fixed at 2 metres. Adjacent cells and command
shortcuts require a clear sphere sweep. This is horizontal planning, not full
three-dimensional pathfinding. Normal clearance uses a 0.9-metre sphere.

If flight drifts inside that margin, the planner first tries an escape toward a
clear cell using a 0.45-metre sphere. The physical body's half dimensions are
0.287, 0.104, and 0.297 metres. Its circumscribed radius is below 0.426 metres,
so the escape sphere encloses that collider at any orientation. This geometric
check does not prove that the motor controller will follow the segment exactly.

Navigation runs every 100 milliseconds; the frozen motor pilot acts every
20 milliseconds and resets its own recurrent memory each action. The heading
reference changes by at most 0.1 radians per navigation decision. That limits
the command reference to one radian per second, not the actual body's angular speed.

The planner estimates horizontal target velocity from consecutive visible samples,
limits the estimate to four metres per second, and uses a half-second lead while
tracking. It prefers a six-metre viewing distance. Investigation retains the last
observed point and velocity for at most ten seconds.

The research prototype samples potential target positions on a one-metre grid at Y = 1 metre.
It records when a sample was within the body-centred viewing cone and had a clear
static ray. Recently checked samples are skipped for ten seconds. A clear ray to
a sample does not prove an entire cell is empty. The production sensor uses an
offset eye; aligning this heuristic with that origin remains an integration check.

The query graph and search samples come from geometry, with no named house or
pipe destination. The planner chooses reachable viewing cells and uses clear
shortcuts up to 2.5 metres long. The research prototype allocates graph-search
scratch storage each decision. No browser-performance or allocation claim is made.

## Measured failures and changes

The first map planner survived all 25 selection trials but never saw either held
target. It continued looking at an empty last-seen point. Adding recently checked
search samples recovered all five house cases, while pipe cases remained unseen.

The next trace exposed a different failure: the drone entered the conservative
clearance margin and could no longer connect to the graph. Allowing the smaller
escape sphere recovered all five pipe cases. However, eight moving-target trials
crashed. Limiting heading-reference changes then kept all 25 trials airborne.

Increasing the viewing distance from four to six metres improved moving-target
visibility. That version survived all 25 selection trials and all 160 fresh-seed
trials. Both hidden routes ended with uninterrupted sight in all selection trials,
but two fresh pipe trials never saw the held target. The lowest fresh lateral
visibility was only 958 of 3,000 physics samples.

The initial scan could be interrupted by exploration before the drone finished
turning. The latest variant starts scanning from the measured body heading and
allows up to eight seconds of scanning before exploration if it has no sighting.
Any visible target interrupts scanning immediately. Its test includes all prior
selection and qualification seeds plus 16 new seeds per route.

## Latest native results

All 265 trials survived 60 seconds. The set covers five routes for 53 seeds:
the five selection seeds, 32 earlier qualification seeds, and 16 new seeds.
The lowest overall sight count was 1,968 of 3,000 physics samples for lateral
movement, or 65.6%. Every house and pipe trial ended with ten uninterrupted
seconds of sight.

House trials saw the held robot in at least 500 of 506 sampled decisions.
Pipe trials saw it in at least 464 of 522. The longest sampled gaps while the
robot stayed inside were 0.6 seconds for the house and 5.8 seconds for the pipe.
The earlier provisional 90% held-visibility gate still fails all 53 pipe cases.
The other four routes pass their earlier provisional checks. These results support
further integration work; they do not establish complete gameplay qualification.

![Programmed navigation over learned motor control](progress/drone-planner-paths.png)

The figure uses the same first new seed for every route. The house path finds a
view through the doorway. It does not establish window-only search. Target paths,
speed, arena geometry, and healthy motor configuration remain fixed.

The [research record](progress/drone-planner-baseline.json) retains exact sources,
logs, variant summaries, and restore instructions. The
[compressed trial archive](progress/drone-planner-trials.json.gz) preserves all raw
outputs, including failed variants. Each decompressed entry contains its original
path, SHA-256, and exact UTF-8 text. No research process remains active.

## Playable integration

The playable scene now uses this programmed controller over the same frozen motor
pilot. The [pursuit-search guide](../examples/robots/pursuit_search.rs) demonstrates
the loop without rendering. Navigation consumes one filtered observation every
100 milliseconds; the motor pilot performs five 20-millisecond actions per goal.
Reset clears search history and retains the map and valid model weights.

The implementation uses the same body-local camera offset as the visible lens and
sight sensor. An initial remembered contact seeds investigation with its existing
age deducted. Repeated remembered contacts cannot refresh that deadline. After expiry,
stale remembered contacts remain ignored. A visible contact restarts investigation.
After the sensor clears to unknown, a new brief remembered contact can also restart it.

Graph search reuses buffers bounded by the fixed grid. A geometry test closes the
house door and checks that the planned route reaches a clear window view. Other
checks cover malformed sightings, memory expiry, reset, heading-reference limits,
blocked starts, clearance-margin escape, and unreachable destinations.

All 265 integrated native trials survived. The weakest lateral visibility increased
to 2,092 of 3,000 physics samples, or 69.7%. Every house and pipe trial ended with
ten uninterrupted seconds of sight. The earlier 90% held-visibility gate still
fails all 53 pipe cases. [The integration record](progress/drone-navigation-integration.json)
retains exact probe sources and compressed raw measurements.

Chrome/WASM passes 18 applicable navigation tests, including a minute-long physical
pursuit episode. Desktop and narrow browser recordings show the playable integration.
The first narrow review exposed a pipe-camera obstruction. A geometry regression
reproduces it, and the revised camera lowers its view under a ceiling. Desktop and
narrow captures confirm that the pipe interior and robot remain visible.
[Execution status](EXAMPLE_STATUS.md) tracks visual confirmation and remaining gates.

This is programmed search with learned flight. Learned search, hearing-directed
navigation, damage adaptation, and adversarial training remain in the active roadmap.
