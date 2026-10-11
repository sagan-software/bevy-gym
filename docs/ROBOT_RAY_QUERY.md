# Shared-world collision queries

Run the read-only example:

```sh
nix develop --command cargo run --no-default-features --features robots --example robot-ray-query
```

`RobotWorld::cast_ray` reads the actual shared floor, three drone colliders and 39 droid segment colliders. It returns the nearest solid and its distance without advancing physics. Use the measurement for sensors or to resolve a policy-requested shot. Policies choose aim and trigger requests.

Construct `RobotRay` through `TryFrom<(Vec3, Vec3, f32)>`. Origin coordinates and distances are metres. Construction normalizes finite nonzero directions, including subnormal and maximum finite components. The inclusive range must be finite and positive. The normalized segment endpoint must remain finite in f32 coordinates.

Validation rejects origin, direction, distance, then endpoint failures through the corresponding `InvalidRobotRay` variant. A solid containing the origin reports positive zero. Sensor volumes do not block rays. `None` means no solid intersects the inclusive segment.

An excluded `RobotId` removes every collider of that robot. Droid exclusion covers all 13 segments. Other robots and the floor still block the ray. Equal distances use environment, drone slot, then droid slot and body order. Slot and body order follow `RobotSlot::ALL` and `DroidBody::ALL`.

Hits expose `RobotRayTarget` and metres through read-only accessors. No mutable physics handle is exposed. A stopped world returns its original error until reset. Nonfinite physical state is rejected before collision queries.

Direct shape reads preserve freshly inserted reset poses before the first broad-phase update. Rapier 0.36 describes its last-step query pipeline and solid-ray semantics in the [scene-query guide](https://rapier.rs/docs/user_guides/rust/scene_queries/). The fixed six-agent roster permits a bounded owner scan for each collider, with O(N) time and O(1) auxiliary space for N solids.
