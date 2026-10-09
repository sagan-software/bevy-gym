# Ragdoll compatibility source

These two crates come from [bevy-ragdoll](https://github.com/sagan-software/bevy-ragdoll/tree/05ca5a920c88cec9cfaa661c483aa4e26924e08f),
revision `05ca5a920c88cec9cfaa661c483aa4e26924e08f`.
Only the unpublished `bevy-gym-pursuit-viewer` package uses them.

- `bevy-ragdoll/src` copies upstream `src`.
- `bevy-ragdoll-rapier3d/src` copies upstream `crates/bevy_ragdoll_rapier3d/src`.
- Both directories retain the upstream MIT and Apache-2.0 licenses.
- Standalone manifests select Bevy 0.18.1 and bevy_rapier3d 0.34.0.
  Upstream selects Bevy 0.19.1 and bevy_rapier3d 0.36.
- The only source change replaces `WorldAssetRoot` with Bevy 0.18's `SceneRoot`
  in the core crate's introductory documentation example.

The adapter uses its own Rapier 0.32 physics world. The drone environment keeps
Rapier 0.36. No Rapier body, collider, handle, or query type crosses between them.
Arena boxes supply the ragdoll world's static geometry. Mannequin health selects
kinematic animation or dynamic death; reset restores kinematic animation.

The viewer sets `RagdollPhysicsSettings::force_sleep_after` to zero. The adapter's
optional timer ages kinematic bodies and survives mode changes. A longer browser
run exposed a Rapier 0.32 sleeping-island panic after reset. This setting disables
the adapter timer while retaining Rapier's automatic sleep. Browser qualification
must include waiting after death and continuing play after reset.

The viewer's real-model integration test checks falling, floor collision, stopped
animation, reset, and stable body count. This establishes the viewer's use case;
it does not establish compatibility for every upstream feature or backend.

To run that test:

```sh
nix develop --command cargo test -p bevy-gym-pursuit-viewer --example drone-pursuit loaded_mannequin_becomes
```

The browser build copies both license files from each crate into `LICENSES`.
Keep this revision, source-diff list, and validation scope current when updating.
