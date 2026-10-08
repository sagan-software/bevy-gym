# Reusing verified Gymnasium output

Status: implemented; cache miss/hit execution awaits CI, 2026-10-08.

The last successful Pages build took 63 minutes. Its Gymnasium build and browser
checks took 36 minutes 27 seconds. Changes limited to the drone viewer's source
do not change those inputs. The cache retains the verified Gymnasium site and its reports.
Each deployment still builds the drone viewer and checks the browser runtime.

## Contract

- An exact cache hit may reuse a successful qualification of identical inputs.
  A miss runs the existing complete `nix run .#gymnasium-check` command.
- The key includes runner OS, architecture, image name/version, and the source
  hash. Source inputs include Rust manifests/lockfile/toolchain/configuration,
  Nix definitions, library source, shared ecosystem modules, Gymnasium files,
  its asset test, copied Acrobot license, and the Pages workflow.
- No prefix fallback is allowed. A changed input or runner image requires a new
  successful qualification before an archive can be saved under that key.
- When qualification succeeds, save before adding drone, gallery, or ecosystem output.
  Use the restore step's primary key for every save.
- Retain the qualifying commit, run URL, and key beside the test reports.
  A cache hit keeps that original evidence instead of claiming a fresh test run.
- Manual workflow dispatch always reruns qualification and does not save a cache.
- Keep browser runtime checks, ecosystem checks, drone builds, and deployment
  outside this cache. The repository's strict CI remains independent.

The [cache restore v5 documentation](https://github.com/actions/cache/blob/main/restore/README.md)
defines an exact hit and distinguishes prefix fallbacks. The
[cache save v5 documentation](https://github.com/actions/cache/blob/main/save/README.md)
defines explicit saves and reuse of the primary key. The
[runner image setup](https://github.com/actions/runner-images/blob/main/images/ubuntu/scripts/build/configure-environment.sh)
supplies `ImageOS` and `ImageVersion`. These sources control the workflow choices.

When Gymnasium starts reading another repository path, add that path to the key
before publishing. The current source and Trunk inputs were inspected; a directory
name alone does not prove that future inputs will remain covered.

## Acceptance checklist

- [x] Inspect primary action documentation and all current build/test inputs.
- [x] Validate the workflow with actionlint 1.7.12.
- [x] Audit included dependencies and excluded drone-only inputs.
- [ ] Observe a cache miss that qualifies, saves, and deploys successfully.
- [ ] Observe an exact hit that retains provenance and builds fresh drone output.
- [ ] Confirm changed inputs and manual dispatch cannot skip qualification.

The cache does not add new training qualification. Existing opt-in Pendulum,
Acrobot learning, and Acrobot compatibility tests retain their current exclusions.
No deployment speed improvement is measured until a cache hit completes.

The input audit covered 167 tracked files and found no omitted current build/test
inputs. It also checked eight drone, gallery, and status paths that must remain
outside this cache. This audit checks the path list; it does not emulate GitHub's
digest algorithm or establish a successful cache restore.
