# Browser examples index

The deployment copies this directory to `/bevy-gym/examples/`. Each card opens
a working route: the manual drone viewer or an existing Classic Control task.
The browser workflow qualifies the existing tasks before publishing the site.
The drone's manual controls do not represent a trained policy.

`images/drone.png` is a 600 × 400 crop of the
[Bevy flight screenshot](../docs/progress/drone-flight.png), at source offset
`(340, 175)` pixels. It depicts NateGazzard's
[CC BY 3.0 model](../assets/robots/README.md). The index retains attribution.

The four Classic Control images came from Chromium renderer tests in
[run 37830890801](https://github.com/sagan-software/bevy-gym/actions/runs/37830890801):
`cartpole-upright`, `mountain-car-slope`, `pendulum-down`, and `acrobot-bent`.
Both MountainCar cards use the same track thumbnail. Thumbnails identify examples;
they do not establish complete visual or learning qualification.

The index uses the repository's [Mona Sans font](../assets/fonts/README.md).
Desktop and 390-pixel frame layouts were inspected. Device emulation remains unverified.
