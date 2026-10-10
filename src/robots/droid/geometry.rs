//! Mannequin-derived local collision profile, not a human biomechanics model.
//!
//! Bind anchors follow the CC0 mannequin in `assets/robots/survival/manifest.json`.
//! Rotate its +Z forward by pi about Y to the robot convention -Z forward.
//! Dimensions are metres, masses kilograms, torque bounds newton-metres.
//! Rapier 0.36 derives inertia from each box and specified mass.
//! Joint semantics: <https://rapier.rs/docs/user_guides/rust/joints/>.

use super::{DroidActuator, DroidBody};
use rapier3d::prelude::Vector;
use DroidActuator as A;
use DroidBody as B;

/// Collider and inertial profile of one segment in the common bind frame.
pub(super) struct Segment {
    /// Bind centre in metres.
    pub centre: Vector,
    /// Box half extents in metres.
    pub half_extents: Vector,
    /// Segment mass in kilograms.
    pub mass: f32,
}

/// Read the fixed collision profile for a closed segment identity.
pub(super) fn segment(body: DroidBody) -> &'static Segment {
    SEGMENTS.get(body as usize).expect("closed segment index")
}

/// Box profiles in the stable `DroidBody::ALL` order.
static SEGMENTS: [Segment; 13] = [
    Segment {
        centre: Vector::new(0.0, 0.99, 0.028),
        half_extents: Vector::new(0.12, 0.085, 0.08),
        mass: 12.0,
    },
    Segment {
        centre: Vector::new(0.0, 1.25, 0.008),
        half_extents: Vector::new(0.15, 0.2, 0.09),
        mass: 24.0,
    },
    Segment {
        centre: Vector::new(0.0, 1.58, 0.0),
        half_extents: Vector::new(0.075, 0.1, 0.08),
        mass: 5.0,
    },
    Segment {
        centre: Vector::new(-0.3291, 1.4408, 0.06775),
        half_extents: Vector::new(0.135, 0.04, 0.04),
        mass: 2.0,
    },
    Segment {
        centre: Vector::new(-0.6476, 1.4408, 0.06775),
        half_extents: Vector::new(0.177, 0.035, 0.035),
        mass: 1.5,
    },
    Segment {
        centre: Vector::new(0.3291, 1.4408, 0.06775),
        half_extents: Vector::new(0.135, 0.04, 0.04),
        mass: 2.0,
    },
    Segment {
        centre: Vector::new(0.6476, 1.4408, 0.06775),
        half_extents: Vector::new(0.177, 0.035, 0.035),
        mass: 1.5,
    },
    Segment {
        centre: Vector::new(-0.089, 0.73195, 0.0),
        half_extents: Vector::new(0.06, 0.196, 0.06),
        mass: 7.0,
    },
    Segment {
        centre: Vector::new(-0.089, 0.31775, 0.0186),
        half_extents: Vector::new(0.05, 0.209, 0.05),
        mass: 3.5,
    },
    Segment {
        centre: Vector::new(-0.089, 0.045, -0.065),
        half_extents: Vector::new(0.055, 0.035, 0.13),
        mass: 0.5,
    },
    Segment {
        centre: Vector::new(0.089, 0.73195, 0.0),
        half_extents: Vector::new(0.06, 0.196, 0.06),
        mass: 7.0,
    },
    Segment {
        centre: Vector::new(0.089, 0.31775, 0.0186),
        half_extents: Vector::new(0.05, 0.209, 0.05),
        mass: 3.5,
    },
    Segment {
        centre: Vector::new(0.089, 0.045, -0.065),
        half_extents: Vector::new(0.055, 0.035, 0.13),
        mass: 0.5,
    },
];

/// One permitted rotational degree of freedom.
#[derive(Clone, Copy)]
pub(super) struct Axis {
    /// Policy output owning this degree of freedom.
    pub actuator: DroidActuator,
    /// Maximum absolute torque in newton-metres.
    pub torque_limit: f32,
    /// Rapier joint angular limits in radians; these are solver coordinates.
    pub limits: [f32; 2],
}

/// A linked pair with a coincident bind anchor and up to three rotational axes.
pub(super) struct Joint {
    /// Segment receiving the opposite torque.
    pub parent: DroidBody,
    /// Segment receiving the signed requested torque.
    pub child: DroidBody,
    /// Common bind anchor in metres.
    pub anchor: Vector,
    /// Parent-local X, Y and Z axes; absent axes are locked.
    pub axes: [Option<Axis>; 3],
}

/// The fixed twelve-joint tree; absent axes are locked.
pub(super) const JOINTS: [Joint; 12] = [
    Joint {
        parent: B::Pelvis,
        child: B::Torso,
        anchor: Vector::new(0.0, 1.0505, 0.0156),
        axes: [
            Some(Axis {
                actuator: A::SpineX,
                torque_limit: 80.0,
                limits: [-0.5, 0.5],
            }),
            Some(Axis {
                actuator: A::SpineY,
                torque_limit: 80.0,
                limits: [-0.5, 0.5],
            }),
            Some(Axis {
                actuator: A::SpineZ,
                torque_limit: 80.0,
                limits: [-0.5, 0.5],
            }),
        ],
    },
    Joint {
        parent: B::Torso,
        child: B::Head,
        anchor: Vector::new(0.0, 1.4876, 0.0105),
        axes: [
            Some(Axis {
                actuator: A::NeckX,
                torque_limit: 8.0,
                limits: [-0.4, 0.4],
            }),
            Some(Axis {
                actuator: A::NeckY,
                torque_limit: 8.0,
                limits: [-0.4, 0.4],
            }),
            Some(Axis {
                actuator: A::NeckZ,
                torque_limit: 8.0,
                limits: [-0.4, 0.4],
            }),
        ],
    },
    Joint {
        parent: B::Torso,
        child: B::LeftUpperArm,
        anchor: Vector::new(-0.1919, 1.4408, 0.0654),
        axes: [
            Some(Axis {
                actuator: A::LeftShoulderX,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
            Some(Axis {
                actuator: A::LeftShoulderY,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
            Some(Axis {
                actuator: A::LeftShoulderZ,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
        ],
    },
    Joint {
        parent: B::LeftUpperArm,
        child: B::LeftForearm,
        anchor: Vector::new(-0.4663, 1.4408, 0.0701),
        axes: [
            None,
            Some(Axis {
                actuator: A::LeftElbowY,
                torque_limit: 20.0,
                limits: [-2.4, 0.1],
            }),
            None,
        ],
    },
    Joint {
        parent: B::Torso,
        child: B::RightUpperArm,
        anchor: Vector::new(0.1919, 1.4408, 0.0654),
        axes: [
            Some(Axis {
                actuator: A::RightShoulderX,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
            Some(Axis {
                actuator: A::RightShoulderY,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
            Some(Axis {
                actuator: A::RightShoulderZ,
                torque_limit: 30.0,
                limits: [-1.8, 1.8],
            }),
        ],
    },
    Joint {
        parent: B::RightUpperArm,
        child: B::RightForearm,
        anchor: Vector::new(0.4663, 1.4408, 0.0701),
        axes: [
            None,
            Some(Axis {
                actuator: A::RightElbowY,
                torque_limit: 20.0,
                limits: [-0.1, 2.4],
            }),
            None,
        ],
    },
    Joint {
        parent: B::Pelvis,
        child: B::LeftThigh,
        anchor: Vector::new(-0.089, 0.9321, -0.0014),
        axes: [
            Some(Axis {
                actuator: A::LeftHipX,
                torque_limit: 120.0,
                limits: [-1.5, 1.5],
            }),
            Some(Axis {
                actuator: A::LeftHipY,
                torque_limit: 120.0,
                limits: [-0.7, 0.7],
            }),
            Some(Axis {
                actuator: A::LeftHipZ,
                torque_limit: 120.0,
                limits: [-0.6, 0.6],
            }),
        ],
    },
    Joint {
        parent: B::LeftThigh,
        child: B::LeftCalf,
        anchor: Vector::new(-0.089, 0.5318, 0.0014),
        axes: [
            Some(Axis {
                actuator: A::LeftKneeX,
                torque_limit: 100.0,
                limits: [-2.4, 0.12],
            }),
            None,
            None,
        ],
    },
    Joint {
        parent: B::LeftCalf,
        child: B::LeftFoot,
        anchor: Vector::new(-0.089, 0.1037, 0.0358),
        axes: [
            Some(Axis {
                actuator: A::LeftAnkleX,
                torque_limit: 60.0,
                limits: [-0.65, 0.65],
            }),
            None,
            Some(Axis {
                actuator: A::LeftAnkleZ,
                torque_limit: 60.0,
                limits: [-0.4, 0.4],
            }),
        ],
    },
    Joint {
        parent: B::Pelvis,
        child: B::RightThigh,
        anchor: Vector::new(0.089, 0.9321, -0.0014),
        axes: [
            Some(Axis {
                actuator: A::RightHipX,
                torque_limit: 120.0,
                limits: [-1.5, 1.5],
            }),
            Some(Axis {
                actuator: A::RightHipY,
                torque_limit: 120.0,
                limits: [-0.7, 0.7],
            }),
            Some(Axis {
                actuator: A::RightHipZ,
                torque_limit: 120.0,
                limits: [-0.6, 0.6],
            }),
        ],
    },
    Joint {
        parent: B::RightThigh,
        child: B::RightCalf,
        anchor: Vector::new(0.089, 0.5318, 0.0014),
        axes: [
            Some(Axis {
                actuator: A::RightKneeX,
                torque_limit: 100.0,
                limits: [-2.4, 0.12],
            }),
            None,
            None,
        ],
    },
    Joint {
        parent: B::RightCalf,
        child: B::RightFoot,
        anchor: Vector::new(0.089, 0.1037, 0.0358),
        axes: [
            Some(Axis {
                actuator: A::RightAnkleX,
                torque_limit: 60.0,
                limits: [-0.65, 0.65],
            }),
            None,
            Some(Axis {
                actuator: A::RightAnkleZ,
                torque_limit: 60.0,
                limits: [-0.4, 0.4],
            }),
        ],
    },
];
