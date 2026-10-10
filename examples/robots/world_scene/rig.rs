//! Root-scoped licensed models project snapshots without retaining a physics handle.

use super::{player::Player, view::Models};
use crate::{drone_model, standing_scene::projection};
use bevy::{prelude::*, scene::SceneInstanceReady};
use bevy_gym::robots::{DroidBody, DroneMotor, RobotId, RobotSlot};

/// Model lifecycle retains either a complete immutable bind cache or a visible failure.
pub(super) enum State<T> {
    /// The scene instance is still loading.
    Loading,
    /// Capture original bind data after transform propagation.
    AwaitingBind,
    /// Every required node has a unique owning physical identity.
    Ready(T),
    /// Mapping or loading failed; this model cannot enable playback.
    Failed(String),
}

/// Each model kind retains only its own valid bind-cache type.
#[derive(Component)]
pub(super) enum Visual {
    /// Drone meshes map exactly four named rotors.
    Drone {
        /// Stable drone identity.
        slot: RobotSlot,
        /// Complete rotor cache or its model failure.
        state: State<Box<[Rotor; 4]>>,
    },
    /// Droid meshes map exactly thirteen body bones.
    Droid {
        /// Stable droid identity.
        slot: RobotSlot,
        /// Complete bone cache or its model failure.
        state: State<Box<[Bone; 13]>>,
    },
}

/// Immutable licensed skeleton frames; mutable transforms remain presentation data.
pub(super) struct Bone {
    /// Display bone entity within this model root.
    entity: Entity,
    /// Physical segment represented by this bone.
    body: DroidBody,
    /// Original global bind frame at model origin, rotated to -Z forward.
    bind: GlobalTransform,
    /// Original immediate-parent global bind frame.
    parent_bind: GlobalTransform,
    /// Nearest mapped ancestor, including intervening unmapped skeleton nodes.
    ancestor: Option<DroidBody>,
}

/// Cosmetic rotor motion reads actual requests and cannot choose an actuator value.
pub(super) struct Rotor {
    /// Named rotor entity belonging to this drone root.
    entity: Entity,
    /// Source-space mesh centre in metres.
    pivot: Vec3,
    /// Closed motor identity for the actual requested fraction.
    motor: DroneMotor,
    /// Dimensionless clockwise or counter-clockwise mesh rotation sign.
    sign: f32,
    /// Immutable local model transform before cosmetic spinning.
    bind: Transform,
}

impl Visual {
    /// Construct a loading drone; no ready cache can be borrowed from another root.
    pub(super) const fn drone(slot: RobotSlot) -> Self {
        Self::Drone {
            slot,
            state: State::Loading,
        }
    }
    /// Construct a loading droid with its own skeleton lifecycle.
    pub(super) const fn droid(slot: RobotSlot) -> Self {
        Self::Droid {
            slot,
            state: State::Loading,
        }
    }
    /// Derive identity from the model kind and closed slot.
    pub(super) const fn id(&self) -> RobotId {
        match self {
            Self::Drone { slot, .. } => RobotId::Drone(*slot),
            Self::Droid { slot, .. } => RobotId::Droid(*slot),
        }
    }
    /// A complete cache is required before physical motion is shown.
    const fn is_ready(&self) -> bool {
        matches!(
            self,
            Self::Drone {
                state: State::Ready(_),
                ..
            } | Self::Droid {
                state: State::Ready(_),
                ..
            }
        )
    }
    /// Borrow the first model failure for the visible readout.
    pub(super) fn error(&self) -> Option<&str> {
        match self {
            Self::Drone {
                state: State::Failed(error),
                ..
            }
            | Self::Droid {
                state: State::Failed(error),
                ..
            } => Some(error),
            Self::Drone {
                state: State::Loading | State::AwaitingBind | State::Ready(_),
                ..
            }
            | Self::Droid {
                state: State::Loading | State::AwaitingBind | State::Ready(_),
                ..
            } => None,
        }
    }
    /// Ready notifications cannot revive an existing failed or captured model.
    fn awaiting_bind(&mut self) {
        match self {
            Self::Drone { state, .. } => {
                if matches!(state, State::Loading) {
                    *state = State::AwaitingBind;
                }
            }
            Self::Droid { state, .. } => {
                if matches!(state, State::Loading) {
                    *state = State::AwaitingBind;
                }
            }
        }
    }
    /// Retain the first diagnostic; reset never substitutes a different model.
    fn fail(&mut self, error: String) {
        if self.error().is_some() {
            return;
        }
        match self {
            Self::Drone { state, .. } => *state = State::Failed(error),
            Self::Droid { state, .. } => *state = State::Failed(error),
        }
    }
}

/// Require one usable root for each of the six identities, with no extra roots.
pub(super) fn ready(visuals: &Query<'_, '_, &Visual>) -> bool {
    visuals.iter().count() == 6
        && visuals.iter().all(Visual::is_ready)
        && RobotSlot::ALL.into_iter().all(|slot| {
            visuals
                .iter()
                .filter(|visual| visual.id() == RobotId::Drone(slot))
                .count()
                == 1
                && visuals
                    .iter()
                    .filter(|visual| visual.id() == RobotId::Droid(slot))
                    .count()
                    == 1
        })
}

/// Bevy 0.18.1 triggers readiness on the exact scene root, keeping identical names separate.
/// <https://docs.rs/bevy/0.18.1/bevy/scene/struct.SceneInstanceReady.html>.
pub(super) fn scene_ready(
    ready: On<'_, '_, SceneInstanceReady>,
    mut visuals: Query<'_, '_, &mut Visual>,
) {
    if let Ok(mut visual) = visuals.get_mut(ready.entity) {
        visual.awaiting_bind();
    }
}

/// Report loader failures before another presentation tick can consume actions.
pub(super) fn asset_failures(
    assets: Res<'_, AssetServer>,
    models: Res<'_, Models>,
    mut visuals: Query<'_, '_, &mut Visual>,
) {
    for mut visual in &mut visuals {
        let handle = match *visual {
            Visual::Drone { .. } => &models.drone,
            Visual::Droid { .. } => &models.droid,
        };
        let load_state = assets.get_load_state(handle.id());
        if let Some(bevy::asset::LoadState::Failed(error)) = load_state {
            visual.fail(error.to_string());
        }
    }
}

/// Capture each original model at origin once; projection supplies all physical translations later.
pub(super) fn capture(
    mut visuals: Query<'_, '_, (Entity, &mut Visual)>,
    nodes: Query<
        '_,
        '_,
        (
            Entity,
            &Name,
            &GlobalTransform,
            &Transform,
            Option<&ChildOf>,
        ),
    >,
    hierarchy: Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) {
    for (root, mut visual) in &mut visuals {
        match &mut *visual {
            Visual::Drone { state, .. } if matches!(state, State::AwaitingBind) => {
                *state = match bind_rotors(root, &nodes, &hierarchy) {
                    Ok(rotors) => State::Ready(rotors),
                    Err(error) => State::Failed(error),
                };
            }
            Visual::Droid { state, .. } if matches!(state, State::AwaitingBind) => {
                *state = match bind_bones(root, &nodes, &hierarchy) {
                    Ok(bones) => State::Ready(bones),
                    Err(error) => State::Failed(error),
                };
            }
            Visual::Drone { .. } | Visual::Droid { .. } => {}
        }
    }
}

/// Walk only immutable display ancestry; another root's matching names never belong here.
fn within_root(
    entity: Entity,
    root: Entity,
    hierarchy: &Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) -> Result<bool, String> {
    let mut cursor = Some(entity);
    while let Some(entity) = cursor {
        if entity == root {
            return Ok(true);
        }
        let (_, _, parent) = hierarchy.get(entity).map_err(|error| error.to_string())?;
        cursor = parent.map(ChildOf::parent);
    }
    Ok(false)
}

/// Validate all thirteen unique bone identities before enabling this droid.
fn bind_bones(
    root: Entity,
    nodes: &Query<
        '_,
        '_,
        (
            Entity,
            &Name,
            &GlobalTransform,
            &Transform,
            Option<&ChildOf>,
        ),
    >,
    hierarchy: &Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) -> Result<Box<[Bone; 13]>, String> {
    let mut bones = Vec::with_capacity(13);
    for (entity, name, bind, _, parent) in nodes {
        let Some(body) = projection::body_named(name.as_str()) else {
            continue;
        };
        if !within_root(entity, root, hierarchy)? {
            continue;
        }
        if bones.iter().any(|bone: &Bone| bone.body == body) {
            return Err(format!("duplicate droid bone: {name}"));
        }
        let parent_bind = match parent {
            Some(parent) => {
                *hierarchy
                    .get(parent.parent())
                    .map_err(|error| error.to_string())?
                    .1
            }
            None => GlobalTransform::IDENTITY,
        };
        // Include unmapped clavicle and spine offsets, stopping at this model's own root.
        let mut ancestor = None;
        let mut cursor = parent.map(ChildOf::parent);
        while let Some(entity) = cursor {
            if entity == root {
                break;
            }
            let (name, _, parent) = hierarchy.get(entity).map_err(|error| error.to_string())?;
            ancestor = name.and_then(|name| projection::body_named(name.as_str()));
            if ancestor.is_some() {
                break;
            }
            cursor = parent.map(ChildOf::parent);
        }
        bones.push(Bone {
            entity,
            body,
            bind: *bind,
            parent_bind,
            ancestor,
        });
    }
    bones
        .into_boxed_slice()
        .try_into()
        .map_err(|bones: Box<[Bone]>| {
            let count = bones.len();
            format!("droid model needs thirteen unique body bones, found {count}")
        })
}

/// Retain exactly four distinct motor meshes within this drone's own scene root.
fn bind_rotors(
    root: Entity,
    nodes: &Query<
        '_,
        '_,
        (
            Entity,
            &Name,
            &GlobalTransform,
            &Transform,
            Option<&ChildOf>,
        ),
    >,
    hierarchy: &Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) -> Result<Box<[Rotor; 4]>, String> {
    let mut rotors = Vec::with_capacity(4);
    for (entity, name, _, bind, _) in nodes {
        let Some((pivot, motor, sign)) = drone_model::rotor(name.as_str()) else {
            continue;
        };
        if !within_root(entity, root, hierarchy)? {
            continue;
        }
        if rotors.iter().any(|rotor: &Rotor| rotor.motor == motor) {
            return Err(format!("duplicate drone rotor: {name}"));
        }
        rotors.push(Rotor {
            entity,
            pivot,
            motor,
            sign,
            bind: *bind,
        });
    }
    rotors
        .into_boxed_slice()
        .try_into()
        .map_err(|rotors: Box<[Rotor]>| {
            let count = rotors.len();
            format!("drone model needs four unique rotors, found {count}")
        })
}

/// Project one complete snapshot; all parent and child poses use the same physical frame.
pub(super) fn project(
    player: Res<'_, Player>,
    mut visuals: Query<'_, '_, (Entity, &mut Visual, &mut Visibility)>,
    mut transforms: Query<'_, '_, &mut Transform>,
) {
    let Ok(session) = player.session() else {
        return;
    };
    for (root, mut visual, mut visibility) in &mut visuals {
        let result = match &*visual {
            Visual::Drone {
                slot,
                state: State::Ready(rotors),
            } => project_drone(root, *slot, rotors, session, &mut transforms),
            Visual::Droid {
                slot,
                state: State::Ready(bones),
            } => project_droid(*slot, bones, session, &mut transforms),
            Visual::Drone { .. } | Visual::Droid { .. } => {
                *visibility = Visibility::Hidden;
                continue;
            }
        };
        match result {
            Ok(()) => *visibility = Visibility::Visible,
            Err(error) => {
                visual.fail(error);
                *visibility = Visibility::Hidden;
            }
        }
    }
}

/// Display the body and cosmetic rotor phase using this slot's actual last requested fractions.
fn project_drone(
    root: Entity,
    slot: RobotSlot,
    rotors: &[Rotor; 4],
    session: &crate::standing::world_session::Session,
    transforms: &mut Query<'_, '_, &mut Transform>,
) -> Result<(), String> {
    let body = session.snapshot().drone(slot);
    let physical = Transform::from_translation(body.position()).with_rotation(body.orientation());
    *transforms
        .get_mut(root)
        .map_err(|error| error.to_string())? =
        physical.mul_transform(drone_model::model_alignment());
    for rotor in rotors {
        let fraction = session.last_drone_action(slot).map_or(0.0, |action| {
            let [front_left, front_right, rear_right, rear_left] = action.fractions();
            match rotor.motor {
                DroneMotor::FrontLeft => front_left,
                DroneMotor::FrontRight => front_right,
                DroneMotor::RearRight => rear_right,
                DroneMotor::RearLeft => rear_left,
            }
        });
        // Forty radians/second at full request is cosmetic mesh speed, not simulated motor speed.
        let angle = session.snapshot().elapsed().as_secs_f32() * fraction * rotor.sign * 40.0;
        *transforms
            .get_mut(rotor.entity)
            .map_err(|error| error.to_string())? =
            drone_model::spin_about(rotor.pivot, angle).mul_transform(rotor.bind);
    }
    Ok(())
}

/// Preserve unmapped parent offsets while translating only this droid's physical segments.
fn project_droid(
    slot: RobotSlot,
    bones: &[Bone; 13],
    session: &crate::standing::world_session::Session,
    transforms: &mut Query<'_, '_, &mut Transform>,
) -> Result<(), String> {
    let observation = session.snapshot().droid(slot);
    for bone in bones {
        let target = projection::world_pose(bone.body, observation.body(bone.body), bone.bind);
        let parent = bone.ancestor.map_or(bone.parent_bind, |body| {
            projection::world_pose(body, observation.body(body), bone.parent_bind)
        });
        *transforms
            .get_mut(bone.entity)
            .map_err(|error| error.to_string())? = target.reparented_to(&parent);
    }
    Ok(())
}

#[cfg(test)]
#[path = "rig_tests.rs"]
mod tests;
