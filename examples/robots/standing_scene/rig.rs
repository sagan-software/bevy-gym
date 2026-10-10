//! One-way projection from the physical segments to the licensed mannequin skeleton.

use super::{projection, Viewer};
use bevy::prelude::*;
use bevy_gym::robots::DroidBody;

/// Model lifecycle; a failed mapping never permits playback.
#[derive(Resource, Default)]
pub(super) enum Rig {
    /// The GLTF scene has not finished instantiating.
    #[default]
    Loading,
    /// Capture bind transforms after the scene-ready event and transform propagation.
    AwaitingBind,
    /// Exactly thirteen unique display bones with immutable bind frames.
    Ready(Box<[Bone; 13]>),
    /// An incomplete or incompatible skeleton cannot display policy motion.
    Failed(String),
}

/// Cached display bone; no writable physics object is retained.
pub(super) struct Bone {
    /// Entity owning the render-local transform.
    entity: Entity,
    /// Physical segment owning this bone.
    body: DroidBody,
    /// Immutable global model bind frame, already rotated to -Z forward.
    bind: GlobalTransform,
    /// Immediate parent's immutable global bind frame.
    parent_bind: GlobalTransform,
    /// Nearest mapped ancestor, if any, which moves that immediate parent.
    ancestor: Option<DroidBody>,
}

/// Capture once after Bevy 0.18.1 propagates the loaded model's bind transforms.
/// <https://docs.rs/bevy/0.18.1/bevy/prelude/struct.GlobalTransform.html>.
pub(super) fn capture(
    mut rig: ResMut<'_, Rig>,
    nodes: Query<'_, '_, (Entity, &Name, &GlobalTransform, Option<&ChildOf>)>,
    hierarchy: Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) {
    if !matches!(*rig, Rig::AwaitingBind) {
        return;
    }
    *rig = match bind(&nodes, &hierarchy) {
        Ok(bones) => Rig::Ready(bones),
        Err(error) => Rig::Failed(error),
    };
}

/// Validate the complete mapping before retaining any usable display rig.
fn bind(
    nodes: &Query<'_, '_, (Entity, &Name, &GlobalTransform, Option<&ChildOf>)>,
    hierarchy: &Query<'_, '_, (Option<&Name>, &GlobalTransform, Option<&ChildOf>)>,
) -> Result<Box<[Bone; 13]>, String> {
    let mut bones = Vec::with_capacity(13);
    for (entity, name, bind, parent) in nodes {
        let Some(body) = projection::body_named(name.as_str()) else {
            continue;
        };
        if bones.iter().any(|bone: &Bone| bone.body == body) {
            return Err(format!("duplicate standing bone: {name}"));
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
        // Walk the actual hierarchy, including unmapped spine and clavicle bones.
        let mut ancestor = None;
        let mut cursor = parent.map(ChildOf::parent);
        while let Some(entity) = cursor {
            let (name, _pose, parent) = hierarchy.get(entity).map_err(|error| error.to_string())?;
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
            format!("standing model needs thirteen unique body bones, found {count}")
        })
}

/// Use one observation for every bone and its parent, avoiding a previous-frame parent pose.
pub(super) fn project(
    viewer: Res<'_, Viewer>,
    mut rig: ResMut<'_, Rig>,
    mut transforms: Query<'_, '_, &mut Transform>,
) {
    let (Ok(session), Rig::Ready(bones)) = (&viewer.session, &*rig) else {
        return;
    };
    let observation = session.observation();
    for bone in bones.iter() {
        let target = projection::world_pose(bone.body, observation.body(bone.body), bone.bind);
        let parent = bone.ancestor.map_or(bone.parent_bind, |body| {
            projection::world_pose(body, observation.body(body), bone.parent_bind)
        });
        let Ok(mut local) = transforms.get_mut(bone.entity) else {
            *rig = Rig::Failed("standing model lost a mapped bone transform".to_owned());
            return;
        };
        *local = target.reparented_to(&parent);
    }
}

/// Capture a complete synthetic skeleton for control tests that do not load assets.
#[cfg(test)]
pub(super) fn ready_fixture() -> Rig {
    let mut app = App::new();
    app.insert_resource(Rig::AwaitingBind)
        .add_systems(Update, capture);
    for name in [
        "pelvis",
        "spine_01",
        "neck_01",
        "upperarm_l",
        "lowerarm_l",
        "upperarm_r",
        "lowerarm_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "thigh_r",
        "calf_r",
        "foot_r",
    ] {
        app.world_mut()
            .spawn((Name::new(name), GlobalTransform::IDENTITY));
    }
    app.update();
    app.world_mut()
        .remove_resource::<Rig>()
        .expect("captured skeleton")
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Missing or repeated body identities prevent the display rig from becoming usable.
    #[test]
    fn incomplete_and_duplicate_skeletons_fail_visibly() {
        let mut app = App::new();
        app.insert_resource(Rig::AwaitingBind)
            .add_systems(Update, capture);
        app.update();
        assert!(matches!(app.world().resource::<Rig>(), Rig::Failed(_)));
        for _ in 0..2 {
            app.world_mut()
                .spawn((Name::new("pelvis"), GlobalTransform::IDENTITY));
        }
        *app.world_mut().resource_mut::<Rig>() = Rig::AwaitingBind;
        app.update();
        let error = match app.world().resource::<Rig>() {
            Rig::Failed(error) => Some(error),
            _ => None,
        }
        .expect("duplicate must fail");
        assert!(error.contains("duplicate"));
    }

    /// Unmapped ancestors inherit their nearest physical segment from this same observation.
    #[test]
    fn capture_keeps_unmapped_parent_offsets_without_changing_physics() {
        let mut app = App::new();
        app.insert_resource(Rig::AwaitingBind)
            .add_systems(Update, capture);
        let root = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let pelvis = app
            .world_mut()
            .spawn((
                Name::new("pelvis"),
                GlobalTransform::IDENTITY,
                ChildOf(root),
            ))
            .id();
        let intermediate = app
            .world_mut()
            .spawn((
                Name::new("clavicle"),
                GlobalTransform::from_xyz(0.0, 1.0, 0.0),
                ChildOf(pelvis),
            ))
            .id();
        for name in [
            "spine_01",
            "neck_01",
            "upperarm_l",
            "lowerarm_l",
            "upperarm_r",
            "lowerarm_r",
            "thigh_l",
            "calf_l",
            "foot_l",
            "thigh_r",
            "calf_r",
            "foot_r",
        ] {
            app.world_mut().spawn((
                Name::new(name),
                GlobalTransform::IDENTITY,
                ChildOf(intermediate),
            ));
        }
        app.update();
        let bones = match app.world().resource::<Rig>() {
            Rig::Ready(bones) => Some(bones),
            _ => None,
        }
        .expect("complete rig");
        assert_eq!(bones.len(), 13);
        assert!(bones
            .iter()
            .any(|bone| bone.body == DroidBody::Pelvis && bone.ancestor.is_none()));
        assert!(bones
            .iter()
            .filter(|bone| bone.body != DroidBody::Pelvis)
            .all(|bone| bone.ancestor == Some(DroidBody::Pelvis)
                && bone.parent_bind.translation() == Vec3::Y));
        // Once captured, a later frame cannot replace immutable bind data.
        app.world_mut()
            .get_mut::<GlobalTransform>(intermediate)
            .expect("parent")
            .clone_from(&GlobalTransform::from_xyz(9.0, 9.0, 9.0));
        app.update();
        let bones = match app.world().resource::<Rig>() {
            Rig::Ready(bones) => Some(bones),
            _ => None,
        }
        .expect("retained rig");
        assert!(bones
            .iter()
            .filter(|bone| bone.body != DroidBody::Pelvis)
            .all(|bone| bone.parent_bind.translation() == Vec3::Y));
        let session = super::super::session::load(
            include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
            include_bytes!("../../../docs/progress/droid-standing-trial.json"),
            42,
        )
        .expect("RL candidate");
        let snapshot = session.observation();
        let expected: Vec<_> = bones
            .iter()
            .map(|bone| {
                let target = projection::world_pose(bone.body, snapshot.body(bone.body), bone.bind);
                let parent = bone.ancestor.map_or(bone.parent_bind, |body| {
                    projection::world_pose(body, snapshot.body(body), bone.parent_bind)
                });
                (bone.entity, target.reparented_to(&parent))
            })
            .collect();
        for (entity, _) in &expected {
            app.world_mut()
                .entity_mut(*entity)
                .insert(Transform::IDENTITY);
        }
        app.insert_resource(Viewer {
            title: "Trial",
            session: Ok(session),
            checkpoint_label: String::new(),
            playback: super::super::Playback::Paused,
            speed: super::super::Speed::One,
        })
        .add_systems(Update, project);
        app.update();
        for (entity, expected) in expected {
            assert_eq!(app.world().get::<Transform>(entity), Some(&expected));
        }
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .observation(),
            snapshot
        );
        // Missing transforms and failed sessions cannot mutate physical state.
        app.world_mut().entity_mut(pelvis).remove::<Transform>();
        app.update();
        assert!(matches!(app.world().resource::<Rig>(), Rig::Failed(_)));
        app.world_mut().resource_mut::<Viewer>().session = Err("invalid checkpoint".to_owned());
        app.update();
    }
}
