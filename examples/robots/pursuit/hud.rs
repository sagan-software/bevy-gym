//! Compact weapon controls and feedback, separate from the scene's movement buttons.

use super::{
    combat::{
        health::Damage,
        pistol::{FireError, PickupError},
    },
    controls,
    firing::Feedback,
    shot::target::Shot,
    Game,
};
use bevy::prelude::*;

/// Interactions consumed by the same input system as E and the fire key.
#[derive(Component)]
pub(super) enum ActionButton {
    /// Collect the nearby pistol once per press.
    PickUp,
    /// Fire repeatedly while held, respecting simulation cooldown.
    Fire,
}

/// Text updated only when meaningful combat state changes.
#[derive(Component)]
pub(super) struct Status;

/// Keep the target's current prototype status and weapon actions visible.
pub(super) fn setup(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    let font = assets.load("fonts/MonaSans-VariableFont.ttf");
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: px(112),
            left: px(16),
            right: px(16),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::FlexStart,
            row_gap: px(6),
            ..default()
        })
        .with_children(|root| {
            root.spawn((
                Status,
                Text::new("Hover policy · E to collect pistol"),
                TextFont {
                    font: font.clone(),
                    font_size: 16.0,
                    ..default()
                },
                Node {
                    padding: UiRect::all(px(8)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.09, 0.11, 0.12)),
            ));
            root.spawn(Node {
                column_gap: px(6),
                ..default()
            })
            .with_children(|row| {
                controls::button(row, "Pick up · E", ActionButton::PickUp, &font);
                controls::button(row, "Fire · Space", ActionButton::Fire, &font);
            });
        });
}

/// Avoid formatting or allocating HUD text while its values remain unchanged.
pub(super) fn project(
    game: Res<'_, Game>,
    mut text: Query<'_, '_, &mut Text, With<Status>>,
    mut previous: Local<'_, Option<(Feedback, u8, u8, bool)>>,
) {
    let state = (
        game.combat.feedback(),
        game.combat.rounds(),
        game.combat.target().health().body_hits_remaining(),
        game.flight.error().is_some(),
    );
    if *previous == Some(state) {
        return;
    }
    *previous = Some(state);
    let message = message(state.0);
    let rounds = state.1;
    let health = state.2;
    for mut text in &mut text {
        text.0 = if let Some(error) = game.flight.error() {
            format!("Flight stopped: {error} · R to reset")
        } else if game.combat.is_armed() {
            format!("Ammo {rounds} · Drone {health}/6 · {message}")
        } else {
            message.to_owned()
        };
    }
}

/// State-specific messages describe the action outcome without implying learned pursuit.
const fn message(feedback: Feedback) -> &'static str {
    match feedback {
        Feedback::Crashed => "Drone crashed · R to reset",
        Feedback::Unarmed => "Hover policy · E to collect pistol",
        Feedback::Armed => "Point to aim · Click to fire",
        Feedback::Pickup(PickupError::TooFar) => "Move closer to the pistol",
        Feedback::Pickup(PickupError::AlreadyOwned) => "Pistol already collected",
        Feedback::Pickup(PickupError::InvalidPosition) => "Cannot collect pistol here",
        Feedback::Rejected(FireError::Unarmed) => "Collect the pistol first",
        Feedback::Rejected(FireError::Empty) => "Out of ammo · R to reset",
        Feedback::Rejected(FireError::CoolingDown) => "Weapon cooling down",
        Feedback::Fired(Shot::Miss { .. }) => "Miss",
        Feedback::Fired(Shot::Wall { .. }) => "Shot blocked",
        Feedback::Fired(Shot::Hit {
            damage: Damage::RotorDestroyed(_),
            ..
        }) => "Rotor destroyed",
        Feedback::Fired(Shot::Hit {
            damage: Damage::Destroyed,
            ..
        }) => "Drone destroyed · R to reset",
        Feedback::Fired(Shot::Hit {
            damage: Damage::Hit,
            ..
        }) => "Target hit",
        Feedback::Fired(Shot::Hit {
            damage: Damage::Ignored,
            ..
        }) => "No damage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shot::target::Part;
    use bevy_gym::robots::DroneMotor;

    #[test]
    fn every_feedback_variant_has_an_exact_actionable_message() {
        let cases = [
            (Feedback::Unarmed, "Hover policy · E to collect pistol"),
            (Feedback::Crashed, "Drone crashed · R to reset"),
            (Feedback::Armed, "Point to aim · Click to fire"),
            (
                Feedback::Pickup(PickupError::TooFar),
                "Move closer to the pistol",
            ),
            (
                Feedback::Pickup(PickupError::AlreadyOwned),
                "Pistol already collected",
            ),
            (
                Feedback::Pickup(PickupError::InvalidPosition),
                "Cannot collect pistol here",
            ),
            (
                Feedback::Rejected(FireError::Unarmed),
                "Collect the pistol first",
            ),
            (
                Feedback::Rejected(FireError::Empty),
                "Out of ammo · R to reset",
            ),
            (
                Feedback::Rejected(FireError::CoolingDown),
                "Weapon cooling down",
            ),
            (Feedback::Fired(Shot::Miss { point: Vec3::ZERO }), "Miss"),
            (
                Feedback::Fired(Shot::Wall { point: Vec3::ZERO }),
                "Shot blocked",
            ),
        ];
        for (feedback, expected) in cases {
            assert_eq!(message(feedback), expected);
        }
        for (damage, expected) in [
            (Damage::Hit, "Target hit"),
            (Damage::Ignored, "No damage"),
            (Damage::Destroyed, "Drone destroyed · R to reset"),
            (
                Damage::RotorDestroyed(DroneMotor::FrontLeft),
                "Rotor destroyed",
            ),
        ] {
            assert_eq!(
                message(Feedback::Fired(Shot::Hit {
                    part: Part::Body,
                    point: Vec3::ZERO,
                    damage,
                })),
                expected
            );
        }
    }
}
