//! Keyboard and held buttons produce the same closed movement action.

use super::{Game, Movement};
use bevy::prelude::*;

/// Current held action, sampled at the fixed simulation rate.
#[derive(Resource, Default)]
pub(super) struct Input {
    /// Direction after opposite buttons cancel.
    pub(super) movement: Movement,
    /// One captured interaction, retained until a fixed update consumes it.
    pub(super) action: Option<super::firing::Action>,
    /// Last world aim, retained while the pointer is over a control.
    pub(super) aim: Option<Dir3>,
}

/// Each on-screen button represents its displayed keyboard key.
#[derive(Component)]
pub(super) enum Control {
    /// Walk forward while held.
    Forward,
    /// Walk backward while held.
    Backward,
    /// Walk left while held.
    Left,
    /// Walk right while held.
    Right,
    /// Reset once per press.
    Reset,
}

/// Keep the scene dominant, with a compact title and held movement buttons.
pub(super) fn setup(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    let font = assets.load("fonts/MonaSans-VariableFont.ttf");
    commands
        .spawn(Node {
            width: percent(100.0),
            height: percent(100.0),
            padding: UiRect::all(px(16)),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|root| {
            label(root, "Pursuit arena", 26.0, &font);
            movement_buttons(root, &font);
        });
}

/// Combine held keyboard and pointer input; resetting clears movement for this frame.
pub(super) fn read(
    keys: Res<'_, ButtonInput<KeyCode>>,
    mut buttons: Query<'_, '_, (&Control, &Interaction, &mut BackgroundColor)>,
    mut input: ResMut<'_, Input>,
    mut game: ResMut<'_, Game>,
    mut reset_held: Local<'_, bool>,
) {
    let mut forward = keys.pressed(KeyCode::KeyW);
    let mut backward = keys.pressed(KeyCode::KeyS);
    let mut left = keys.pressed(KeyCode::KeyA);
    let mut right = keys.pressed(KeyCode::KeyD);
    let mut reset = keys.pressed(KeyCode::KeyR) || keys.just_pressed(KeyCode::KeyR);
    for (control, interaction, mut color) in &mut buttons {
        if *interaction == Interaction::Pressed {
            match control {
                Control::Forward => forward = true,
                Control::Backward => backward = true,
                Control::Left => left = true,
                Control::Right => right = true,
                Control::Reset => reset = true,
            }
            color.0 = Color::srgb(0.19, 0.36, 0.39);
        } else {
            color.0 = Color::srgb(0.09, 0.11, 0.12);
        }
    }
    if reset && !*reset_held {
        game.reset();
        input.action = None;
        input.aim = None;
    }
    *reset_held = reset;
    input.movement = if reset {
        Movement::Idle
    } else {
        Movement::from([forward, backward, left, right])
    };
}

/// Group held directions separately from the scene title.
fn movement_buttons(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>) {
    root.spawn(Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(6),
        ..default()
    })
    .with_children(|controls| {
        label(controls, "WASD to move · R to reset", 16.0, font);
        controls
            .spawn(Node {
                flex_wrap: FlexWrap::Wrap,
                column_gap: px(6),
                row_gap: px(6),
                ..default()
            })
            .with_children(|row| {
                for (label, key) in [
                    ("W", Control::Forward),
                    ("A", Control::Left),
                    ("S", Control::Backward),
                    ("D", Control::Right),
                    ("Reset", Control::Reset),
                ] {
                    button(row, label, key, font);
                }
            });
    });
}

/// A touch-sized held control uses the same action as its keyboard label.
pub(super) fn button(
    row: &mut ChildSpawnerCommands<'_>,
    label: &str,
    key: impl Component,
    font: &Handle<Font>,
) {
    row.spawn((
        Button,
        key,
        Node {
            min_width: px(48),
            min_height: px(48),
            padding: UiRect::all(px(12)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.11, 0.12)),
    ))
    .with_children(|button| {
        button.spawn((
            Text::new(label),
            TextFont {
                font: font.clone(),
                font_size: 18.0,
                ..default()
            },
        ));
    });
}

/// Keep labels readable over both sunlit concrete and shaded interiors.
fn label(parent: &mut ChildSpawnerCommands<'_>, text: &str, font_size: f32, font: &Handle<Font>) {
    parent.spawn((
        Node {
            align_self: AlignSelf::FlexStart,
            padding: UiRect::all(px(8)),
            ..default()
        },
        Text::new(text),
        TextFont {
            font: font.clone(),
            font_size,
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgb(0.09, 0.11, 0.12)),
    ));
}
