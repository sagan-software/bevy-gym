//! Animation graph and bounded presentation state for one mannequin.

use super::Game;
use bevy::prelude::*;

/// Named graph nodes prevent clip-order changes from silently selecting another action.
#[derive(Component)]
pub(crate) struct Rig {
    /// Unarmed standing loop, also used for armed lower-body standing.
    idle: AnimationNodeIndex,
    /// In-place forward locomotion, measured against actual capsule displacement.
    run: AnimationNodeIndex,
    /// Authored upper-body weapon stance.
    armed: AnimationNodeIndex,
    /// Authored high aim pose, masked to the upper body.
    aim_up: AnimationNodeIndex,
    /// Authored low aim pose, masked to the upper body.
    aim_down: AnimationNodeIndex,
    /// Authored recoil played when ammunition decreases.
    shoot: AnimationNodeIndex,
    /// Previous displayed magazine count detects newly accepted shots.
    rounds: u8,
    /// Smoothed running blend weight, bounded to zero through one.
    running: f32,
    /// Previous life state stops animation once when physics takes ownership.
    life: Life,
}

/// Animation owns living bones; ragdoll physics owns dead bones.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Life {
    /// Locomotion and weapon overlays may play.
    Alive,
    /// Physics owns the skeleton; animation tracks remain stopped.
    Dead,
}

impl Rig {
    /// Resolve validated bundled clip names and reserve independent upper-body masking.
    pub(super) fn new(model: &Gltf) -> (AnimationGraph, Self) {
        let mut graph = AnimationGraph::new();
        let mut clip = |name: &str, mask| {
            graph.add_clip_with_mask(
                model
                    .named_animations
                    .get(name)
                    .expect("Validated mannequin clip")
                    .clone(),
                mask,
                1.0,
                graph.root,
            )
        };
        let rig = Self {
            idle: clip("Idle_Loop", 0),
            run: clip("Jog_Fwd_Loop", 0),
            armed: clip("Pistol_Aim_Neutral", 2),
            shoot: clip("Pistol_Shoot", 2),
            aim_up: clip("Pistol_Aim_Up", 2),
            aim_down: clip("Pistol_Aim_Down", 2),
            rounds: 0,
            running: 0.0,
            life: Life::Alive,
        };
        (graph, rig)
    }

    /// Death releases the skeleton to physics; reset restores living animation loops.
    pub(super) fn advance(
        &mut self,
        game: &Game,
        seconds: f32,
        player: &mut AnimationPlayer,
        graph: &mut AnimationGraph,
    ) {
        let life = if game.robot_health.is_alive() {
            Life::Alive
        } else {
            Life::Dead
        };
        if life != self.life {
            player.stop_all();
            self.life = life;
            self.running = 0.0;
        }
        if life == Life::Alive {
            self.locomotion(game, seconds, player, graph);
            self.weapon(game, player);
        }
        self.rounds = game.combat.rounds();
    }

    /// Blend over 150 milliseconds and stop running when collision prevents movement.
    fn locomotion(
        &mut self,
        game: &Game,
        seconds: f32,
        player: &mut AnimationPlayer,
        graph: &mut AnimationGraph,
    ) {
        let wanted = if game.motion.length() > 0.001 {
            1.0
        } else {
            0.0
        };
        self.running =
            (wanted - self.running).mul_add((seconds / 0.15).clamp(0.0, 1.0), self.running);
        let mask = u64::from(game.combat.is_armed());
        for node in [self.idle, self.run] {
            graph.get_mut(node).expect("Owned graph node").mask = mask;
        }
        player
            .play(self.idle)
            .repeat()
            .set_weight(1.0 - self.running);
        let stride = super::locomotion::Stride::new(game.motion, game.facing());
        player
            .play(self.run)
            .repeat()
            .set_weight(self.running)
            .set_speed(stride.speed);
    }

    /// Play recoil only for a newly spent round; dry fire cannot restart it.
    fn weapon(&self, game: &Game, player: &mut AnimationPlayer) {
        if !game.combat.is_armed() {
            player.stop(self.armed);
            player.stop(self.shoot);
            player.stop(self.aim_up);
            player.stop(self.aim_down);
            return;
        }
        if game.combat.rounds() < self.rounds {
            player.play(self.shoot).replay();
        }
        let shooting = player
            .animation(self.shoot)
            .is_some_and(|clip| !clip.is_finished());
        let pitch = game.aim.map_or(0.0, |aim| aim.y.clamp(-1.0, 1.0).asin());
        let elevation = (pitch / std::f32::consts::FRAC_PI_3).clamp(-1.0, 1.0);
        let neutral = 1.0 - elevation.abs();
        player
            .play(self.armed)
            .repeat()
            .set_weight(if shooting { 0.0 } else { neutral });
        player.play(self.aim_up).set_weight(elevation.max(0.0));
        player.play(self.aim_down).set_weight((-elevation).max(0.0));
        if shooting {
            player
                .animation_mut(self.shoot)
                .expect("Playing recoil")
                .set_weight(neutral);
        }
        if !shooting {
            player.stop(self.shoot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{firing::Action, Movement};

    /// Construct named clip handles without requiring a GPU or asynchronous asset loading.
    fn model() -> Gltf {
        Gltf {
            scenes: default(),
            named_scenes: default(),
            meshes: default(),
            named_meshes: default(),
            materials: default(),
            named_materials: default(),
            nodes: default(),
            named_nodes: default(),
            skins: default(),
            named_skins: default(),
            default_scene: None,
            animations: default(),
            named_animations: [
                "Idle_Loop",
                "Jog_Fwd_Loop",
                "Pistol_Aim_Neutral",
                "Pistol_Shoot",
                "Pistol_Aim_Up",
                "Pistol_Aim_Down",
                "Death01",
            ]
            .into_iter()
            .map(|name| (name.into(), Handle::default()))
            .collect(),
            source: None,
        }
    }

    #[test]
    fn measured_motion_blends_and_weapon_masks_only_the_upper_body() {
        let (mut graph, mut rig) = Rig::new(&model());
        let mut player = AnimationPlayer::default();
        let mut game = Game::default();
        rig.advance(&game, 0.15, &mut player, &mut graph);
        assert_eq!(player.animation(rig.idle).expect("Idle").weight(), 1.0);
        assert_eq!(player.animation(rig.run).expect("Run").weight(), 0.0);
        game.step(Movement::Right);
        rig.advance(&game, 0.075, &mut player, &mut graph);
        assert!((rig.running - 0.5).abs() < 1e-6);
        game.motion = Vec3::ZERO;
        rig.advance(&game, 0.15, &mut player, &mut graph);
        assert_eq!(rig.running, 0.0);
        assert_eq!(graph.get(rig.idle).expect("Idle node").mask, 0);
        game.act(Action::PickUp);
        rig.advance(&game, 0.15, &mut player, &mut graph);
        assert_eq!(graph.get(rig.idle).expect("Idle node").mask, 1);
        assert_eq!(graph.get(rig.run).expect("Run node").mask, 1);
        assert_eq!(graph.get(rig.armed).expect("Armed node").mask, 2);
        assert!(player.is_playing_animation(rig.armed));
    }

    #[test]
    fn aiming_blends_neutral_and_elevation_without_moving_the_legs() {
        let (mut graph, mut rig) = Rig::new(&model());
        let mut player = AnimationPlayer::default();
        let mut game = Game::default();
        game.act(Action::PickUp);
        for (aim, neutral, up, down) in [
            (None, 1.0, 0.0, 0.0),
            (Some(Dir3::Y), 0.0, 1.0, 0.0),
            (Some(Dir3::NEG_Y), 0.0, 0.0, 1.0),
        ] {
            game.aim = aim;
            rig.advance(&game, 0.02, &mut player, &mut graph);
            for (node, weight) in [(rig.armed, neutral), (rig.aim_up, up), (rig.aim_down, down)] {
                assert_eq!(player.animation(node).expect("Aim pose").weight(), weight);
                assert_eq!(graph.get(node).expect("Aim node").mask, 2);
            }
            assert_eq!(player.animation(rig.run).expect("Run").weight(), 0.0);
        }
    }

    #[test]
    fn accepted_shots_replay_recoil_but_rejected_shots_do_not() {
        let (mut graph, mut rig) = Rig::new(&model());
        let mut player = AnimationPlayer::default();
        let mut game = Game::default();
        game.act(Action::PickUp);
        rig.advance(&game, 0.02, &mut player, &mut graph);
        game.act(Action::Fire(Dir3::NEG_Z));
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(player.is_playing_animation(rig.shoot));
        player
            .animation_mut(rig.shoot)
            .expect("Recoil")
            .seek_to(0.1);
        game.act(Action::Fire(Dir3::NEG_Z));
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert_eq!(
            player.animation(rig.shoot).expect("Recoil").seek_time(),
            0.1
        );
        // Zero remaining repetitions creates a completed clip through Bevy's public API.
        player
            .animation_mut(rig.shoot)
            .expect("Recoil")
            .set_repeat(bevy::animation::RepeatAnimation::Count(0));
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(!player.is_playing_animation(rig.shoot));
        assert_eq!(player.animation(rig.armed).expect("Aim").weight(), 1.0);
        game.reset();
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(!player.is_playing_animation(rig.shoot));
        assert!(!player.is_playing_animation(rig.armed));
    }

    #[test]
    fn death_stops_animation_and_reset_restores_living_loops() {
        let (mut graph, mut rig) = Rig::new(&model());
        let mut player = AnimationPlayer::default();
        let mut game = Game::default();
        for _ in 0..1000 {
            game.step(Movement::Idle);
            if !game.robot_health.is_alive() {
                break;
            }
        }
        assert!(!game.robot_health.is_alive());
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(player.playing_animations().next().is_none());
        assert!(!player.is_playing_animation(rig.idle));
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(player.playing_animations().next().is_none());
        game.reset();
        rig.advance(&game, 0.02, &mut player, &mut graph);
        assert!(player.is_playing_animation(rig.idle));
    }
}
