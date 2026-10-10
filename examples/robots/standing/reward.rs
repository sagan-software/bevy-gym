//! Training-only posture weighting; frozen standing evaluation uses the original environment.

use bevy::math::Vec3;
use bevy_gym::{
    robots::{DroidAction, DroidBody, DroidObservation, DroidStanding},
    Env, Reset, Step,
};
#[cfg(not(target_arch = "wasm32"))]
use clap::ValueEnum;
use serde::Serialize;

use super::evaluation::{MAX_HEIGHT_M, MIN_HEIGHT_M};

/// Closed training recipes; evaluation always uses the original standing task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(ValueEnum))]
pub(crate) enum Recipe {
    /// Preserve the original reward and default artifact set.
    #[default]
    Original,
    /// Posture and foot-support weighting, recorded before collecting transitions.
    #[serde(rename = "droid-standing-posture-reward-v1")]
    PostureV1,
    /// Posture, contact and pelvis-height weighting for a separate training experiment.
    #[serde(rename = "droid-standing-posture-height-reward-v1")]
    PostureHeightV1,
}

impl Recipe {
    /// Select an episode constructor without inspecting or selecting an action.
    pub(crate) fn factory(self) -> fn() -> TrainingTask {
        match self {
            Self::Original => TrainingTask::default,
            Self::PostureV1 => TrainingTask::posture,
            Self::PostureHeightV1 => TrainingTask::posture_height,
        }
    }
}

/// Dimensionless up-projection deviation at the original f32 15-degree gate.
pub(crate) const POSTURE_SCALE: f64 = 1.0 - 0.965_925_8_f32 as f64;
/// Pelvis-height target in metres, derived from the original accepted f32 bounds.
pub(crate) const HEIGHT_TARGET_M: f64 = (MIN_HEIGHT_M as f64).midpoint(MAX_HEIGHT_M as f64);
/// Height-error scale in metres, equal to half the original accepted height span.
pub(crate) const HEIGHT_SCALE_M: f64 = (MAX_HEIGHT_M as f64 - MIN_HEIGHT_M as f64) / 2.0;

/// One authoritative physical episode with a training reward projection.
pub(crate) struct TrainingTask {
    /// Body, contacts, reset stream and lifecycle remain owned by the original task.
    environment: DroidStanding,
    /// The selected reward projection is fixed throughout each physical episode.
    recipe: Recipe,
}

impl Default for TrainingTask {
    fn default() -> Self {
        Self {
            environment: DroidStanding::default(),
            recipe: Recipe::Original,
        }
    }
}

impl TrainingTask {
    /// Create the original body without commanding or adjusting any actuator.
    pub(crate) fn posture() -> Self {
        Self {
            environment: DroidStanding::default(),
            recipe: Recipe::PostureV1,
        }
    }

    /// Create the same physical body with the posture-height training projection.
    pub(crate) fn posture_height() -> Self {
        Self {
            environment: DroidStanding::default(),
            recipe: Recipe::PostureHeightV1,
        }
    }
}

impl Env for TrainingTask {
    type Observation = DroidObservation;
    type Action = DroidAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.environment.reset(seed)
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        // Forward the policy's validated torques before measuring the resulting body.
        let mut step = self.environment.step(action);
        if self.recipe == Recipe::Original || step.status.is_done() {
            return step;
        }
        let observation = step.observation;
        let upright = (observation.body(DroidBody::Torso).orientation() * Vec3::Y)
            .y
            .clamp(-1.0, 1.0);
        step.reward *= multiplier(
            upright,
            observation.body(DroidBody::LeftFoot).floor_contact(),
            observation.body(DroidBody::RightFoot).floor_contact(),
        );
        // If height weighting is selected, measure the same authoritative post-action pelvis.
        if self.recipe == Recipe::PostureHeightV1 {
            step.reward *= height_multiplier(observation.body(DroidBody::Pelvis).position().y);
        }
        step
    }
}

/// Multiply dimensionless posture and contact weights, retaining signal before landing.
///
/// The dimensionless denominator is `1 - 0.9659258`, using the original f32
/// torso-up gate. At that gate the posture weight is exp(-1); upright is 1.
/// Inverted poses receive a smaller weight, avoiding the upright/inverted ambiguity
/// of squared projected-gravity XY in the upstream orientation penalty:
/// <https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py>.
/// This local experiment does not claim upstream equivalence or learned competence.
fn multiplier(upright: f32, left: bool, right: bool) -> f64 {
    let posture = (-(1.0 - f64::from(upright)) / POSTURE_SCALE).exp();
    let support = match (left, right) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.5,
        (false, false) => 0.25,
    };
    posture * support
}

/// Weight pelvis height with a dimensionless Gaussian around the fixed target.
///
/// Height, target and scale are metres; dividing their difference by the scale
/// produces a dimensionless error. The Gaussian is a local bounded projection.
/// Isaac Lab v3.0.0-EA uses squared world-frame height error on flat terrain:
/// <https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab/isaaclab/envs/mdp/rewards.py#L102>.
/// This experiment measures pelvis Y above the zero-height floor; upstream measures root Z.
fn height_multiplier(pelvis_height_m: f32) -> f64 {
    let deviation_m = f64::from(pelvis_height_m) - HEIGHT_TARGET_M;
    let normalized_error = deviation_m / HEIGHT_SCALE_M;
    (-normalized_error * normalized_error).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed kernel peaks at the target and gives exp(-1) at both frozen height bounds.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn height_kernel_has_fixed_target_and_boundary_weights() {
        assert_eq!(HEIGHT_TARGET_M.to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            HEIGHT_SCALE_M.to_bits(),
            0.149_999_976_158_142_1_f64.to_bits()
        );
        assert_eq!(height_multiplier(1.0).to_bits(), 1.0_f64.to_bits());
        let boundary_weight = 0.367_879_441_171_442_33;
        for height_m in [MIN_HEIGHT_M, MAX_HEIGHT_M] {
            assert!((height_multiplier(height_m) - boundary_weight).abs() < 1e-15);
        }
        for height_m in [0.5, 0.7, 0.8, 1.2, 1.3, 2.0] {
            assert!((0.0..boundary_weight).contains(&height_multiplier(height_m)));
        }
        assert!(height_multiplier(0.84) < height_multiplier(MIN_HEIGHT_M));
        assert!(height_multiplier(MIN_HEIGHT_M) < height_multiplier(0.9));
        assert!(height_multiplier(0.9) < height_multiplier(1.0));
    }

    /// Known posture weights distinguish upright, threshold, horizontal and inverted poses.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn posture_and_each_contact_combination_have_bounded_weights() {
        for (left, right, expected) in [
            (true, true, 1.0_f64),
            (true, false, 0.5),
            (false, true, 0.5),
            (false, false, 0.25),
        ] {
            assert_eq!(multiplier(1.0, left, right).to_bits(), expected.to_bits());
        }
        assert!((multiplier(0.965_925_8, true, true) - 0.367_879_441_171_442_33).abs() < 1e-15);
        let horizontal = multiplier(0.0, true, true);
        let inverted = multiplier(-1.0, true, true);
        assert!((0.0..1e-12).contains(&horizontal));
        assert!(inverted > 0.0 && inverted < horizontal);
    }
}
