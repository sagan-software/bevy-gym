//! Shared 348-value humanoid state used by the two `MuJoCo` humanoid examples.

use super::SplitMix64;

/// Number of controllable humanoid joints.
pub const HUMANOID_ACTIONS: usize = 17;
/// Default Gymnasium observation width.
pub const HUMANOID_OBSERVATIONS: usize = 348;
/// Per-joint actuator bound.
pub const HUMANOID_ACTION_LIMIT: f32 = 0.4;

/// Motion profile selected by one humanoid environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidMotion {
    /// Rise from a prone pose and remain upright.
    Stand,
    /// Remain healthy while walking forward.
    Walk,
}

/// Keep both shared profiles reachable in each standalone humanoid target.
const _: [HumanoidMotion; 2] = [HumanoidMotion::Stand, HumanoidMotion::Walk];

/// Exact default observation blocks plus compact procedural dynamics.
#[derive(Debug, Clone)]
pub struct HumanoidModel {
    /// Twenty-four qpos values including global x and y.
    pub positions: [f32; 24],
    /// Twenty-three qvel values.
    pub velocities: [f32; 23],
    /// Thirteen body inertia records.
    pub center_inertia: [f32; 130],
    /// Thirteen body center-velocity records.
    pub center_velocity: [f32; 78],
    /// Seventeen actuator constraint forces.
    pub actuator_forces: [f32; 17],
    /// Thirteen clipped external-force records.
    pub contact_forces: [f32; 78],
    /// Observable motion phase.
    pub phase: f32,
}

impl Default for HumanoidModel {
    fn default() -> Self {
        // Preserve the fixed observation and action contract while updating the derived model state.
        let mut positions = [0.0; 24];
        positions[2] = 0.35;
        positions[3] = 1.0;
        Self {
            positions,
            velocities: [0.0; 23],
            center_inertia: [0.0; 130],
            center_velocity: [0.0; 78],
            actuator_forces: [0.0; 17],
            contact_forces: [0.0; 78],
            phase: 0.0,
        }
    }
}

impl HumanoidModel {
    /// Return Gymnasium's concatenated 348-value observation.
    #[must_use]
    pub fn observation(&self) -> Vec<f32> {
        // Preserve the fixed observation and action contract while updating the derived model state.
        let mut observation = Vec::with_capacity(HUMANOID_OBSERVATIONS);
        observation.extend_from_slice(&self.positions[2..]);
        observation.extend_from_slice(&self.velocities);
        observation.extend_from_slice(&self.center_inertia);
        observation.extend_from_slice(&self.center_velocity);
        observation.extend_from_slice(&self.actuator_forces);
        observation.extend_from_slice(&self.contact_forces);
        observation
    }

    /// Reset all observation blocks with Gymnasium's bounded noise profile.
    pub fn reset(&mut self, rng: &mut SplitMix64, motion: HumanoidMotion) {
        // Preserve the fixed observation and action contract while updating the derived model state.
        *self = Self::default();
        self.phase = 0.0;
        self.positions[2] = match motion {
            HumanoidMotion::Stand => 0.35,
            HumanoidMotion::Walk => 1.4,
        };
        for value in &mut self.positions {
            *value += rng.f32_between(-0.01, 0.01);
        }
        for value in &mut self.velocities {
            *value = rng.f32_between(-0.01, 0.01);
        }
        self.update_derived();
    }

    /// Return the motion-specific target for all seventeen joints.
    #[must_use]
    pub fn desired_joints(phase: f32, motion: HumanoidMotion) -> [f32; 17] {
        // Preserve the fixed observation and action contract while updating the derived model state.
        match motion {
            HumanoidMotion::Stand => [
                0.0, 0.0, 0.0, -0.08, 0.0, -0.35, -0.65, 0.08, 0.0, -0.35, -0.65, 0.15, -0.15,
                -0.35, -0.15, 0.15, -0.35,
            ],
            HumanoidMotion::Walk => {
                let opposite = phase + std::f32::consts::PI;
                [
                    0.02 * phase.sin(),
                    0.02 * phase.cos(),
                    0.02 * phase.sin(),
                    -0.08,
                    0.0,
                    0.35 * phase.sin(),
                    0.2_f32.mul_add(-(phase + 0.7).sin(), -0.55),
                    0.08,
                    0.0,
                    0.35 * opposite.sin(),
                    0.2_f32.mul_add(-(opposite + 0.7).sin(), -0.55),
                    0.2 * opposite.sin(),
                    -0.15,
                    -0.35,
                    0.2 * phase.sin(),
                    0.15,
                    -0.35,
                ]
            }
        }
    }

    /// Integrate one action and return horizontal velocity.
    #[must_use]
    pub fn integrate(&mut self, action: [f32; 17], motion: HumanoidMotion) -> f32 {
        // Preserve the fixed observation and action contract while updating the derived model state.
        const STEP: f32 = 0.05;
        for (((action_value, velocity), position), actuator_force) in action
            .iter()
            .zip(&mut self.velocities[6..])
            .zip(&mut self.positions[7..])
            .zip(&mut self.actuator_forces)
        {
            let acceleration = action_value.mul_add(45.0, -5.0 * *velocity);
            *velocity += acceleration * STEP;
            *position = velocity.mul_add(STEP, *position);
            *actuator_force = *action_value * 100.0;
        }
        self.phase = 3.0_f32
            .mul_add(STEP, self.phase)
            .rem_euclid(std::f32::consts::TAU);
        let desired = Self::desired_joints(self.phase, motion);
        let tracking_error = self.positions[7..]
            .iter()
            .zip(desired)
            .map(|(position, target)| (*position - target).powi(2))
            .sum::<f32>();
        let tracking_quality = (-0.35 * tracking_error).exp();
        match motion {
            HumanoidMotion::Stand => {
                let target_height = 1.05_f32.mul_add(tracking_quality, 0.35);
                let previous_height = self.positions[2];
                self.positions[2] = 0.92_f32.mul_add(previous_height, 0.08 * target_height);
                self.velocities[2] = (self.positions[2] - previous_height) / STEP;
                self.velocities[0] *= 0.8;
            }
            HumanoidMotion::Walk => {
                self.positions[2] = 0.04_f32.mul_add(self.phase.cos(), 1.4);
                self.velocities[2] = -0.12 * self.phase.sin();
                let target_velocity = 6.0 * tracking_quality;
                self.velocities[0] = 0.8_f32.mul_add(self.velocities[0], 0.2 * target_velocity);
                self.positions[0] = self.velocities[0].mul_add(STEP, self.positions[0]);
            }
        }
        self.positions[4] = 0.04 * self.phase.sin();
        self.positions[3] = self.positions[4].mul_add(-self.positions[4], 1.0).sqrt();
        self.velocities[3] = 0.12 * self.phase.cos();
        self.update_derived();
        self.velocities[0]
    }

    /// Return a low-energy PD action for the selected motion.
    #[must_use]
    pub fn expert_action(observation: &[f32], motion: HumanoidMotion) -> [f32; 17] {
        // Preserve the fixed observation and action contract while updating the derived model state.
        let phase =
            observation_value(observation, 2).atan2(observation_value(observation, 25) / 3.0);
        let desired = Self::desired_joints(phase, motion);
        std::array::from_fn(|index| {
            let target = desired.get(index).copied().unwrap_or_default();
            let position = observation_value(observation, index + 5);
            let velocity = observation_value(observation, index + 28);
            0.8_f32
                .mul_add(target - position, -0.12 * velocity)
                .clamp(-HUMANOID_ACTION_LIMIT, HUMANOID_ACTION_LIMIT)
        })
    }

    /// Encode target errors while preserving the exact public observation.
    #[must_use]
    pub fn encode_observation(observation: &[f32], motion: HumanoidMotion) -> Vec<f32> {
        // Preserve the fixed observation and action contract while updating the derived model state.
        let phase =
            observation_value(observation, 2).atan2(observation_value(observation, 25) / 3.0);
        let desired = Self::desired_joints(phase, motion);
        let mut encoded = vec![0.0; observation.len()];
        if let Some(value) = encoded.get_mut(2) {
            *value = observation_value(observation, 2);
        }
        if let Some(value) = encoded.get_mut(25) {
            *value = observation_value(observation, 25) / 3.0;
        }
        for (index, target) in desired.into_iter().enumerate() {
            if let Some(value) = encoded.get_mut(index + 5) {
                *value = target - observation_value(observation, index + 5);
            }
            if let Some(value) = encoded.get_mut(index + 28) {
                *value = observation_value(observation, index + 28) / 10.0;
            }
        }
        encoded
    }

    /// Return the squared external-force magnitude.
    #[must_use]
    pub fn contact_force_squared(&self) -> f32 {
        self.contact_forces.iter().map(|value| value * value).sum()
    }

    /// Update deterministic inertia, center-velocity, and contact blocks.
    fn update_derived(&mut self) {
        // Preserve the fixed observation and action contract while updating the derived model state.
        let horizontal_velocity = self.velocities[0];
        let vertical_velocity = self.velocities[2];
        for (body, ((inertia, center_velocity), contact_forces)) in self
            .center_inertia
            .chunks_exact_mut(10)
            .zip(self.center_velocity.chunks_exact_mut(6))
            .zip(self.contact_forces.chunks_exact_mut(6))
            .enumerate()
        {
            if let Some(value) = inertia.first_mut() {
                *value = 0.01 * (body + 1) as f32;
            }
            if let Some(value) = inertia.last_mut() {
                *value = 0.1_f32.mul_add(body as f32, 1.0);
            }
            if let Some(value) = center_velocity.first_mut() {
                *value = horizontal_velocity;
            }
            if let Some(value) = center_velocity.get_mut(2) {
                *value = vertical_velocity;
            }
            if let Some(value) = contact_forces.get_mut(2) {
                *value = if body == 5 || body == 8 { 0.2 } else { 0.0 };
            }
        }
    }
}

/// Read one observation value, using zero for an undersized external slice.
fn observation_value(observation: &[f32], index: usize) -> f32 {
    observation.get(index).copied().unwrap_or_default()
}
