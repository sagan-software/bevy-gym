//! Isolated signed torque fixtures measure the original plant; never autonomous agents.
use bevy_gym::{
    robots::{DroidAction, DroidActuator, DroidBody, DroidStanding},
    Env,
};
/// Report first-action measured angular speeds without training or qualification.
fn main() {
    for seed in [0, 1, 2, 42, u64::MAX] {
        for actuator in DroidActuator::ALL {
            for fraction in [-0.01, 0.01] {
                let mut environment = DroidStanding::default();
                environment.reset(Some(seed));
                let mut fractions = [0.0; 26];
                *fractions
                    .get_mut(actuator as usize)
                    .expect("closed actuator index") = fraction;
                let action = DroidAction::try_from(fractions).expect("bounded isolated fixture");
                let result = environment.step(action);
                let speeds = DroidBody::ALL
                    .map(|body| result.observation.body(body).angular_velocity().length());
                let record = serde_json::json!({"schema":"standing-first-action-response-v1","seed":seed,"actuator":format!("{actuator:?}"),"index":actuator as usize,"torque_fraction":fraction,"angular_speed_rad_s":speeds,"done":result.status.is_done(),"qualification":false});
                println!("{record}");
            }
        }
    }
}
