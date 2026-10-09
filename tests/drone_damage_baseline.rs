//! Measure scheduled damage through the public environment and action boundary.
#![cfg(feature = "robots")]

#[path = "../examples/robots/damage/assessment.rs"]
mod assessment;

use assessment::{assess, FailureTime, Outcome};
use bevy::math::Vec3;
use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneMotorState};
use bevy_gym::Env;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn half_thrust_receives_the_same_post_failure_window_for_both_schedules() {
    let command = DroneAction::try_from([0.5; 4]).expect("half thrust");
    for (schedule, before) in [
        (FailureTime::TwoSeconds, 100),
        (FailureTime::FiveSeconds, 250),
    ] {
        let result = assess(42, DroneMotor::FrontLeft, schedule, |_| Ok(command))
            .expect("valid deterministic commands");
        let Outcome::Crashed(score) = result else {
            panic!("hover must reach the scheduled failure");
        };
        assert_eq!(score.steps, 36);
        assert_eq!(schedule.actions(), before);
        assert_eq!(
            result,
            assess(42, DroneMotor::FrontLeft, schedule, |_| Ok(command)).unwrap()
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn an_episode_that_ends_before_damage_is_reported_separately() {
    let command = DroneAction::try_from([0.0; 4]).expect("power off");
    let result = assess(42, DroneMotor::FrontLeft, FailureTime::TwoSeconds, |_| {
        Ok(command)
    })
    .expect("valid deterministic commands");
    assert!(matches!(result, Outcome::ApproachEnded(_)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn the_controller_observes_only_the_selected_failure_at_the_scheduled_action() {
    for motor in DroneMotor::ALL {
        for schedule in [FailureTime::TwoSeconds, FailureTime::FiveSeconds] {
            let mut commands = 0;
            let result = assess(42, motor, schedule, |observation| {
                for candidate in DroneMotor::ALL {
                    let expected = if commands >= schedule.actions() && candidate == motor {
                        DroneMotorState::Failed
                    } else {
                        DroneMotorState::Working
                    };
                    assert_eq!(observation.motor_state(candidate), expected);
                }
                commands += 1;
                Ok(DroneAction::try_from([0.5; 4])?)
            })
            .unwrap();
            let Outcome::Crashed(score) = result else {
                panic!("constant thrust must crash after failure");
            };
            assert_eq!(commands, schedule.actions() + score.steps);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn balanced_diagonal_thrust_exercises_the_entire_survival_window() {
    let result = assess(
        42,
        DroneMotor::FrontLeft,
        FailureTime::TwoSeconds,
        |observation| {
            // This diagnostic controller permits yaw and uses the remaining diagonal pair.
            let fractions =
                if observation.motor_state(DroneMotor::FrontLeft) == DroneMotorState::Failed {
                    [0.0, 1.0, 0.0, 1.0]
                } else {
                    [0.5; 4]
                };
            Ok(DroneAction::try_from(fractions)?)
        },
    )
    .unwrap();
    let Outcome::Survived(score) = result else {
        panic!("balanced thrust should complete the diagnostic horizon: {result:?}");
    };
    assert_eq!(score.steps, 500);
    assert!(score.reward > 400.0 && score.reward <= 500.0);
    assert!(score.peak_tilt_radians < 0.001);
    assert!(score.final_body_yaw_rate_radians_per_second.abs() > 1.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn controller_errors_stop_either_phase_without_another_command() {
    for failure_call in [0, 100, 101] {
        let mut calls = 0;
        let error = assess(42, DroneMotor::FrontLeft, FailureTime::TwoSeconds, |_| {
            let current = calls;
            calls += 1;
            if current == failure_call {
                return Err(std::io::Error::other("controller failed").into());
            }
            Ok(DroneAction::try_from([0.5; 4])?)
        })
        .unwrap_err();
        assert_eq!(calls, failure_call + 1);
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::Other
        );
        assert_eq!(error.to_string(), "controller failed");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reported_measurements_include_only_the_damage_phase() {
    let command = DroneAction::try_from([0.5; 4]).unwrap();
    let mut drone = DroneHover::default();
    drone.reset(Some(42));
    for _ in 0..100 {
        assert!(!drone.step(command).is_done());
    }
    drone.fail_motor(DroneMotor::FrontLeft).unwrap();
    let mut minimum_height = drone.observation().position().y;
    let mut peak_tilt = 0.0_f32;
    let mut reward = 0.0;
    let mut steps = 0;
    let final_observation = loop {
        let step = drone.step(command);
        steps += 1;
        reward += step.reward;
        minimum_height = minimum_height.min(step.observation.position().y);
        let up = step.observation.orientation() * Vec3::Y;
        peak_tilt = peak_tilt.max(up.y.clamp(-1.0, 1.0).acos());
        if step.is_done() {
            break step.observation;
        }
    };
    let outcome = assess(42, DroneMotor::FrontLeft, FailureTime::TwoSeconds, |_| {
        Ok(command)
    })
    .unwrap();
    let Outcome::Crashed(score) = outcome else {
        panic!("expected a post-failure crash");
    };
    assert_eq!(score.steps, steps);
    // Replaying identical actions must preserve the exact accumulated values.
    assert_eq!(score.reward.to_bits(), reward.to_bits());
    assert_eq!(score.minimum_height_m.to_bits(), minimum_height.to_bits());
    assert_eq!(score.peak_tilt_radians.to_bits(), peak_tilt.to_bits());
    assert_eq!(
        score.final_distance_m.to_bits(),
        final_observation
            .position()
            .distance(Vec3::new(0.0, 2.0, 0.0))
            .to_bits()
    );
    let body_velocity =
        final_observation.orientation().inverse() * final_observation.angular_velocity();
    assert_eq!(
        score.final_body_yaw_rate_radians_per_second.to_bits(),
        body_velocity.y.to_bits()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn serialized_outcomes_and_schedules_name_the_failure_phase() {
    let command = DroneAction::try_from([0.5; 4]).unwrap();
    let outcome = assess(42, DroneMotor::FrontLeft, FailureTime::TwoSeconds, |_| {
        Ok(command)
    })
    .unwrap();
    let json = serde_json::to_value(outcome).unwrap();
    assert_eq!(json["outcome"], "crashed_after_failure");
    assert_eq!(json["steps"], 36);
    assert_eq!(
        serde_json::to_value(FailureTime::TwoSeconds).unwrap(),
        "two_seconds"
    );
    assert_eq!(
        serde_json::to_value(FailureTime::FiveSeconds).unwrap(),
        "five_seconds"
    );
}
