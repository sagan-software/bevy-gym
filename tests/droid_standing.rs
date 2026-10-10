//! Articulated standing mechanics through the public environment boundary.
#![cfg(feature = "robots")]

use bevy_gym::robots::{DroidAction, DroidBody, DroidStanding};
use bevy_gym::Env;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// The same seed and torque commands reproduce all physical body states.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn seeded_articulated_steps_replay() {
    let mut first = DroidStanding::default();
    let mut second = DroidStanding::default();
    let initial = first.reset(Some(42)).observation;
    assert_eq!(initial, second.reset(Some(42)).observation);
    assert!(initial.body(DroidBody::Head).position().y > 1.5);
    for fractions in [[0.0; 26], [0.1; 26], [-0.1; 26]] {
        let action = DroidAction::try_from(fractions).expect("bounded test fixture");
        assert_eq!(first.step(action), second.step(action));
    }
}

/// Every actuator independently rejects non-finite and out-of-range inputs.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn actions_validate_all_axes_before_physics() {
    use bevy_gym::robots::{DroidActuator, InvalidDroidAction};
    for actuator in DroidActuator::ALL {
        for value in [-1.0, -0.0, 0.0, 1.0] {
            let mut fractions = [0.0; 26];
            *fractions.get_mut(actuator as usize).expect("closed index") = value;
            let action = DroidAction::try_from(fractions).expect("inclusive boundary");
            assert_eq!(action.fraction(actuator).to_bits(), value.to_bits());
        }
        for value in [
            -1.000_000_1,
            1.000_000_1,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut fractions = [0.0; 26];
            *fractions.get_mut(actuator as usize).expect("closed index") = value;
            assert_eq!(DroidAction::try_from(fractions), Err(InvalidDroidAction));
        }
    }
    assert_eq!(
        InvalidDroidAction.to_string(),
        "droid torque commands must be 26 finite fractions between -1 and 1"
    );
    assert!(std::error::Error::source(&InvalidDroidAction).is_none());
}

/// Unpowered joints fall physically, then subsequent commands cannot move the terminal body.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn passive_fall_freezes_until_reset() {
    use bevy_gym::EpisodeStatus;
    let mut environment = DroidStanding::default();
    let initial = environment.reset(Some(42)).observation;
    let passive = DroidAction::try_from([0.0; 26]).expect("passive test fixture");
    let mut last = environment.step(passive);
    assert_eq!(last.status, EpisodeStatus::Continuing);
    assert!((0.0..=1.0).contains(&last.reward));
    assert!(last.observation.body(DroidBody::Pelvis).linear_velocity().y < -0.1);
    let mut foot_contact = false;
    for _ in 1..1_000 {
        foot_contact |= last.observation.body(DroidBody::LeftFoot).floor_contact()
            || last.observation.body(DroidBody::RightFoot).floor_contact();
        if last.status == EpisodeStatus::Terminated {
            break;
        }
        last = environment.step(passive);
    }
    assert!(foot_contact);
    assert_eq!(last.status, EpisodeStatus::Terminated);
    assert_eq!(last.reward.to_bits(), 0.0_f64.to_bits());
    for value in [-1.0, 0.0, 1.0] {
        assert_eq!(
            environment.step(DroidAction::try_from([value; 26]).expect("fixture")),
            last
        );
    }
    assert_eq!(environment.reset(Some(42)).observation, initial);
    assert_ne!(environment.reset(None).observation, initial);
    assert!(format!("{environment:?}").contains("Standing"));
}

/// Torque commands change the articulated state while seeded resets clear that change.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn torques_actuate_every_axis() {
    use bevy_gym::robots::DroidActuator;
    let mut environment = DroidStanding::default();
    let passive = DroidAction::try_from([0.0; 26]).expect("fixture");
    for actuator in DroidActuator::ALL {
        environment.reset(Some(42));
        let baseline = environment.step(passive).observation;
        environment.reset(Some(42));
        let mut fractions = [0.0; 26];
        *fractions.get_mut(actuator as usize).expect("closed index") = 0.1;
        let actual = environment.step(DroidAction::try_from(fractions).expect("fixture"));
        assert_ne!(actual.observation, baseline, "{actuator:?}");
        for identity in DroidBody::ALL {
            let body = actual.observation.body(identity);
            assert!(body.orientation().is_finite());
            assert!(body.angular_velocity().is_finite());
            assert!(body.position().is_finite());
            assert!(body.linear_velocity().is_finite());
        }
    }
}
