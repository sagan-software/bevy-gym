//! Public `MuJoCo` simulation boundary tests.

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use bevy_gym as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap as _;
#[cfg(feature = "render")]
use serde as _;
use serde_json as _;
use tokio as _;

#[cfg(feature = "mujoco")]
mod enabled {
    use bevy_gym::mujoco::MujocoSimulation;
    use mujoco_rs as _;

    const ONE_JOINT_MODEL: &str = r#"
<mujoco model="one joint">
  <option timestep="0.01"/>
  <worldbody>
    <body>
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"#;

    #[test]
    fn simulation_exposes_exact_state_and_steps_multiple_frames() {
        // Translate native MuJoCo state without exposing unchecked wrapper internals to callers.
        let mut simulation =
            MujocoSimulation::from_xml_string(ONE_JOINT_MODEL).expect("one-joint model loads");
        simulation
            .set_state(&[0.25], &[0.0])
            .expect("matching state dimensions are accepted");

        simulation
            .step(&[1.0], 4)
            .expect("matching control dimensions are accepted");

        assert_eq!(simulation.qpos().len(), 1);
        assert_eq!(simulation.qvel().len(), 1);
        assert!(simulation.qpos().first().copied().unwrap_or_default() > 0.25);
        assert!((simulation.time() - 0.04).abs() < f64::EPSILON);
    }

    #[test]
    fn state_and_control_dimension_mismatches_are_rejected() {
        let mut simulation =
            MujocoSimulation::from_xml_string(ONE_JOINT_MODEL).expect("one-joint model loads");

        assert!(simulation.set_state(&[], &[0.0]).is_err());
        assert!(simulation.step(&[], 1).is_err());
        assert!(simulation.step(&[0.0], 0).is_err());
    }
}
