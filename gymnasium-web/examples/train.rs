//! Train `CartPole` without a browser through the worker's public session API.
use bevy_gym as _;
use bevy_gym_browser::{AdvanceSteps, Session};
use serde as _;
use serde_json as _;
use tokio as _;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::train(42)?;
    let snapshot = session.advance(AdvanceSteps::try_from(32)?)?;
    println!("{snapshot:?}");
    Ok(())
}
