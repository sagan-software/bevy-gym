//! Frozen selected-candidate evaluation; no optimization or promotion.
#![expect(dead_code,unused_imports,reason="Shared lesson modules retain their independent CLI and training paths.")]
#[path="/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/learning/mod.rs"]
#[expect(clippy::type_complexity,reason="The imported collector retains its owning repository's allowed function-pointer contract.")]
mod learning;
#[path="/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/standing/mod.rs"]
mod standing;
use bevy_gym::{Env,EpisodeStatus,robots::{DroidStanding,DroidBody},training::{RecurrentPpoPolicy,RecurrentPpoConfig}};
use sha2::{Digest,Sha256};
use std::{error::Error,path::Path};
/// Run the one frozen selected checkpoint without training or fallback.
fn main()->Result<(),Box<dyn Error>>{tokio::runtime::Builder::new_current_thread().build()?.block_on(run())}
/// Validate bytes before evaluating the predeclared roots and emitting physical motion evidence.
async fn run()->Result<(),Box<dyn Error>>{
 let source=Path::new("/home/sagan/.cache/bevy-gym-quality-validation/standing-exploratory-continuation-probe-20261011/run/standing-8720.mpk");
 let output=Path::new("/home/sagan/.cache/bevy-gym-quality-validation/standing-seed179-update8720-evaluation-20261011/result");
 let bytes=tokio::fs::read(source).await?;let digest=Sha256::digest(&bytes);
 if format!("{digest:x}")!="bbe86ac6ebb4f7df65168c0a3850d7f4cbb7d45f1e7e2bfa05fcefb36a26633e" {return Err("selected checkpoint hash mismatch".into());}
 let config=RecurrentPpoConfig::default();
 let policy=RecurrentPpoPolicy::load_bytes(bytes.clone(),204,204,1,&[-1.0;26],&[1.0;26],&config)?;
 let selection=&standing::evaluation::SELECTION_SEEDS;
 let selected=standing::evaluation::evaluate(&policy,selection)?;let selected_control=standing::control::evaluate(&policy,selection)?;
 if !(standing::evaluation::passes(&selected,selection)&&standing::control::passes(&selected_control,selection)){return Err("frozen selection did not reproduce".into());}
 let held=standing::evaluation::held_out_seeds();
 let episodes=standing::evaluation::evaluate(&policy,&held)?;let control=standing::control::evaluate(&policy,&held)?;
 let passed=standing::evaluation::passes(&episodes,&held)&&standing::control::passes(&control,&held);
 let report=serde_json::json!({"schema":"standing-selected-frozen-evaluation-v1","checkpoint_sha256":format!("{digest:x}"),"selection":selected,"selection_control":selected_control,"held_out":episodes,"held_out_control":control,"passed":passed,"qualification":"pending full early-motion and browser inspection; no optimization or held-out ranking"});
 assert_eq!(bytes,policy.to_bytes()?);
 tokio::fs::create_dir(output).await?;
 tokio::fs::write(output.join("evaluation.json"),serde_json::to_vec_pretty(&report)?).await?;
 for seed in *selection {tokio::fs::write(output.join(format!("trace-{seed}.jsonl")),trace(&policy,seed)?).await?;}
 println!("{report}");Ok(())
}
/// Retain all thirteen physical segments and policy-selected torques from reset through termination.
fn trace(policy:&RecurrentPpoPolicy,seed:u64)->Result<String,Box<dyn Error>>{
 let mut env=DroidStanding::default();let mut observation=env.reset(Some(seed)).observation;let mut memory=policy.initial_memory();let mut output=String::new();
 for step in 1..=1000 {
  let inferred=policy.mean_action(&standing::encoding::encode(&observation),&memory)?;memory=inferred.next_memory;
  let action=standing::encoding::decode(&inferred.action)?;let result=env.step(action);observation=result.observation;
  let segments=DroidBody::ALL.map(|body|{let b=observation.body(body);serde_json::json!({"position":b.position().to_array(),"orientation":b.orientation().to_array(),"angular_velocity":b.angular_velocity().to_array(),"linear_velocity":b.linear_velocity().to_array(),"floor_contact":b.floor_contact()})});
  let row=serde_json::json!({"seed":seed,"step":step,"nominal_time_ms":step*20,"torques":inferred.action,"segments":segments,"done":result.status.is_done()});output.push_str(&serde_json::to_string(&row)?);output.push('\n');
  if result.status.is_done(){break;}
 }
 Ok(output)
}
