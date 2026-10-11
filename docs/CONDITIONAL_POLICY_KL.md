# Measuring policy changes

`RecurrentPpoPolicy::pre_tanh_kl_from` measures conditional `KL(previous || current)` in nats for the diagonal Gaussian before tanh and action scaling. It performs two actor forwards and changes neither policy nor caller memory. Its arithmetic uses f64; it is not a bit-exact divergence of the f32 sampled actions.

Supply the same local observation and each policy's own recurrent memory. Advance both memories separately when measuring a trajectory. Critic observations, rewards, action bounds and sampled actions are excluded. Observation and action widths must match; hidden widths may differ when each memory matches its policy.

The method checks profile widths, observation width and finite values, previous memory, then current memory before inference. Existing `DimensionMismatch` and `NonFiniteValue` errors retain their fields and dimensions. Nonfinite inferred Gaussian parameters and divergence beyond finite f64 precision fail explicitly.

The direction follows [PPO equation 4, August 2017](https://arxiv.org/pdf/1707.06347). This diagnostic does not impose a KL bound, change optimizer behavior or qualify a policy. The existing PPO approximate-KL metric is computed before its optimizer step. A one-epoch full-batch update can therefore report near-zero approximate KL while its resulting policy changes substantially.

An isolated 512-transition physical rollout from standing parent `73b1a1096d2a0dd9cac9d6f5ec6b3e7632ed71ef914a03e2d60ee3aa284006e5` measured mean post-update divergence 0.1212286 at actor learning rate 0.0003 and 0.0134404 at 0.0001. Both used the same rollout, bounded posture reward and fresh optimizer. Neither resulting policy passed standing selection. This comparison does not establish that the smaller rate will produce standing.

Standing, walking and fighting remain unfinished. The smaller-rate research trial records post-update mean and maximum divergence; promotion still requires physical selection, visible early-motion review, held-out evaluation and browser execution.

Native tests, strict Clippy, browser tests and personal Rust lint pass. The public recurrent suite passes 25 cases, including nine new KL cases; three private KL tests pass. Measured production coverage hits 139/141 regions and both outcomes of all five explicit conditional branches. Two distribution backend-error propagation regions are not injected; validated public inputs do not produce these backend errors. Existing error sources remain propagated. The [validation record](progress/conditional-policy-kl-validation.json) and [physical update comparison](progress/standing-post-update-kl-diagnostic.json) retain the evidence.
