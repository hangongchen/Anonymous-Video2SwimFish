# Reactive natural-swimming fish — AMP tank-swim pipeline (design spec)

Goal: a reference-free, closed-loop policy that swims **forever** inside a square water tank,
**reacts to the four walls** (turns to avoid / recover from them), and whose **gait AND turns are
stylistically indistinguishable from real zebrafish** (learned from real tank-swimming video).
Straight cruising and turning are one policy; turning is induced by the walls (the "situation") and
shaped to look real by an Adversarial Motion Prior (AMP). No goal/waypoint — the walls are the task.

## Environment (v1)
- Square water tank, **four walls with real collision** (static rigid planes/boxes).
- Zero-gravity neutrally-buoyant water + analytic MuJoCo hydro (unchanged from the swim env).
- Cooked FEM fish asset (`fish_articulated_cooked.usd`, 165-node sim mesh), all stability cures
  (dt 1/960, elasticityDamping, vertex-velocity damping, 96 solver iters).
- **Bone colliders RE-ENABLED** for wall contact (they are disabled in the swim/IL env); FEM
  self-collision stays OFF. RISK: hard wall contact injects impulsive energy -> can trigger the FEM
  CFL blow-up. De-risk FIRST: tank + collision on the cooked fish must run stable before AMP/data.

## Perception (egocentric, reference-free observation)
- Wall-distance rays (lateral-line / vision) -> distance to each wall around the body (lets the
  policy anticipate and turn BEFORE hitting, so turns look real; collision is the physical backstop).
- Proprioception: body up-vector, root lin/ang velocity (body frame), joint pos/vel, last action.
- NO reference index, NO phase, NO world-frame demo target.

## Reference motion dataset (must include turns)
- From ZeF gt.txt, detect turning bouts (high head-heading curvature) across sequences; carve those
  frames (segment -> two-view -> voxel -> cloud) + straight bouts.
- AMP motion feature per step: `(kappa(s), kappa_dot(s), body yaw-rate, forward speed)` where
  kappa(s) is the spine curvature profile (reuse centerline_num_bins). yaw-rate+speed put TURN
  dynamics into the style, not just static body bend.

## Reward
`r = w_style*r_style + w_task*r_task - w_energy*P - (stability penalties)`
- **r_style (AMP)**: discriminator on (feature_t, feature_t+1) vs the real turn+cruise dataset
  (AMP least-squares style reward). Natural gait AND natural turns.
- **r_task**: alive bonus + keep-moving (forward speed so it doesn't hug a corner); wall-contact
  penalty so it learns to turn away rather than scrape.
- **ENERGY PENALTY** `P = sum_j |tau_j * omega_j|` (mechanical power; applied joint torque * joint
  velocity). r_energy = -w_energy * P. Encourages efficient natural gaits; also damps violent
  actuation -> helps FEM stability. (Normalize by forward speed for a true cost-of-transport.)
- **Stability**: bounded joint velocity, bounded body roll/pitch, action-rate smoothness.

## Episode
- Long horizon; reset ONLY on failure (blow-up guard / NaN). Tank walls keep it bounded (no
  "left the arena" reset needed). RSI from random real gait phases (incl. turn phases) for exploration.
- No reference-length cap, no mismatch termination.

## Algorithm + training procedure
- The installed rl_games has NO AMP support (PPO only). So the AMP **discriminator lives INSIDE the
  env**: the env owns a small MLP + optimizer + a replay buffer of its own recent motion-feature
  transitions; every K reward-steps it does a few LSGAN+grad-penalty SGD steps (policy transitions =
  fake, reference transitions = real) and the per-step style reward is read from that discriminator.
  Training is then plain rl_games PPO (no internals touched). Defensive: modest style weight + NaN
  guards so task/energy/stability terms keep the fish swimming even if the discriminator wobbles.
- Real vs sim motion features MUST be extracted identically (same spine_bend_profile code on the FEM
  nodes as on the carved clouds) or the discriminator learns the extraction gap, not the motion.
- PPO (rl_games) + the env-owned discriminator + replay buffer (policy transitions vs real dataset).
- Reward curriculum INSIDE the full pipeline (not reduced scope): (a) AMP + keep-moving -> natural
  cruise; (b) walls active -> reactive avoidance turns; (c) tune energy/stability weights.

## Evaluation
- Gait & turn realism: discriminator ~0.5; Strouhal 0.2-0.4; distributions of yaw-rate / turn-radius
  / peak body-curvature-during-turn vs real ZeF turns.
- Reactivity: wall-collision rate, anticipatory turn behavior, coverage of the tank.
- Runs-forever/stability: K long rollouts (~20k steps) -> survival fraction (no blow-up, bounded
  jvel), speed non-decay, limit-cycle check on straight segments.
- Efficiency: cost-of-transport (uses the energy term P / (weight*speed)).
- Robustness: perturbations, different tank sizes.

## Build order
1. Tank env: square tank + 4 collidable walls, reuse cooked FEM asset, bone colliders re-enabled,
   verify wall collision + FEM stability. (THIS STEP FIRST — the riskiest.)
2. Reference turn+cruise dataset (turn detection + extended carve + (kappa,kappa_dot,yaw,speed)).
3. AMP discriminator + style reward + energy penalty + task reward in the rl_games PPO loop.
4. Eval script (naturalness incl. turns, reactivity, forever, efficiency).
