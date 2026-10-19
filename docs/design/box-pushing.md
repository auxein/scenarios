# Cooperative box-pushing: design

**Status:** Draft for review · **Date:** 2026-10-20 · **Auxein:** `v0.3.3` · **Milestone:** R1 (two verbs, `train` and `deploy`; brains as safetensors; a smaller config)

---

## 1. Purpose and story

**A team of agents, whose shared brain is a small neural network evolved by Auxein, learns to push a box too heavy for any one of them to a target, in an arena with obstacles.** The scenario exists to show three things, in this order:

1. **A clueless start.** The first generation's teams wander, spin and bump into walls.
2. **Cooperation emerging.** Over generations, teams find the box, gather on one side of it and push it together, because nothing less moves it.
3. **Deployment to unseen situations.** The evolved team, unchanged, delivers the box in layouts it has never seen.

**The outputs are videos and simple charts**, for a website and for talks: clips of the champion at chosen generations, a montage of them, and fitness over generations (§7). **Training runs on a laptop** (Apple Silicon M4) in hours, not days, which bounds every size in this document (§10.3).

**Box-pushing specifics stay in this repository.** Anything generic that turns out to be missing from Auxein is described as a request to Auxein, separately, and never built here on the side (`CLAUDE.md`, "Working on Auxein from this repository").

**Numbers in this document are starting defaults**, all configurable and expected to be tuned once the world runs (M1) and the first runs are in (M2). Details this design needed beyond the decisions it was first given were marked *(proposed)* and listed in §10.6; most were decided at M1a, and the ones still open (fitness details for M2, CMA-ES for M4) keep the marker.

---

## 2. The world

### 2.1 Why its own physics

**The world is its own vectorised 2-D physics, wrapped as an Auxein `BatchedEnvironment`** (M1b) around a pure physics core (M1a, §2.10). One `run_batch` call simulates every candidate of a batch on every scenario of the set, and every agent of every team, at once: arrays shaped `(n, s, ...)` (candidates, scenarios) and `(n, s, N, ...)` (agents), with **no Python loops over candidates, scenarios or agents**; the only Python loop is over time steps.

- **Why not an existing engine.** A general physics engine (Box2D, pymunk, MuJoCo) steps one world at a time from Python, so a generation of 100 candidates × 32 scenarios would be 3,200 worlds stepped in a loop. Batched GPU engines exist but bring their own array library, while Auxein's numeric layer is the Array API standard on numpy and torch (core design §7). A small, purpose-built model written against the same namespace runs where the genomes already are, with no host round trip (the pattern of Auxein's team fixture, `tests/support/team.py`).
- **Array API, as Auxein.** The code uses the namespace of its inputs (`array-api-compat`): numpy or torch, float64 or float32, CPU, CUDA or Metal. It follows Auxein's rules for numeric code (core design §7.2): no in-place mutation (so updates are `xp.where`, never masked assignment), no global random state, arrays stay on their device, and everything works in float32, the only precision Metal has.
- **Deterministic.** For a given scenario seed, backend, precision and device, an episode gives the same numbers every time. Contacts and sensing are written as **dense pairwise arrays reduced with sums**, never as scatters (`index_add`), whose order on a GPU is not fixed. The scope is Auxein's (core design §8): float32 results are not portable between CPU architectures or between a CPU and a GPU.
- **Integration.** A fixed time step `dt` and semi-implicit (symplectic) Euler: velocities are updated from forces first, then positions from the new velocities. It is the cheapest integrator that is stable for the stiff springs of penalty contacts (§2.6). The policy acts once per episode step, `dt = 0.05` s, and the physics runs `substeps = 2` steps of `h = dt / substeps = 0.025` s per episode step, so that contact stiffness and the control rate can be tuned apart. The velocity-proportional terms (the agents' drag and lateral damping, the turning lag, the box's angular damping) are applied **implicitly** (`v / (1 + c h)`), which is stable at any rate, so only the contact springs bound the substep.

### 2.2 Arena and obstacles

- **The arena** is a walled square, 20 × 20 units by default, with the origin at its centre.
- **Obstacles are random per scenario**: a few circles and axis-aligned rectangles (by default 2 to 6 in all at full difficulty, §4.2). Circle radii 0.5 to 1.5 units; rectangle sides 0.5 to 4 units, so no obstacle is thinner than an agent's diameter.
- **Validity, with rejection.** A layout is accepted only if the free space is connected and the box can reach the target; otherwise it is redrawn from the same scenario stream. The check rasterises the arena on a grid of 0.25 units, inflates the obstacles and walls by the **box's circumscribed radius** (half its diagonal, so the check holds whatever the box's orientation) and flood-fills from the box's cell to the target's; a second flood fill, with obstacles inflated by an agent's radius, checks that every agent's spawn point is connected to the box (and no agent overlaps an obstacle, the box, a wall or another agent). New obstacles also keep clear of the box's starting footprint and of the target zone (and, from generator 2, the box keeps a clearance: the next item). This is conservative (it may reject a layout the box could cross by turning), which is the safe side. Generation happens once per scenario set, on the host, so it is plain numpy and may loop; a scenario records in `attempts` how many draws it took (most take one; dense difficulties take several).
- **Box clearance (generator 2, M2b).** Agents can only push, so a box against a wall or tucked beside an obstacle, with no
  room behind it for an agent, cannot be moved toward a target on its other side. The first generator kept the box's
  circumscribed circle only 0.3 units from obstacles and 0.5 from walls, less than an agent's diameter (0.5): five of the
  overnight run's 64 held-out layouts and one training layout were never moved, all with the box against a wall or beside an
  obstacle (§10.8). **Generator 2 keeps the box's centre at least its circumscribed radius plus 1.0 unit (two agent diameters)
  from every wall and every obstacle's surface**, so the clearance holds whatever the box's orientation. A box drawn too close
  to a wall redraws the layout; an obstacle drawn too close to the box is drawn again (as obstacles too close to the box or
  the target always were). The target, the obstacles' sizes and the agents' spawns are drawn as before. It is
  `Difficulty.box_clearance` (`scenarios.BOX_CLEARANCE`), at both difficulties.
- **One generator** (since R1). M2b kept the generator before the clearance as **version 1**, bit for bit, so that runs
  recorded before M2b could still be resumed and reported (`scenarios.generator` in their config). R1 dropped those runs and
  removed version 1 with the config key: the generator is the former version 2, unchanged (its sets are pinned against it),
  and `EASY` and `FULL` carry the clearance.

### 2.3 The box

- **A rectangle**, 2 × 1.2 units by default, with mass `M` and moment of inertia `I = M (w² + h²) / 12` (a uniform rectangle).
- **It translates and rotates.** Contact forces applied off its centre create torque, so pushing at a corner turns the box, and a team must push near the middle (or balance its pushes) to move it straight.
- **It moves only on the floor's terms** (§2.4); it has no actuator of its own.

### 2.4 Cooperation is necessary: friction

The box rests on the floor with **static friction**, so it moves only when the combined push exceeds a threshold, and the threshold is set so that it **needs at least `k` agents pushing together** (`k = 4` by default). One agent alone cannot move it; neither can `k − 1`.

- **The threshold from `k`.** With `F_a` an agent's maximum thrust, the static threshold is set between what `k − 1` and `k` agents can push: `F_s = (k − 0.5) · F_a` (7 with the defaults, `F_a = 2`). Four agents pushing exactly in line produce `4 F_a`, enough with a margin of `0.5 F_a`; they move the box as long as each push is within about 29° of the direction they move it in (`4 cos 29° ≈ 3.5`). This margin is a lever: smaller makes cooperation stricter and the first generations' search harder.
- **At rest.** While the box is at rest, it stays at rest as long as the net horizontal force on it is below `F_s` **and** the net torque below `τ_s`; `τ_s = F_s · ρ`, with `ρ` a friction radius of the order of the box's size (`ρ = (w + h) / 6` by default, about the mean distance of the box's footprint from its centre). Treating force and torque with two separate thresholds is a simplification of a real friction limit surface, chosen because it is two comparisons on arrays.
- **Moving.** Once moving, **kinetic friction** opposes the velocity with a constant magnitude `F_k = μ · F_s`, a **kinetic torque** `τ_k = μ · τ_s` opposes the spin, and **angular damping** (rate 2/s) slows it further. Each substep removes `F_k h / M` from the speed and `τ_k h / I` from the spin, clamped at zero so that friction never reverses a motion. The box comes to rest again when its speed and spin are both below small thresholds (0.01) and the push no longer exceeds the static thresholds.
- **`μ = 0.9`, changed at M1a from the 0.8 first proposed.** Kinetic friction must exceed what `k − 1` agents can push, `μ (k − 0.5) > k − 1`, or a team that got the box moving once (with an impact, or a `k`th agent that then leaves) would need one agent fewer from then on. With `k = 4` that needs `μ > 0.857`; 0.8 failed it. `WorldParams` refuses a `μ` that fails it, with the value needed (for `k = 6`, `μ > 0.909`).
- **The kinetic torque, added at M1a.** The first design had only viscous angular damping on a moving box. The threshold demo showed the problem: once the first contact's jolt (below) had set the box moving, `k − 1` agents pushing slightly off-centre kept it turning, and drifting, indefinitely, since any torque however small sustains a spin against viscous damping alone. A Coulomb torque, like the force, makes a small torque stop it.
- **Agents pushing agents.** Contact forces pass through agents, so a line of agents pushing each other into the box adds up like agents pushing it side by side. This is allowed: it is cooperation too.
- **Impacts.** Static friction holds against forces, not impulses: an agent hitting the box at full speed briefly exceeds `F_s` (the contact spring peaks at about `√(k_c m) v ≈ 19`), and the box slides a little, then stops. `k − 1` agents starting in contact jolt it about 1 cm as their springs overshoot (damping ratio 0.3), then hold it. §10.2 notes what evolution might make of impacts.

### 2.5 Agents

- **Discs** of radius 0.25 by default, mass `m` (1 by default).
- **Differential drive.** Each agent has a heading; its actions are a **forward thrust** (a force along the heading, at most `F_a = 2`; negative for reverse, limited to half the forward thrust) and a **turning rate** (at most `ω_max = π` rad/s). Linear and angular drag, and maximum speeds: the maximum speed (1.5 units/s) sets the linear drag `F_a / v_max`, so it is the terminal speed under full thrust. The commanded turning rate is reached through a first-order lag (time constant 0.1 s), and the velocity across the heading is damped strongly (rate 20/s), as wheels resist sliding sideways; contacts can still push an agent sideways.
- **Scaling.** Actions arrive in `[-1, 1]` (§3.6) and are scaled to the agent's limits: thrust to `[-F_a / 2, F_a]` and turning to `[-ω_max, ω_max]`.

### 2.6 Collisions: penalty contacts

Agents collide with each other, with walls, with obstacles and with the box; the box collides with walls and obstacles. **Contacts are penalty-based**: an overlap `δ > 0` between two bodies produces a spring–damper force `f = k_c δ + c_c δ̇` along the contact normal (only pushing, never pulling), applied at the contact point, so it creates torque on the box. Penalty contacts are chosen **because they vectorise cleanly**: every pair is an independent formula evaluated on arrays, with no constraint solver and no data-dependent iteration. Contacts are frictionless (agents slide along walls and along the box): friction between agents and the box is a possible later refinement. A consequence seen in the demos: an agent pushes the box only along the face's normal, so a push at an angle to a face loses its tangential part and the agent slides off along the face.

The contact pairs:

| Pair | Bodies | Note |
|---|---|---|
| disc–disc | agent–agent, agent–circular obstacle | the `N²` pair (§2.7) |
| disc–axis-aligned rectangle | agent–rectangular obstacle | closest point by clamping |
| disc–oriented rectangle | agent–box | the same, in the box's frame |
| oriented box–circle | box–circular obstacle | the circle against the box, in the box's frame |
| oriented box–axis-aligned rectangle | box–rectangular obstacle | **the hardest one** |
| everything–wall | agents and the box against the four walls | half-planes; the box by its four corners |

- **The rotating box against rectangles** is the hardest pair: two rectangles at an angle meet corner-to-edge or edge-to-edge. It is computed as corner penetration both ways: the box's four corners against the obstacle (in the obstacle's frame) and the obstacle's four corners against the box (in the box's frame), each penetrating corner contributing a spring force along the face it is closest to. It is a separating-axis-like approximation, exact for the corner–edge contacts that dominate and good enough for the rare edge-to-edge ones (two corners penetrate). Where the corners overlap each other (the box's corner inside the obstacle and the obstacle's inside the box) both count, so that contact is twice as stiff; a test pins it. A crossing with no corner inside either body is missed, but it can only be reached through a corner contact first, which the stiff box contact resists. §10.2 lists its risks.
- **Stiffness against the time step.** A penalty contact is a spring: stiffer means less overlap but needs a smaller step to stay stable (for semi-implicit Euler, `ω dt < 2` with `ω = √(k_c / m)`, and comfortably below for accuracy). The stiffness is chosen so that the overlap at full push stays small against the agent's radius (a few hundredths of a unit), and the substep from it. **Tuned at M1a:**
  - agents: `k_c = 200` (an agent at full thrust overlaps 0.01 units);
  - **the box against walls and obstacles: `k_box = 2000`**, a separate stiffness added at M1a. The box is four times heavier than an agent and a crowd pushes it, so with the agents' stiffness 20 agents would press it 0.2 units into an obstacle and a large crowd could push its corner past the middle of a thin one; ten times stiffer, the same frequency for a mass four times larger;
  - **damping as a ratio of critical damping, `ζ = 0.3`**, per contact type from its reduced mass (an agent pair `m/2`, agent–box `m M / (m + M)`), so that all contacts lose about the same fraction of energy per bounce;
  - `WorldParams` computes `ω h` for each contact type (agent–agent, agent–wall, agent–box, box–wall) and refuses a combination above 1, half the stability bound of 2, which leaves room for crowds where a body feels several springs at once. The defaults give 0.56 (box–wall).
- **No tunnelling.** At maximum speed an agent moves much less than its radius in one substep (for example 1.5 units/s × 0.025 s ≈ 0.04 units against 0.25), and walls are half-planes, so nothing passes through a wall; tests check it at maximum speed, for walls and for the thinnest obstacle (M1a).

### 2.7 Cost: agent pairs grow as agents²

Agent–agent contacts and teammate sensing (§3.5) are computed for all pairs, `(n, s, N, N)` arrays. This is fine at 20–40 agents; at 100 agents the pairs are 10,000 per world, and **the torch path on Metal is expected**. A spatial grid (agents binned into cells, pairs only between neighbouring cells) could replace the dense pairs later if needed; it is not in v1, because sorting and binning are where the Array API is weakest and determinism on a GPU is hardest. §10.3 has the measured cost (M1a).

### 2.8 Episodes in the batch

Episodes end early on delivery (§4.4), but a batched world cannot drop a scenario from its arrays. **Finished scenarios keep their array shape but stop changing and scoring**: a `done` mask of shape `(n, s)` freezes their state through `xp.where`, and their measurements are kept from the step they finished. The batch stops early only when every scenario of every candidate is done (`simulate` checks it once per step, one host read). Frozen scenarios still cost compute, which is the price of no Python loops. The state's `steps` counter stops on delivery, so it is the delivery step (M2's `steps` measurement).

### 2.9 What `run_batch` returns, and rendering

`run_batch(agents, scenarios, rng)` returns an `EpisodeBatchResult` of **raw measurements** (§5.2), each an `(n, s)` array on the backend, as Auxein's environments do: the score is the aggregator's (§5.1). For rendering (§7), the same simulation can be run for one genome on a few scenarios with a hook that records the states at every step (positions, headings, the box's pose); one code path serves training and rendering, so the clips show what was scored. Built at M1a: `simulate(..., hook=TrajectoryRecorder(), every=k)` copies the state to host numpy arrays every `k`th step, and the renderer draws the resulting `Trajectory`, for the scripted controllers that read the true state. **Built at M1b** (§2.11): `BoxPushingWorld.run`, the step loop of `run_batch` and `run_episode`, takes a hook that sees each step's state with what the agents sensed in it, and `BoxPushingWorld.watch` records the trajectory for videos of brains, so a video of a brain is the episode that would be scored (M1b's recorder of what agents sensed, for the sensing overlay, was removed at R1).

### 2.10 The physics core as built (M1a)

M1a builds the physics as a pure, batched core in `world.py`, separate from Auxein's environment interface, which M1b wraps around it (`run_batch`, sensing, measurements):

- **`Layout`**: what a scenario fixes for the episode (obstacles, target, starting poses) as `(1, s, ...)` arrays, which broadcast over the candidates without a copy. **Obstacles are padded** to a fixed number of circle and rectangle slots per batch, with validity masks, so layouts with different numbers of obstacles share arrays. Contacts with obstacles are accumulated **slot by slot** (a loop over the few slots, never over candidates, scenarios or agents), so a padded slot adds an exact zero after the real ones: a scenario gives the same numbers, to the last bit, whether padded to 6 or to 10 slots. `scenarios.layout_arrays` builds it from scenarios' params.
- **`State`**: the agents' positions, velocities, headings and turning rates `(n, s, N)`; the box's pose, velocity, spin and at-rest flag `(n, s)`; the `done` mask and the `steps` counter.
- **`step(state, actions, layout, params) -> State`**: actions `(n, s, N, 2)` in `[-1, 1]` (clipped), applied for `substeps` substeps (thrust and contact forces, the box's friction, implicit damping, integration), then delivery detection and the freeze of done scenarios through `where`.
- **`simulate(state, layout, params, controller, steps, hook=None, every=1)`**: the runner for tests, demos and rendering; a controller maps the state to actions.
- **`WorldParams`** (`config.py`): every parameter with these defaults, the derived quantities (`F_s`, `τ_s`, `F_k`, `τ_k`, the inertia, the drag, the contact damping) computed in one place, the checks above, and `to_dict` / `from_dict` for JSON. The TOML training config (M2) will build it.
- **Scripted controllers** (`scripted.py`) drive the core from the true state for tests and demos: `idle`, `constant`, `go_to` and `push` (agents queue in columns behind the box's far side from a goal and push toward it; naive, it ignores obstacles). With 20 agents it delivers the box within the 600-step limit in 10 of 16 easy scenarios and 5 of 16 full ones (seed 1), the rest ending 1 to 4 units short (§10.2).
- **The checks it passes** (`tests/box_pushing/world_test.py`, on numpy float64 and torch float32): `k − 1` agents leave the box at rest and `k` move it, for `k` = 2, 3 and 4; `k − 1` agents cannot keep a moving box moving; an off-centre push turns the box the right way and a centred one does not turn it; no agent passes through a wall at full speed, and neither agents nor the box pass through a 0.5-thick obstacle; with thrust, drag and damping off, collisions conserve momentum and never add kinetic energy; 100 agents crowding the box with random actions for 2,000 steps stay finite in float32 (numpy and torch); delivery freezes its scenario while the others go on; trajectories are identical run to run, identical whether a scenario runs alone or in a batch (bit for bit on numpy, within 1e-3 on torch float32) and whatever the obstacle padding.

### 2.11 The environment as built (M1b)

`environment.py` wraps the physics core and the sensors (§3) as an Auxein environment. **`BoxPushingWorld(params, sensing, steps=None, backend=None)`** implements both of Auxein's interfaces, so `EpisodeEvaluator(brain, world, training_set, aggregator)` works on either path:

- **`run_batch(agents, scenarios, rng)`**, the batched path (taken when the decoder has `decode_batch`): every candidate on every scenario at once, measurements as `(n, s)` arrays on the backend. `agents` is anything with `act_batch` (Auxein's `BatchPolicy`), not only `MLPPolicy`'s networks: the scripted policies of the tests and demos use it too. The world simulates where the agents are: it reads the backend from `MLPAgentBatch`'s weights (or a policy's `backend`, or the world's `backend=`) and the batch size from `n`, since Auxein's interface carries neither (§10.5).
- **`roles = ("team",)` and `run_episode(agents, scenario, rng)`**, the per-episode path `EpisodeEvaluator` requires of every environment: the same loop with `n = 1, s = 1`, the agent's `act` applied to the team's `(N, 43)` observations, on the world's `backend=` (numpy float64 by default). The two paths agree: `EpisodeEvaluator` on either finds the same best candidate, with the same objective, along the same trace, on numpy float64 (the brain's products are grouped differently, a few ulps apart).
- **One step loop**, `BoxPushingWorld.run(act, layout, n, hook=None)`: observe the state at the start of the step (§3), `act(observations)`, one physics step; at most `steps` steps (600 by default), ending early once every scenario of every candidate is done (one host read per step). The hook sees each step's starting state with its senses, then the final state.
- **No randomness** (§4.1): `rng` is accepted, as the interfaces require, and unused; a test checks that two streams give the same numbers.
- **The layout cache.** The padded `Layout` of a scenario sequence is built once and kept (the four most recent) by the scenarios' fingerprint (ids, seeds and params, as `ScenarioSet` computes it) and the backend. Not by ids alone: two training sets of different seeds share their ids (`train-0000`...).
- **Failures.** After every step, a scenario whose state is no longer finite (a physics blow-up, or a policy returning NaN) is **frozen at its last finite state** and marked done, so it cannot poison the rest of the batch or delay the early stop; `run_batch` reports it in `EpisodeBatchResult.failures` as `FAILED`, with an error naming the scenario and the step. Auxein then fails that candidate's evaluation (core design §6.6), records the episode, and the run goes on. Its measurements are finite but meaningless, and ignored.
- **The brain**, `environment.brain(sensing, hidden=(32,))`: `MLPPolicy(43, (32,), 2, activation="tanh", output="tanh")`, its sizes from the observation layout (the smoke test takes them from there, M1b).
- **The checks it passes** (`tests/box_pushing/environment_test.py`, `auxein_test.py`): the six measurements as `(n, s)` arrays on the backend and device; the paths agree; the layout is built once for repeated calls; early stop; the M1a push team, driven through the loop, delivers what `simulate` delivers at the same step and to the same final distance; `max_pushers` and `approach` against hand-built scenes; a NaN action fails only its scenario, on both paths; a two-generation `auxein.run` with `RandomSearch`, a placeholder aggregator (defined in the tests only; the real one is M2's) and recording on completes, and every recorded episode holds the six measurements; the same genomes give the same measurements twice, a candidate gets the same alone as in a batch, and a scenario the same whatever its place in the set (bit for bit on numpy; within 1e-3 on torch float32, though they were identical on this machine's CPU). On numpy float64 and torch float32.

---

## 3. Agents: sensing and acting

**All sensing is local and in the agent's own frame** (bearings relative to its heading), **except the target**, which every agent knows. Every input is normalised to roughly `[-1, 1]` or `[0, 1]`, since the network's initial weights assume inputs of unit scale (`MLPPolicy.space`, core design §6.2).

**Built at M1b** (`sensors.py`, `SensingParams` in `config.py`), as designed here, with no change of decision; §3.8 has the details the design left open. **Sensing runs once per episode step**, on the state at the start of the step (not per physics substep).

### 3.1 Target (3 values)

Distance to the target (normalised by the arena's diagonal) and bearing to it as `(sin, cos)`. Always available.

### 3.2 Range rays (12 × 2 = 24 values)

12 rays evenly spaced around the agent (by default), each cast to a maximum range (the sensing range `r`, §3.3) and returning:

- the **normalised distance** to the nearest wall, obstacle or box along the ray (1 when nothing is within range);
- **what it hit**: 1 for the box, 0 for a wall or an obstacle (and for nothing).

Rays do not see teammates; teammates are sensed through sector densities (§3.4), so that a crowd around an agent does not blind it to the box.

### 3.3 Box cue in range (6 values)

Within the sensing range `r` (5 units by default): a **flag** that the box is in range (1), the **distance** to the box's centre (normalised by `r`), the **bearing** to it `(sin, cos)`, and the **box's heading relative to the agent's** as `(sin 2θ, cos 2θ)`. The doubled angle because the box is symmetric under a half turn: `θ` and `θ + π` are the same situation, and `(sin θ, cos θ)` would give them different inputs that the network would have to learn to treat alike (decided at M1a, §10.6). Beyond `r`, all six are 0 and the flag is off. The box is also visible to rays beyond `r` only if the ray range is longer; with both at `r`, an agent farther than `r` from the box knows nothing of it (§10.1).

### 3.4 Teammates: sector densities (8 values)

The space around the agent is divided into 8 sectors (by default); each value is the **closeness-weighted count** of teammates in that sector within the sensing range, `x = Σ (1 − d / r)` over the teammates at distance `d < r`, **squashed** to `1 − e^(−x)`. The size is **fixed whatever the team size**, and the squashing keeps the values in `[0, 1)` however dense the team, so the same brain sees inputs of the same range with 20 or 100 agents (decided at M1a, §10.6): one teammate close by gives about 0.6, a crowd saturates toward 1.

### 3.5 Proprioception (2 values)

The agent's own forward speed and turning rate, each normalised by its maximum.

### 3.6 Actions and the brain

- **No signalling** between agents in v1: they coordinate only through what they sense of each other and of the box. Signals (an output broadcast to neighbours) are a v2 idea.
- **Actions:** two outputs in `[-1, 1]`, from `MLPPolicy` in `"tanh"` output mode: forward thrust and turning, scaled to the agent's limits (§2.5).
- **One genome is one network, and that network is the whole team's brain.** The brain is one `MLPPolicy` shared by all agents (parameter sharing): every agent runs its own copy with identical weights, fed its own observations, so **agents act differently only because they sense differently**. Evolution selects teams, not individuals, so there is no credit assignment problem between agents, and the same brain works for any team size. This is exactly Auxein's batched path: `decode_batch` turns the `(n, d)` genome array into `n` networks, and `act_batch` takes the observations `(n, s, N, 43)` and returns the actions `(n, s, N, 2)` as batched matrix products on the backend's device.
- **The network:** one hidden layer of 32 `tanh` units by default, feed-forward, **no memory**. Recurrent brains are the first fallback if searching for the box proves too hard for a memoryless brain (§10.1); Auxein has no recurrent decoder (core design §15), so that would be a request to Auxein or a decoder here.
- **Future options:** heterogeneous teams (per-agent weights; the genome grows with the team) and castes (a few networks, each agent assigned to one).

### 3.7 Sizes

| Input | Values |
|---|---|
| target: distance, bearing (sin, cos) | 3 |
| range rays: 12 × (distance, hit the box) | 24 |
| box cue: flag, distance, bearing (sin, cos), relative heading (sin 2θ, cos 2θ) | 6 |
| teammates: 8 sector densities, squashed | 8 |
| proprioception: forward speed, turning rate | 2 |
| **observation size** | **43** |

With one hidden layer of 32 units and 2 outputs, the genome is `43 × 32 + 32` (hidden layer) `+ 32 × 2 + 2` (output layer) = **1,474 weights** (`MLPPolicy(43, (32,), 2, output="tanh").dim`; checked by the scaffold's smoke test). It is laid out as Auxein's `MLPPolicy` lays it out (core design §6.2), and searched in `policy.space(scale)`, whose fan-in–scaled bounds make the GA's uniform first generation a Glorot-like initialisation.

### 3.8 Sensing as built (M1b)

**The observation layout is defined once**, in `sensors.observation_layout(sensing)`: the named parts of the vector, their slices and sizes (12 rays and 8 sectors, fixed in code since R1; `SensingParams` holds only `range = 5`, which the rays reach too), and the range of each. The environment, the senses-only controller, the brain files (which record it and refuse another, §7.1) and the smoke test read it from there.

| Part | Slice | Values | Range |
|---|---|---|---|
| `target_distance` | 0 | the distance to the target over the arena's diagonal (28.3) | `[0, 1]` |
| `target_bearing` | 1–2 | `(sin, cos)` of the bearing to the target | `[-1, 1]` |
| `ray_distance` | 3–14 | per ray, the distance to the nearest wall, obstacle or box over the ray range; 1 for nothing in range | `[0, 1]` |
| `ray_box` | 15–26 | per ray, 1 if what it hit first is the box | `{0, 1}` |
| `box_flag` | 27 | 1 if the box's centre is within `r` | `{0, 1}` |
| `box_distance` | 28 | the distance to the box's centre over `r` | `[0, 1)` |
| `box_bearing` | 29–30 | `(sin, cos)` of the bearing to the box's centre | `[-1, 1]` |
| `box_heading` | 31–32 | `(sin 2θ, cos 2θ)`, `θ` the box's heading less the agent's | `[-1, 1]` |
| `teammates` | 33–40 | per sector, `1 − e^(−x)`, `x = Σ (1 − d / r)` over the teammates within `r` in it | `[0, 1)` |
| `speed` | 41 | forward speed over the maximum speed, clipped | `[-1, 1]` |
| `turn` | 42 | turning rate over the maximum turning rate, clipped | `[-1, 1]` |

The box cue's five values after the flag are 0 when the box is out of range. Details the design left open, decided at M1b:

- **Frames.** Bearings are counter-clockwise from the heading. Ray `k` points at `2πk / 12` (ray 0 straight ahead); sector `k` is centred on `2πk / 8` (sector 0 straight ahead, each 45° wide).
- **Rays start at the agent's centre** (an agent touching a wall reads 0.05) and **report 0 from inside a shape** (an agent pressed slightly into an obstacle). They are cast by the slab method against the arena (from inside), the axis-aligned rectangles and the box (in its frame), and by the quadratic against circles (`geometry.ray_*`, M1b); a ray grazing an edge, a corner or a circle hits it.
- **Speed is clipped** to `[-1, 1]`: reverse thrust tops out at −0.5, but a contact can push an agent faster than its top speed for a moment.
- **Cost.** The rays are `(n, s, N, 12)` arrays, cast slot by slot against the padded obstacles (an exact minimum, so padding does not change a reading); the teammates are `(n, s, N, N)` arrays: **pairwise distances once per step**, shared by the weights and the sectors, summed per sector in a loop over the 8 sectors (rather than a one-hot `(n, s, N, N, 8)` array, eight times the memory). Sensing costs more than the physics (§10.3).
- **What the tests pin** (`tests/box_pushing/sensors_test.py`, on numpy float64 and float32 and torch float32): the bearing to the target in the agent's frame, rotating back as the agent turns; rays hitting a wall, an obstacle and the box with the right flag, an obstacle hiding the box, nothing in range reading 1; the box cue on within `r` and all zeros beyond; the relative heading identical for the box at `θ` and `θ + π`; one teammate 0.5 ahead reading 0.59 in sector 0, a crowd of ten above 0.99, an agent never sensing itself, sectors turning with the agent; proprioception normalised; every value within its range on random states; an agent's observation independent of the batch and of the obstacle padding (bit for bit on numpy). The ray casts have their own hand-computed tests (`tests/common/geometry_test.py`).

---

## 4. Scenarios, episodes and the training set

### 4.1 A scenario

A scenario is drawn from a seed: the **obstacle layout**, the **box's position and orientation**, the **target position** (at least a minimum distance from the box), and the **agents' starting positions and headings, scattered across the arena**, free of obstacles (and of each other). It is an Auxein `Scenario` in a `ScenarioSet`, so **common random numbers** apply: every candidate faces the same scenarios, and differences in fitness are differences between brains.

- **The generated layout is stored in the scenario's `params`** (obstacles, box pose, target, agent poses, the difficulty), not regenerated from the seed in the world. The world then has no randomness of its own, the scenario set's fingerprint covers the actual layouts (so a change to the generator cannot silently change a recorded run's scenarios: resuming is refused, core design §6.4), and the set can be saved and inspected as JSON. A layout of 100 agents is a few kilobytes. The format (`scenarios.py`): `box` `[x, y, angle]`, `target` `[x, y]`, `circles` `[[x, y, radius], ...]`, `rects` `[[x, y, half_width, half_height], ...]`, `agents` `[[x, y, heading], ...]`, plus `difficulty` and `attempts`.
- **The world has no noise in v1** (no sensor or actuator noise): the episode's randomness is entirely the scenario's. Noise is a possible lever for robustness on unseen layouts (§10.6).

### 4.2 Difficulty and the training set

**Difficulty is a parameter of the generator**: the number of obstacles, the box–target distance, and the spawn spread (how far from the box the agents start). **The training set mixes difficulties**, so the first generations get a gradient:

| Parameter | easy | full |
|---|---|---|
| obstacles | 0 to 2 | 2 to 6 |
| box–target distance | 3 to 5 units | 6 to 12 units |
| spawn | within 3 units of the box | scattered across the arena |

**A training run uses 32 scenarios: 4 easy and 28 full** (decided at M2b; M2a's runs used 4 easy and 12 full), the training
config's default (`[scenarios]`, §6), in that order (ids `train-0000`...), **fixed for the whole run**: parents and children
are judged on the same scenarios, so a better score is a better brain, not easier scenarios (no rotating subsets). M2a chose 16
to fit about half a minute per generation on Metal (§10.3); the overnight run then showed the generalisation gap growing with
training (0.10 to 0.24, §10.8), part of what later generations learned being the maps, so M2b gives the run **more full
layouts**. The cost is about twice the time per generation: the probe's 100 × 32 row is 41 s at 600 steps, about a minute at
900; **the new pilot measured 56 s per generation, so the overnight run is about 6.2 hours** for 400 generations, with a
recording of about 0.7 GB (§10.4, §10.9); `train`'s start line estimates the run's hours. The generator's own default without a
mix (`scenarios.training_set`) is the same 4 + 28 since R1 (M1's measurements in this document used 8 easy and 24 full). Easy scenarios give a clueless team something to stumble into (agents start near the box, close enough to see it, §10.1); the full ones keep evolution pointed at the real task.

### 4.3 Held-out sets

**Fresh layouts never seen in training, at full difficulty**, evaluated at the end of a run with `auxein.driver.evaluate_held_out` (`held_out.csv`, §6.1): the evidence for the "deployment to unseen situations" claim. 64 scenarios by default (ids `held-0000`...), from **`scenarios.seed + 1`** (since R1; a separate `held_out_seed` before) on a **stream name** of their own, so that they are disjoint from the training set by construction, even with the same seed; `check_disjoint` verifies that no layout repeats. The strategy never sees them. `deploy` draws its scenarios on the same stream from another seed, fresh by default (§7.2), so deploying on the held-out seed repeats the held-out evaluation.

### 4.4 Episodes and delivery

- **A fixed limit**, **900 steps** by default (45 s of simulated time at `dt = 0.05`), **ending early on delivery** (§2.8). **Raised at M2a from 600**: four agents in line barely move the box (about 0.3 units/s, §10.2), so 600 steps left a team little time to find the box and then push it up to 12 units; at about 29 s per generation (§10.3) the longer episodes still fit the budget. The friction threshold (`k = 4`) does not change: it is the cooperation story. The step limit is a parameter (`world.simulation.max_steps`), not physics. M1's measurements in this document (§2.10, §10.1, §10.3) were taken at 600.
- **Delivery:** the **box's centre enters the target zone**, a circle (of radius 1.5 units); the box's orientation does not matter.
- **Team size** is a parameter: 20–40 typical, up to 100. One run uses one team size; the brain does not depend on it (§3.4), so a brain evolved with 20 agents can be shown with 40.

---

## 5. Fitness

### 5.1 One shaped score

**A single score per episode, maximised** (`Objective("score", "maximise")`), aggregated over scenarios with the mean by default (the aggregator is configurable: a lower quantile or `cvar_lower` would reward teams that rarely fail). Its terms:

- **Progress** of the box toward the target, the main term: `(d₀ − d_end) / d₀`, the initial minus the final distance of the box's centre to the target's, normalised by the initial one (1 when delivered at the centre, 0 when untouched, negative when pushed away; clipped at −1, decided at M2a).
- **A delivery bonus**, large, and **a time bonus** for delivering early: `delivered · (w_d + w_t · (1 − t / T))`, with `t` the delivery step and `T` the step limit.
- **An approach term**, small: how close the agents get to the box, the mean over agents of `1 − min(1, d_min / r)`, with `d_min` the agent's closest gap to the box's surface during the episode (§5.2). It helps the first generations get started, when no team moves the box, and **must stay small so it cannot dominate**: a team that only gathers around the box must score less than one that moves it. **Decided at M2a:** it counts, at weight 0.1, for the whole run (an aggregator is fixed for a run, so fading it over generations would take a staged run).

**Default weights**, decided at M2a and to be tuned after the overnight run (M2b):

| Term | Weight |
|---|---|
| progress | 1.0 |
| delivery bonus `w_d` | 2.0 |
| time bonus `w_t` | 1.0 |
| approach | 0.1 |

### 5.2 Measurements, and where the score is computed

As Auxein's design asks (core design §6.2), **the world returns raw measurements, not the score**; the score is computed by the `Aggregator` from them. Then the weights can be changed and a recorded run re-judged without re-simulating (`RunReader.reaggregate`), and the measurements are recorded per episode (the `episodes` table) for the reader and the charts. The measurements:

| Measurement | Meaning |
|---|---|
| `delivered` | 1 if the box reached the target zone, else 0 |
| `steps` | the step it was delivered at, or the step limit |
| `initial_distance` | the box's initial distance to the target |
| `final_distance` | its distance at the end (at delivery, if delivered) |
| `approach` | the approach term's value (§5.1) |
| `max_pushers` | the largest number of agents touching the box at once: a direct measure of cooperation, for the charts |

**Built at M1b** (`environment.py`, §2.11), as raw `(n, s)` arrays on the backend, in the run's float dtype:

- `delivered` is the `done` mask of a scenario that did not fail; `steps` the state's step counter, which stops on delivery (§2.8), so it is the delivery step or the steps simulated; `initial_distance` and `final_distance` are the box's centre to the target's, at the start and at the end (frozen at delivery).
- **`approach`** is §5.1's mean over agents of `1 − min(1, d_min / r)`, with `d_min` the running minimum, from the initial state on, of the **gap between the agent's disc and the box's surface** (its centre's distance to the surface less its radius, at least 0), so that an agent pressed against the box scores 1 rather than 0.95. Whether the term should count in the score, and how much, is still item 11 (§10.6).
- **`max_pushers`** is the running maximum, from the initial state on, of the number of agents whose gap is at most 0.02 (the renderer's "touching", `TOUCH_TOLERANCE`). **Decided at M1b** (item 12): it costs one `(n, s, N)` comparison per step. It counts **contacts, not contributions**: agents queued behind others push through them (§2.4), so a delivering team may touch the box with as few as two agents (the M1a push team, seen in a test). The chart should be read as "agents at the box", and a column of pushers counts once per contact.

The aggregator's **descriptors** carry the per-run summaries the charts and the early stop need: the success rate (mean of `delivered`), the mean progress, the mean approach and the mean of `max_pushers`.

### 5.3 Fitness as built (M2a)

`fitness.py`. Per episode, with `d₀` and `d` the box's initial and final distance to the target, `t` the delivery step and `T` the step limit:

```
score = w_p · max(−1, (d₀ − d) / d₀) + delivered · (w_d + w_t · (1 − t / T)) + w_a · approach
```

with `Weights(progress=1.0, delivery=2.0, time=1.0, approach=0.1, progress_floor=−1.0)`. The score is in `[−1, 4.1]`: −1 for a box pushed away by its own distance or more, 0.1 at most for gathering, 3 to 4.1 for a delivery.

- **The score lives in the aggregator** (`fitness.aggregator(weights, max_steps)`), never in the world: one objective, **`score`, the mean over the scenarios, maximised**, and the descriptors `success_rate` (mean of `delivered`), `progress` (mean of the clipped progress), `approach` and `max_pushers` (means). A recorded run can be re-judged with other weights (`RunReader.reaggregate`).
- **The score is a named, parameterised source**, `EpisodeScore(weights, max_steps)`, whose label lists the weights and `T`: `mean(<episode_score(progress=1.0, delivery=2.0, time=1.0, approach=0.1, progress_floor=-1.0, max_steps=900)>)`. The run's description, which Auxein compares on resume, therefore says which weights scored it, and a resume with a changed weight is refused (§6.1). Auxein `v0.3.2` labelled a callable source by its `__qualname__` alone, hiding the weights, so a changed weight resumed silently: an Auxein bug, fixed in `v0.3.3`, which labels a source by its `describe()` (§10.5).
- **Gathering against pushing** (a test, with scripted teams in one scenario, 900-step scoring): twelve agents that only gather around the box, all touching it, score 0.100 (approach 1). The push team in the same scenario, after 50 steps (the box moved a third of a unit, 4 % of the way), scores 0.111 and wins. After 40 steps (0.11 units, 1.4 %) it scores 0.079 and **loses**: its column's back agents are still far from the box (approach 0.65 against 1). In general a push beats gathering once its progress exceeds `0.1 × (a_gather − a_push)`, at most 10 % of the way. So the approach term cannot dominate a push that gets going, but a team that only nudges the box while half of it hangs back scores less than one that just gathers: a small valley between "gather" and "push" for the first generations. M2b may lower `w_a` if the pilot or the overnight run shows teams settling for gathering (§10.7).

---

## 6. Training

- **Strategy.** **`GeneticAlgorithm`** is the main strategy: a population evolving is the clearest story. **`PycmaStrategy`** (CMA-ES) on the same setup is a comparison run (M4); §10.6 notes its limits at 1,474 dimensions.
- **Driver settings.** Generation delivery, deterministic mode, recording on, progress printed with `auxein.callbacks.print_progress`, a checkpoint every 10 generations. **The run stops on its number of generations only** (decided at M2a): no held-out evaluation and no early stop during the run (reading the recording of a running run for it is unchecked, §10.5). The held-out evaluation comes at the end of the run (§6.1).
- **The evaluator.** `EpisodeEvaluator(policy, world, training_set, aggregator)`, which takes Auxein's batched path because the world has `run_batch`, `MLPPolicy` has `decode_batch` and the GA's batches are arrays.
- **Sizes** (decided at M2a, from the owner's Metal probe, §10.3): **40 agents, a population of 100** (100 children per generation), **32 training scenarios** (4 easy, 28 full, fixed for the run; 16, with 12 full, until M2b, §4.2), 900 steps. 56 s per generation on Metal (29 s with M2a's 16 scenarios).
- **Backend.** **torch float32 on Metal** (`mps`) by default; torch on the CPU when Metal is not available, with a message saying so (and several times slower). numpy stays the backend of tests.
- **One command** trains from a TOML config, reproducible from its seed: `uv run --extra torch scenarios box-pushing train CONFIG` (§6.1).

### 6.1 Training as built (R1)

**The config** (`config.TrainingConfig`, TOML through `tomllib`) is the run's definition, and since R1 it holds **only what an experiment varies**. A config lists only what it changes; the run's `config.toml` is written with every default filled in, so it defines the run on its own. Unknown keys and bad values are errors naming the key (`unknown key 'evolution.populaton'`, `team.agents must be an integer, got 'forty'`); **a key that is not configurable** (any removed at R1) is an error saying what to do instead (`'strategy' is not a config key: use [evolution] population …`).

| Key | Default | Meaning |
|---|---|---|
| `name` | the file's name | the run directory's prefix |
| `[world.arena]`, `[world.box]`, `[world.friction]`, `[world.agent]`, `[world.contact]`, `[world.simulation]` | §2 | every physics parameter of `WorldParams`, e.g. `[world.simulation] max_steps = 900`; validated as the world is built (§2.6's stability check included) |
| `[sensing] range` | 5.0 | the sensing range `r`; the rays reach as far. 12 rays and 8 teammate sectors are fixed in code: they set the observation's size, so the shape of every brain |
| `[team] agents` | 40 | the team size |
| `[brain] hidden` | `[32]` | the hidden layers' widths (`tanh`); the genome space is `MLPPolicy.space(1.0)` |
| `[scenarios] easy`, `full` | 4, 28 | the training set, easy then full, fixed for the run |
| `[scenarios] held_out` | 64 | the held-out set's size, full difficulty |
| `[scenarios] seed` | 1 | the training set's seed; **the held-out seed is `seed + 1`**, its layouts drawn on a stream of their own, so the two sets cannot share a layout whatever the seeds (`check_disjoint` checks it at the end of a run) |
| `[fitness] progress`, `delivery`, `time`, `approach` | 1.0, 2.0, 1.0, 0.1 | the score's weights (§5.1); the progress floor stays −1 |
| `[evolution] population` | 100 | parents, and children per generation |
| `[evolution] generations` | 400 | the run's length; the only thing a resume may change |
| `[run] seed` | 1 | the run's seed |
| `[run] backend`, `device`, `precision` | `torch`, `mps`, `float32` | where it computes (`numpy` runs on `cpu`) |

**Fixed in code** (`train.py`): tournament size 2, crossover probability 1.0 (every child an intermediate recombination of two parents), self-adaptive mutation from an initial step of 0.1, and a checkpoint every 10 generations.

**Presets** (`configs/`), each listing what differs from the defaults and the sizes, so it reads on its own: `smoke.toml` (10 agents, population 10, 2 + 2 scenarios, 4 held out, 200 steps, 3 generations, numpy float64: seconds; the tests' run), `pilot.toml` (the full sizes, 25 generations: the start of the learning curve, about 25 minutes on Metal) and `full.toml` (the full sizes, 400 generations: about 6.2 hours).

**`scenarios box-pushing train CONFIG [--run-dir DIR]`** builds everything from the config (the training set, the world, the brain, the aggregator, `EpisodeEvaluator`, `GeneticAlgorithm`) and runs Auxein. It prints a start line (backend, sizes, the simulation's measured seconds per generation, the run's expected time at 1.4 times that, and the minutes the held-out judging at the end will take), a progress line per generation, and end lines saying where everything is and how to deploy the final brain. **Training only, no video.** The run directory, `runs/<name>-<timestamp>/` (ignored by git):

- **`config.toml`**: the effective config.
- **`recording/`**: Auxein's run directory (event log, metadata, checkpoints; `auxein.open_run`), a directory of its own because Auxein starts a run only in an empty or new directory.
- **`progress.csv`**, rewritten after every generation from the recording: Auxein's `generation_history()` (generation, evaluations, the best score so far, the generation's score best, mean, median and worst), the means of the descriptors over the generation's candidates, the best-so-far's descriptors and the active seconds. Built from the recording rather than from the callbacks, so that a resumed run's file covers the whole run.
- **`brains/`**: **`gen-<k>.safetensors`** for every **snapshot** (§7), written when its generation is reached, and **`final.safetensors`**, the run's best, written at the end **and on interruption** (Ctrl-C or a crash after the first generation), so a stopped run always has a usable brain.
- **`held_out.csv`**: at the end, every snapshot brain (and the last generation's) judged on the run's held-out set through Auxein's `evaluate_held_out` (`write=False`: nothing is written into the recording), with its training results from the recording: per snapshot, training score and success (all, easy, full), held-out episodes, deliveries, success, progress, `max_pushers`, median delivery step, and **the generalisation gap** (full-training success minus held-out success). Minutes on Metal; the start line says how many.
- **`charts/`**: **fitness** (the best so far, the generation's mean and median), **delivery** (the best-so-far's training success and the generation's mean), **cooperation** (`max_pushers`) over generations, from `progress.csv`, and **generalisation**, held-out against full-training success per snapshot, from `held_out.csv`; PNG (200 dpi) and SVG in the theme (§7).

**`scenarios box-pushing train --resume RUN [--generations N]`** continues a run from its recording and its `config.toml`: after Ctrl-C or a crash (Auxein replays from the last checkpoint; deterministic mode, so the result is the run that never stopped: the same event log, the same brains, the same `held_out.csv`), or with another number of generations (more to extend it, fewer to end it sooner, never fewer than it has). Nothing else can change: an edited `config.toml` is refused by Auxein, whose comparison of the evaluator sees the world's parameters and the score's weights. The new number is written to the run's `config.toml`, and the brains, `held_out.csv` and charts are refreshed at the end. **Ctrl-C stops a run resumably**, after the generation in progress (a second Ctrl-C stops at once).

**The checks** (`tests/box_pushing/train_test.py`, numpy, the smoke preset): a run writes `config.toml`, `progress.csv`, `held_out.csv` (equal to Auxein's evaluation of its snapshots), `brains/gen-*.safetensors` for its snapshots and `final.safetensors` (the run's best), and the four charts; an interrupted run has `final.safetensors` and resumes to the uninterrupted run's event log, brains and `held_out.csv`; an extended run refreshes its brains, charts and `held_out.csv`; fewer generations than a run has and an edited weight are refused; a new run never overwrites one; one Ctrl-C through the `scenarios` command stops a run in another process with exit code 130, leaving its final brain, and it resumes. **And the same numbers as before R1** (decision 9): a smoke run pinned from `main` before R1 (`tests/box_pushing/data/pre_r1_smoke.json`), trained again, has the same evaluations (every number to 9 significant digits) and final best on every platform, and the **same event log bit for bit** on the machine type it was pinned on (macOS arm64, the owner's). Elsewhere a float's last bit differs (the platform's libm, and the SIMD kernels numpy picks for the CPU: two Linux x86-64 machines gave two digests), so CI checks the 9-digit digest and the final best.

---

## 7. Brains, deployment and videos

Since R1, **videos come only from `deploy`, charts only from `train`** (§6.1), and what connects them is a brain file.

### 7.1 Brains as files

A run saves its brains as **`.safetensors`** files (the `safetensors` package's numpy API; `common/brains.py` for what is generic, `box_pushing/brain.py` for the scenario's metadata and checks), so that a brain is a small file that can be deployed, shared and read without the run that made it.

- **Tensors:** the network's layers, `layers.<i>.weight` `(out, in)` and `layers.<i>.bias` `(out,)`, from the input side, **in the run's precision**: exactly the `MLPPolicy` genome, per layer its weights row by row then its biases, so loading rebuilds the genome **bit for bit** (a float32 run's genomes are float32 values; saving refuses a genome its precision cannot hold). Any safetensors reader sees a plain network. A brain of the default sizes is about 8 KB in float32.
- **Metadata** (safetensors' string-to-string header, each value JSON):
  - `format`: `"auxein-scenarios/box-pushing/brain/1"`, and `scenario`: `"box-pushing"`;
  - `brain`: the spec (observation size 43, hidden widths, 2 outputs, `tanh` activation and output);
  - `observation_layout`: the observation's parts in order, `[[name, size], …]` (`sensors.observation_layout`): a brain reads its inputs by position;
  - `sensing` (range, rays, sectors), `world` (every `WorldParams` section) and `team` (agents): what it was trained in, so it runs in its own world wherever it is deployed;
  - `provenance`: the run (its config's name, its directory, its seed, its scenario sets with both seeds), the generation and candidate, whether it is the final brain, its training score and success, the fitness weights, the Auxein and scenarios versions, the scenarios commit (with `+dirty` for uncommitted changes, when run from a checkout) and the creation time.
- **Loading refuses** a file whose format tag is not this code's, or whose observation layout differs from the code's, with a message saying which, before anything runs.
- **Snapshots** (`snapshots.py`, what M3 called milestones): log-spaced generations (1, 2, 5, 10, 20, 50, 100, 200, 500, …) and the last, a snapshot skipped when its champion is the previous one's candidate, one `RunReader.best_up_to(k)` each (Auxein's `champions(every=n)` is arithmetic). A run of 400 generations has up to nine: 1, 2, 5, 10, 20, 50, 100, 200 and 400 (`final`). The decision to save `gen-<k>` reads only the recording, so a resumed run saves the same files as one never stopped.

### 7.2 Deploying brains (R1)

**`scenarios box-pushing deploy BRAIN... [options]`** (`deploy.py`) runs trained brains on scenarios they have not seen. It takes no TOML: its options are its whole interface.

- **The scenarios:** `--scenarios N` (64) layouts of `--difficulty full|easy` (full) from `--seed S`, drawn on the held-out stream for the brain's world and team. **By default the seed is fresh**: random, and not the training or held-out seed of any brain given, so the scenarios are new to every brain; it is recorded, so the deployment can be repeated. `--seed` may name any seed, the held-out one included: deploying a run's final brain on its held-out seed repeats the run's held-out evaluation, episode for episode (decision 9, a test).
- **Each brain runs in its own world**, the world, sensing and team size of its metadata; **`--agents N`** changes the team size, the brain unchanged (every agent runs it on its own senses). Brains from different worlds get different layouts (a layout depends on the arena, the box, the agents' size and the team) and baselines of their own.
- **The baselines on the same scenarios:** 20 **random brains** sampled from the brain's genome space with seed 0 (what an untrained team does, as a run's first generation) and the **senses-only controller** (§10.1: what these senses allow a hand-written memoryless policy).
- **Where it computes:** `--backend torch|numpy`, `--device`, `--precision`; by default torch float32 on the best available device (Metal, CUDA, then the CPU), numpy float64 when torch is not installed. Penalty contacts with many bodies are chaotic, so a float32 brain replayed in float64 or on another device may behave a little differently: `deploy.toml` records where it ran. **Brains of the same world, sensing, team and network run in one batch** (three brains on 64 scenarios with the video: about 2.5 minutes on Metal), so on a GPU a brain's float32 numbers depend slightly on which brains it is deployed with; on numpy they do not.
- **Output**, `deployments/<name>-<timestamp>/` (`name`: the first brain's run's config name; `--out DIR` instead; ignored by git):
  - `deploy.toml`: the effective options (the seed among them) and each brain's path, label, team size and provenance;
  - `results.csv`: one row per brain (or baseline member: `random brains 1` … `20`) per scenario: delivered, steps, progress (clipped at the fitness's floor, as the score), `max_pushers`, the box's starting distance to the target, the obstacles, the score with the brain's own weights;
  - `summary.md`: per brain and baseline, deliveries and success, mean progress, mean `max_pushers`, the median delivery step, and deliveries by the box's starting distance (6–8, 8–10, 10–12 units);
  - **`deployment.mp4`** (unless `--no-video`): **a row of panels per brain, on the same scenarios**: the first `--video-scenarios K` (2) of the set, not chosen for how they went. Each panel is captioned with the brain (its run and generation, or `final`), the scenario, and how its episode ended ("delivered at step k" or "not delivered · progress p"); a panel whose episode has ended holds its last state while the others play on. **Deploying a run's `gen-0001`, a middle snapshot and `final` together gives the before-and-after grid in one command**: this replaces M3's milestone clips and montages, which are removed.

### 7.3 Look and feel

- **One theme** (decided at M1a, `common/theme.py`) for every video and chart, **monochrome silver on near-black**, with no accent colour.
  - **State is shown with brightness, never with hue.** The box is outlined in steel at rest, filled bright white while moving, silver once delivered, when the target zone fills. Agents touching the box are bright white.
  - **Agents** are silver chevrons along their heading, inside a faint outline of their true disc, and agents close to each other are joined by faint lines.
  - **Type:** a small tracked uppercase label, a light headline (Inter Tight Light) and a muted lead (Inter Light) above the panels; small tracked captions in each panel. Inter and Inter Tight are bundled (`common/fonts/`, SIL Open Font License), so videos look the same on every machine, CI included.
  - **Depth:** the floor is lit faintly at its centre and dithered so that the shallow gradient does not band.
  - **No branding** in the videos.
- **Rendered offline** from a trajectory recorded by the same step loop that scores an episode (`BoxPushingWorld.watch`), so a video shows exactly what the physics computed. Each panel is its own trajectory's cell (`render.Panel`), so brains of different worlds and team sizes share a video.
- **Format:** MP4 (H.264, yuv420p), drawn with matplotlib and encoded by ffmpeg from the `imageio-ffmpeg` wheel, so no system ffmpeg is needed and `uv sync` alone is enough on a Mac and in CI. Charts: PNG and SVG, series told apart by brightness and dash, labelled at their ends (`common/charts.py`).
- **The checks** (`deploy_test.py`, `brain_test.py`, `render_test.py`): one brain and three brains deployed, a fresh seed disjoint from the training and held-out sets, `--agents` changing the team, the baselines present, the three files correct and parsing, the video's panels and captions (the text drawn in a few frames, `common.rendering.drawn_text`), `--no-video`, and deployment on the held-out seed equal to `evaluate_held_out`; brains round-tripping bit for bit in float32 and float64, their metadata, the refusals, and their size.

M3's sensing overlay, galleries of layouts, demo scenes, `watch`, the held-out report's recommendation and `compare` were removed at R1 with their commands; their findings stay in §10.

---

## 8. Repository layout

```
auxein_scenarios/
  common/                 # helpers shared by the scenarios
    cli.py                # the `scenarios` command: the registry of scenarios and their verbs; device resolution
    runs.py               # output directories: runs/<name>-<timestamp>, deployments/<name>-<timestamp>
    brains.py             # MLP genomes as safetensors files: layers as tensors, JSON metadata, the format check
    geometry.py           # 2-D geometry on batched arrays: frames, rectangle contacts, corners, ray casts
    rendering.py          # matplotlib + ffmpeg: figures, frame timing, MP4; the text a frame shows (for tests)
    charts.py             # line charts in the theme, PNG and SVG
    theme.py              # the look: colours, fonts, agent glyph, floor (§7.3)
    fonts/                # Inter and Inter Tight, with their licences
  box_pushing/            # this scenario
    cli.py                # its verbs: train and deploy
    world.py              # the physics core: layout, state, step, runner, trajectories (§2.10)
    contacts.py           # the penalty contact model, every pair (§2.6)
    environment.py        # BoxPushingWorld: run_batch, run_episode, the step loop, measurements, failures (§2.11); the brain
    sensors.py            # the observation layout and batched sensing (§3.8)
    scripted.py           # scripted controllers (true state) for tests; the senses-only controller (§10.1)
    scenarios.py          # the scenario generator, difficulties, training and held-out sets (§4)
    fitness.py            # measurements to score: the episode score and the Aggregator (§5.3)
    config.py             # the world's parameters (§2) and the TOML training config (§6.1)
    train.py              # train and resume: the run, progress.csv, brains/, held_out.csv, charts/ (§6.1)
    snapshots.py          # the snapshot generations and their champions (§7.1)
    brain.py              # a box-pushing brain as a file: its metadata and checks (§7.1)
    evaluation.py         # brains and baselines on scenarios: episodes, summaries, distance bands
    charts.py             # the run's four charts (§6.1)
    deploy.py             # deploy: brains on unseen scenarios, the baselines, results, summary, video (§7.2)
    render.py             # the deployment video's panels (§7.3)
configs/                  # training presets: smoke.toml, pilot.toml, full.toml (§6.1)
benchmarks/               # probe.py, run by hand (§10.3)
tests/                    # smoke, cli and standalone tests; common/, box_pushing/, benchmarks/
runs/                     # run directories (ignored by git): config.toml, progress.csv, held_out.csv, recording/, brains/, charts/
deployments/              # deployments (ignored by git): deploy.toml, results.csv, summary.md, deployment.mp4
docs/design/              # one design document per scenario
docs/auxein-issues.md     # the log of Auxein bugs and requests found here
```

**One entry point for every scenario** (decided at R1): `scenarios <scenario> <verb> [options]`, a console script (`common/cli.py`) with a registry, `SCENARIOS`: each scenario registers its name, a line saying what it is, and the module whose `VERBS` it runs (`box_pushing/cli.py`: `train` and `deploy`). The parser is built from them, so `scenarios --help` lists the scenarios and `scenarios <scenario> --help` its verbs; an unknown scenario or verb is argparse's "invalid choice", listing the choices. A verb imports Auxein, torch and matplotlib only when it runs. What every verb shares lives in `common/`: the parser, the device resolution (`backend_from`), the output directories (`runs.timestamped`) and the brain files (`brains.py`).

**Code, tests and configs stand on their own** (decided at R1): no pointer to this document, to `CLAUDE.md`, to milestones or to their decisions, and no history; where a pointer carried a reason, the reason is stated in place. A test (`tests/standalone_test.py`) searches for them. History and decisions live here.

**Adding a second scenario:** a subpackage `auxein_scenarios/<name>/` with a `cli.py` declaring its `VERBS`, an entry in `common.cli.SCENARIOS`, its design document `docs/design/<name>.md`, its tests `tests/<name>/`, and an entry in the README's scenario list. A helper starts in its scenario and moves to `common/` when a second scenario needs it, not before.

**How the layout grew:** M1a added the physics core and `scripted.py`, M1b `environment.py` (keeping the physics core free of sensors and networks), M2a the fitness, the config and training, M3 the held-out report, clips, montages and charts, M2b generator versions and `compare`. **R1 pruned it to two verbs**: it removed the `box-pushing` script and its other commands (`demo`, `brains`, `senses`, `scenarios`, `watch`, `report`, `milestones`, `montage`, `charts`, `showcase`, `compare`) with `demos.py`, `report.py`, `showcase.py`, `compare.py`, the sensing overlay, the galleries, the montage pieces, generator version 1 and `benchmarks/physics_timing.py`; it added `brain.py`, `evaluation.py` (the report's evaluation of brains and baselines, kept), `charts.py`, `deploy.py`, `common/brains.py`, `common/cli.py` and `common/runs.py`. The package went from 6,881 to 5,386 lines.

---

## 9. Milestones

- **M0** (done): the design document and the repository scaffold.
- **M1** is split in two, each ending with something to run and watch:
  - **M1a** (done): the physics core (bodies, contacts, friction, integration, delivery), the scenario generator, and a viewer: MP4 demo clips driven by scripted controllers that use the true state, and PNG galleries of layouts; physics tests (§2.10); a quick timing of the physics (§10.3).
  - **M1b** (done): sensors and the observation layout (§3.8), the Auxein environment `BoxPushingWorld` with both paths, the layout cache, measurements and episode failures (§2.11, §5.2), random `MLPPolicy` brains and a senses-only controller acting on their own senses through the same step loop, the sensing overlay, determinism end to end, and the full performance probe (§10.3).
- **M2** is split in three:
  - **M2a** (done): the real fitness (§5.3), the TOML config and presets, training with resume and extension, a viewer for a run's current best, the probe's fixes and the Metal numbers (§10.3), and a pilot run (§10.7).
  - **The owner's overnight run** (done) on the M4: `configs/full.toml`, 400 generations, 3.2 hours (§10.8).
  - **M2b** (done): **better scenarios, nothing else**: the box clearance and generator versions (§2.2), 4 easy + 28 full training scenarios (§4.2), and comparing runs on one held-out set; the bridge and the new pilot (§10.9). The report had recommended looking first at what stops the team (the push, the search, the step limit); M2b instead changes only the data, so that the next run says whether better data alone closes the gap.
  - **The owner's second overnight run** (done): `configs/full.toml` (generator 2, 4 + 28), 400 generations, 6.2 hours (§10.9). Better data closed the generalisation gap (+0.24 to +0.06) and ended the wall-hugging, but the champion delivers 41 % of its held-out scenarios, below half and below the first champion's 47 % on the same layouts; it stopped improving at generation 288.
- **M3** (done): milestones, the held-out report with two baselines and a recommendation, milestone clips with honest labels, the sequence and grid montages, the charts, and a command making them all (§7, as it was then); run on the overnight run (§10.8). R1 replaced its commands with `train`'s brains, held-out results and charts and `deploy`'s video.
- **R1** (done): **pruned to two verbs before the next experiment**: `scenarios box-pushing train` (the run, its brains, its held-out results, its charts) and `scenarios box-pushing deploy` (brains on scenarios they have not seen: results and one video), brains as safetensors files, a smaller config, one generator, a shared entry point for every scenario, and code that stands on its own (§6.1, §7, §8). Behaviour unchanged: a run trains to the same numbers as before (decision 9, pinned from `main`).
- **Next: the time-bonus weight for the push** (`[fitness] time`, `w_t`, §5.1), on the new CLI: a pilot with a larger weight, then a full run, its snapshot brains deployed together. Not M4 yet: the second run's champion rings the box and pushes at half the hand-written controller's speed, and delivers none of the long pushes (§10.9). One lever at a time; the sensing range for the search is no longer indicated (every layout is reached). The overnight runs predate R1 and are dropped, but their numbers stay comparable: a run with the default `scenarios.seed = 1` has held-out seed 2, the very 64 layouts both overnight champions were compared on (§10.9), so `deploy --seed 2` of a new brain is judged against the same 41 % and 47 %.
- **M4:** the CMA-ES comparison and polished clips for the website, once a retrained run generalises (at least half of the held-out scenarios delivered; the second overnight run delivers 41 %, §10.9).

---

## 10. Risks and open questions

### 10.1 Scattered starts with a memoryless brain

At full difficulty agents start scattered across a 20 × 20 arena and sense the box only within `r = 5` units: many start out of range, **knowing where the target is but not the box**, and a feed-forward brain cannot remember where it has been. It must find a search behaviour that is a pure function of its senses (follow walls, spread out, drift toward teammates who are near something). The levers, in order: the **mixed-difficulty training set** (easy scenarios start agents in sight of the box, §4.2), the **sensing range** (larger `r`, or rays that see the box farther than the cue), the teammate densities (a crowd is a cue), and as a fallback **recurrent brains** (§3.6). The first runs (M2) will tell whether full-difficulty scenarios are ever solved.

**What the senses-only controller showed (M1b).** `scripted.SensesController` is a hand-written, memoryless policy that sees only the 43 inputs, the same for every agent: when the box cue is on it rebuilds the box's and the target's positions from distances and bearings, and the box's extent from its relative heading, and pushes from behind the box (or goes around it to get there); when only a ray sees the box it steers to that ray; otherwise it **searches by orbiting the target at 9 units**, which passes within sensing range of any box the generator places 6 to 12 units from the target. It is not meant to be good, and it ignores obstacles beyond a little avoidance. On the default training set (`uv run box-pushing senses`, numpy float64, 600 steps):

| team | seed | easy delivered | full delivered |
|---:|---:|---:|---:|
| 20 | 1 | 6 / 8 | 3 / 24 |
| 20 | 2 | 7 / 8 | 1 / 24 |
| 40 | 1 | 8 / 8 | 8 / 24 |
| 40 | 2 | 8 / 8 | 10 / 24 |

- **The senses carry what the task needs.** A memoryless policy delivers most easy scenarios and some full ones from scattered starts, so evolution has something to find; nothing is missing from the observation for M2 to start.
- **Search can be memoryless here**, because the target is always known and the box's distance from it is bounded: orbiting the target is a pure function of the target's distance and bearing. Evolution may or may not find it; the levers of the first paragraph stay.
- **Teammates can be a trap.** A first version turned searching agents toward their teammates (a crowd may be at the box); agents far from the box instead trapped each other in small spinning clusters, and it delivered 1 of 24 full scenarios. Random and early brains may fall into the same attractor.
- **A larger team does much better** (more searchers, more pushers): 40 agents delivered three times as many full scenarios as 20. Together with the step limit (§10.2), that argues for 40 agents in M2 if the cost allows (§10.3).
- **What stops it:** easy scenarios it misses end 60 % of the way, out of time (the tight step limit of §10.2); full ones fail on boxes against walls or obstacles (it cannot get behind them), on searchers piling up along a wall where the orbit meets it, and on time.
- **Random brains** (`uv run box-pushing brains`, six on an easy scenario, 20 agents): they wander, circle and crowd; several bring agents to the box (up to 8 touching at once) and one moved it a little, none delivered: lively enough for a "clueless start", with a gradient for the approach term.

### 10.2 Physics

- **The rotating box against rectangles** (§2.6): corner penetration can give a wrong normal near a corner–corner contact, and a box wedged between two obstacles can jitter or be pushed through a thin one. Mitigations: obstacles no thinner than 0.5 units, the stiffer box contact. **M1a:** the tests and the `obstacles` demo show no pass-through (a crowd pushing the box into a 0.5-thick rectangle, the box into a corner); corner–corner overlaps count twice (a stiffer contact, not a wrong one). Not yet seen: a box wedged between two obstacles, which the generator's clearance makes rare.
- **The friction threshold** (§2.4): with penalty contacts the push on the box is the contact spring force, which oscillates around the agents' thrust. **M1a:** no stick-slip at the threshold in the tests (`k − 1` hold it, `k` move it steadily, for `k` = 2 to 4), and no hysteresis was needed. Two changes were (§2.4): `μ` from 0.8 to 0.9, and a kinetic torque. Force and torque still have separate thresholds, so `k − 1` agents pushing at one end can pivot the box (a torque above `τ_s` with a force below `F_s`), as a real box pivots; once it turns, the kinetic torque stops it as soon as they stop.
- **Impacts may be an exploit** (§2.4): an agent ramming the box at full speed nudges it a few centimetres even alone, as a real box would. Ramming in turn is slow (by estimate, about 0.05 units/s for one agent against 0.3 to 1 for a pushing team), but evolution finds slow strategies too. To watch in the first runs (M2); the levers are a lower top speed, a heavier box, or a cost for collisions in the fitness.
- **The step limit may be tight.** With the friction as built, `k` agents barely move the box: four agents in line push `4 F_a = 8` against kinetic friction `6.3`, about 0.3 units/s, so a 10-unit full-difficulty delivery takes about 30 s, the whole episode (600 steps at 0.05 s), before finding the box. Faster needs more pushers in line (5 give about 0.55 units/s, 6 about 0.7), which is the cooperation the scenario is about, but it shrinks the margin for clueless early generations. The naive scripted team delivers 10 of 16 easy and 5 of 16 full scenarios in time (§2.10); with the first version of the friction (no kinetic torque), it delivered all 16 easy ones, because a spinning box never came to rest and the team never had to beat the static threshold again. Levers for M2: a longer step limit, a smaller `k`, a smaller margin between `F_k` and `k F_a`, or a higher top speed. **M2a raised the step limit to 900** (§4.4) and kept `k = 4`.
- **Frictionless contacts limit oblique pushes** (§2.6): agents push only along a face's normal and slide off when they push at an angle. The demos show the scripted team stalling when it pushes the box diagonally; evolved teams will have to push square to a face, or turn the box first. Agent–box friction is the lever if that proves too hard.
- **Stiffness against the time step:** penalty contacts need a small enough step; too soft and agents sink into the box, too stiff and the integration blows up (a NaN measurement fails the candidate, which Auxein records as `FAILED`: a canary, but one that would punish brains for the physics' fault). Substeps (§2.1) are the lever. **M1a:** `WorldParams` refuses unstable combinations (§2.6), and a 2,000-step run of 100 agents crowding the box with random actions stays finite in float32.

### 10.3 Cost

**A generation should take about a minute for hours-long runs** (300 generations in 5 hours). Agent pairs are `n · s · N²` per substep, so the cost grows with the square of the team.

**Measured at M1a** (`uv run python -m benchmarks.physics_timing --device mps`): the physics core alone (contacts, friction, integration; no sensing, no policy), 100 candidates × 32 scenarios, 2 substeps per step, on an Apple M4 Pro (24 GB, torch 2.14 with 12 CPU threads). A generation is 600 steps (an upper bound: delivered scenarios still cost compute).

| agents | numpy float32 | torch float32, CPU | torch float32, Metal |
|---:|---:|---:|---:|
| 20 | 0.093 s/step, 56 s/generation | 0.073 s, 44 s | 0.016 s, **9 s** |
| 40 | 0.313 s, 188 s | 0.098 s, 59 s | 0.029 s, **17 s** |
| 100 | 2.010 s, 1,206 s | 0.342 s, 205 s | 0.122 s, **73 s** |

The earlier back-of-envelope (about 2 minutes per generation on numpy at 40 agents) was close (3 minutes). **Torch on Metal is 4 to 16 times faster than numpy**, and keeps 40 agents within the budget with room for sensing and the policy, and even 100 agents near it; numpy is fine for 20 agents and for tests. Sensing (M1b) adds another pairwise term (teammate sectors) and the rays, so the full probe (M1b) decides the defaults: population, scenarios per generation, team size and backend.

**The full probe (M1b)**, `uv run python -m benchmarks.probe`: the whole episode loop, sensing, the brain and the physics, with random brains sampled from `MLPPolicy.space()` (the GA's first generation, which never delivers, so every one of the 600 steps is simulated: an upper bound), the scenarios a quarter easy and the rest full. Each part is timed on its own, with the device synchronised around it; "loop" is `BoxPushingWorld.run` itself (which also tracks the measurements, the non-finite check and the early stop), and a generation is the loop's time per step × 600. Peak memory is numpy's, through `tracemalloc` (torch does not report it on the CPU). On the CPU of the same Apple M4 Pro (torch 2.14 with 8 CPU threads), in seconds per step:

| backend | agents | n × s | sensing | brain | physics | loop | **s / generation** | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numpy float32 | 20 | 50 × 16 | 0.046 | 0.002 | 0.028 | 0.060 | **36** | 56 |
| torch float32 CPU | 20 | 50 × 16 | 0.032 | 0.002 | 0.019 | 0.048 | **29** | |
| numpy float32 | 20 | 100 × 16 | 0.136 | 0.005 | 0.063 | 0.138 | **83** | 111 |
| torch float32 CPU | 20 | 100 × 16 | 0.053 | 0.003 | 0.039 | 0.089 | **53** | |
| numpy float32 | 20 | 100 × 32 | 0.284 | 0.007 | 0.110 | 0.311 | **187** | 223 |
| torch float32 CPU | 20 | 100 × 32 | 0.145 | 0.006 | 0.115 | 0.216 | **130** | |
| numpy float32 | 40 | 50 × 16 | 0.133 | 0.003 | 0.075 | 0.189 | **113** | 154 |
| torch float32 CPU | 40 | 50 × 16 | 0.076 | 0.004 | 0.052 | 0.110 | **66** | |
| numpy float32 | 40 | 100 × 16 | 0.280 | 0.007 | 0.165 | 0.396 | **238** | 308 |
| torch float32 CPU | 40 | 100 × 16 | 0.137 | 0.005 | 0.095 | 0.233 | **140** | |
| numpy float32 | 40 | 100 × 32 | 0.619 | 0.013 | 0.324 | 0.905 | **543** | 617 |
| torch float32 CPU | 40 | 100 × 32 | 0.194 | 0.009 | 0.136 | 0.337 | **202** | |
| numpy float32 | 100 | 50 × 16 | 0.671 | 0.008 | 0.481 | 1.057 | **634** | 875 |
| torch float32 CPU | 100 | 50 × 16 | 0.175 | 0.003 | 0.149 | 0.343 | **206** | |
| numpy float32 | 100 | 100 × 16 | 1.208 | 0.019 | 0.831 | 1.926 | **1,156** | 1,750 |
| torch float32 CPU | 100 | 100 × 16 | 0.239 | 0.008 | 0.210 | 0.459 | **275** | |
| numpy float32 | 100 | 100 × 32 | 2.662 | 0.031 | 2.268 | 4.335 | **2,601** | 3,500 |
| torch float32 CPU | 100 | 100 × 32 | 0.400 | 0.019 | 0.380 | 0.815 | **489** | |

- **Sensing costs as much as the physics, or more** (about twice on numpy at 20–40 agents): the rays are `(n, s, N, 12)` arrays cast against every padded obstacle slot, and the teammate sectors a pass over the `(n, s, N, N)` pairs per sector. The brain is negligible (1–3 %). If sensing has to get cheaper, the levers are fewer rays, casting rays only against the obstacles within range, and the sectors from one pass (an integer sector index and eight masked sums share the pairs already).
- **Torch on the CPU is 1.2 to 5 times faster than numpy**, more as the team grows (numpy is single-threaded for these element-wise operations).
- **The probe's recommendation on the CPU:** torch float32 on the CPU, **20 agents, a population of 100, 16 scenarios per generation: 53 s per generation**. Nothing at 40 agents fits a minute on the CPU (66 s at 50 × 16 is the nearest). numpy stays the backend of tests and clips.
- **Metal (the owner's run on the M4, at M2a):** `uv run python -m benchmarks.probe --device mps --no-cpu-torch`, at 600 steps per generation:

| backend | agents | n × s | sensing | brain | physics | loop | **s / generation** | peak MB (see below) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numpy float32 | 20 | 50 × 16 | 0.038 | 0.002 | 0.024 | 0.056 | **34** | 56 |
| torch float32 Metal | 20 | 50 × 16 | 0.006 | 0.000 | 0.015 | 0.047 | **28** (warm-up, below) | 57 |
| numpy float32 | 20 | 100 × 16 | 0.104 | 0.004 | 0.051 | 0.129 | **77** | 111 |
| torch float32 Metal | 20 | 100 × 16 | 0.009 | 0.001 | 0.015 | 0.020 | **12** | 121 |
| numpy float32 | 20 | 100 × 32 | 0.221 | 0.006 | 0.093 | 0.284 | **170** | 223 |
| torch float32 Metal | 20 | 100 × 32 | 0.058 | 0.001 | 0.018 | 0.027 | **16** | 1,177 |
| numpy float32 | 40 | 50 × 16 | 0.122 | 0.003 | 0.067 | 0.190 | **114** | 154 |
| torch float32 Metal | 40 | 50 × 16 | 0.008 | 0.001 | 0.016 | 0.019 | **12** | 1,177 |
| numpy float32 | 40 | 100 × 16 | 0.254 | 0.006 | 0.144 | 0.383 | **230** | 308 |
| torch float32 Metal | 40 | 100 × 16 | 0.015 | 0.001 | 0.016 | 0.032 | **19** | 1,177 |
| numpy float32 | 40 | 100 × 32 | 0.570 | 0.014 | 0.312 | 0.806 | **484** | 617 |
| torch float32 Metal | 40 | 100 × 32 | 0.033 | 0.001 | 0.029 | 0.068 | **41** | 1,193 |
| numpy float32 | 100 | 50 × 16 | 0.534 | 0.007 | 0.390 | 0.821 | **492** | 875 |
| torch float32 Metal | 100 | 50 × 16 | 0.028 | 0.001 | 0.034 | 0.063 | **38** | 1,193 |
| numpy float32 | 100 | 100 × 16 | 1.147 | 0.018 | 0.769 | 1.730 | **1,038** | 1,750 |
| torch float32 Metal | 100 | 100 × 16 | 0.057 | 0.001 | 0.062 | 0.136 | **81** | 2,225 |
| numpy float32 | 100 | 100 × 32 | 2.516 | 0.028 | 2.088 | 4.386 | **2,632** | 3,500 |
| torch float32 Metal | 100 | 100 × 32 | 0.126 | 0.003 | 0.122 | 0.255 | **153** | 3,249 |

  Its recommendation: torch on Metal, 100 agents at a population of 50 and 16 scenarios (38 s per generation); also within the budget 40 agents at 100 × 32 (41 s) and at 100 × 16 (19 s).

- **Two flaws in that table, fixed at M2a** (`benchmarks/probe.py`):
  - **A warm-up artefact.** The first Metal row (20 agents at 50 × 16, 28 s) is slower than larger ones (40 agents at 50 × 16, 12 s): it paid for Metal's start-up, kernels compiling on first use. Every backend is now warmed up on a small configuration before anything is timed, and each configuration runs its whole loop once untimed.
  - **The memory column on Metal** was not the GPU's tensors: it read `driver_allocated_memory`, which includes torch's cache and so grew with the configurations before (1,177 MB from the third row on). It now reports, from `torch.mps`, the peak of what tensors hold during the loop and what the driver holds at its end after emptying the cache first, as `allocated / driver`.
- **Decided at M2a: 40 agents at a population of 100, with 16 scenarios** (19 s per generation at 600 steps, about 29 s at 900), **not the probe's recommendation** of 100 agents at a population of 50. The probe ranks by team size first, but the senses-only controller showed that 40 agents already deliver three times as many full scenarios as 20 (§10.1), and a GA learns more reliably with a population of 100 than of 50 on a 1,474-dimensional genome; 100 agents would also quadruple the agent pairs for a team the arena barely holds. The overnight run (400 generations) is then about 3 hours at most.

### 10.4 Recording size

Auxein records every evaluated genome and every episode's measurements. A genome of 1,474 values is 5.9 KB in float32 (11.8 KB in float64), above Auxein's genome-store threshold (4 KB), so each distinct genome is a blob in `events.sqlite`; the GA's children are all distinct, so the store does not shrink them. A run of 300 generations of 100 children is 30,000 genomes, about **180 MB** in float32, plus 30,000 × 32 = 960,000 episode rows of six measurements (about 200 bytes each with keys and index), about **200 MB**: **roughly 0.4 GB per run** (twice that in float64). It matters for disk on a laptop but not for speed (Auxein's reader indexes a 1 GB recording in 0.5 s, core design §10.2), and run directories are never committed (`.gitignore`). Fewer measurements or scenarios are the levers if it does.

### 10.5 What reading Auxein turned up

- **`v0.3.1` reported itself as version `0.3.0`** (fixed in `v0.3.2`). The `v0.3.1` tag's `pyproject.toml` was not bumped, so `auxein.__version__` was `"0.3.0"` when installed from it. Logged in `docs/auxein-issues.md`; the scenarios moved to `v0.3.2` (now `v0.3.3`), and the smoke test checks the tag, the commit (PEP 610 metadata) and the version.
- **Auxein's test fixtures are not installed with it** (`tests/support/` is outside the wheel), so the minimal reproductions the protocol asks for are written in a clone of Auxein, not here.
- **M1b drove Auxein's evaluator and driver for real and found no bug.** `EpisodeEvaluator` took the batched path on its own (`MLPPolicy.decode_batch` and `run_batch`) and the per-episode path with a decoder without `decode_batch`; the two agree; a recorded two-generation `auxein.run` with `RandomSearch` completed on numpy float64 and torch float32, with the six measurements per episode in the recording; a NaN action became a failed episode and a failed candidate, as core design §6.6 says.
- **Feature request: a batched environment cannot learn the batch's size or backend from the protocol** (logged in `docs/auxein-issues.md`). `run_batch` must build its state for `n` candidates on the run's device, but `agents` is opaque and `BatchPolicy` has only `act_batch`. The world reads `MLPAgentBatch`'s `weights` and `n` (as Auxein's own team fixture does) or a policy's `backend` and `n`, a convention for the scenarios' own policies. `n` and `backend` on `BatchPolicy`, or a documented guarantee that `run_batch`'s `rng` is on the run's backend, would remove the convention.
- **Held-out evaluation during a run** (the early stop, §6) would call `evaluate_held_out` on a run directory that the driver is still writing. M2a decided not to (the run stops on its budget only), so it stays unchecked; held-out results come after the run (M3).
- **M2a drove the GA, checkpoints and resume for real, and found two bugs** (both logged in `docs/auxein-issues.md`, fixed in Auxein and released in `v0.3.3`, to which the scenarios moved, removing the workarounds):
  - **A parameterised aggregator source hid its parameters.** A reduction labelled a callable source by its `__qualname__`, so a callable object or a `functools.partial` showed its class's name whatever its parameters (`mean(<EpisodeScore>)`), and resuming a run with changed fitness weights was accepted. Fix: `fix/parameterised-source-labels` (a source is labelled by its `describe()`, its own `__repr__`, or a partial's function and arguments), released in `v0.3.3`. The workaround (`EpisodeScore` and `EpisodeProgress` carried their description as an instance `__qualname__`, the label the fix gives) was removed with the upgrade, and runs recorded with it stay resumable; the training command also compares the whole config on resume.
  - **One Ctrl-C did not stop a batched run.** `asyncio.run` turns Ctrl-C into a cancellation of the driver's task, delivered only when the task yields to the event loop, and the live loop yielded only while replaying, so a batched evaluator ran on to the end of the budget and only a second Ctrl-C stopped it, in the middle of a generation. Auxein's interrupt tests raised `KeyboardInterrupt` from code, never a real SIGINT. Fix: `fix/ctrl-c-between-batches` (the driver yields after every tell), released in `v0.3.3`. The workaround (`train.StopOnCtrlC`, a SIGINT handler of the scenario's own and a callback that raised `KeyboardInterrupt` after the generation's tell) was removed with the upgrade.
- **What worked as documented:** the GA in generation delivery with `offspring_size` fixed (so a budget extension is exact); interrupting after a generation and resuming, and extending a finished run, both give the event log of the run that never stopped; resume refuses a changed world or (from `v0.3.3`) changed weights; `RunReader` reads a recording that another process is writing (SQLite in WAL mode), so `watch` and `progress.csv` work during a run; a callback raising `KeyboardInterrupt` after a tell gives an interrupted run with a checkpoint.
- **M3 drove `evaluate_held_out` for real, and found no bug.** With `candidates=[ids]` on a run directory it found the nine
  milestone champions among 40,000 recorded candidates (iterating the recording takes 2 to 4 s), evaluated them in one
  array batch on the batched path (Metal, 28 s for 9 × 64 episodes), with the run's backend read from the metadata (or the
  one passed), and its per-scenario measurements and aggregated values agree with the scenario's own summaries to 1e-12 on
  numpy (a test). With `write=False` nothing touched the recording (a byte-for-byte test). **One feature request** (logged):
  `write` is a flag that writes `held_out.json` into the run directory it is given, which here is the recording; the report
  keeps everything it writes in `DIR/report/`, so it passes `write=False` and writes `HeldOutReport.to_json()` itself. A path
  for `write` would let Auxein write its own file there.
- **A run directory must be empty or new** for Auxein to start in it, so the scenario's own files (config, progress, clips) sit beside Auxein's recording, in `recording/`, rather than in it (§6.1). Not a bug: it is how Auxein keeps runs from mixing.

### 10.6 Proposed details: decided at M1a

Details this design needed but was not first given. **Items 1–10 and 14 were decided at M1a** as written, except 7 and 8, which were decided as their alternatives; **item 12 was decided at M1b**, **item 11 at M2a**; item 13 stays open (M4).

1. **Decided.** Time step and substeps (§2.1): `dt = 0.05` s per episode step, 2 physics substeps.
2. **Decided.** Obstacle sizes and the validity check (§2.2): the grid flood fill with obstacles inflated by the box's circumscribed radius (conservative).
3. **Decided, with two changes found at M1a.** The friction threshold `F_s = (k − 0.5) F_a` and the torque threshold `τ_s = F_s (w + h) / 6`. Kinetic friction is `F_k = 0.9 F_s` (not 0.8, which let `k − 1` agents keep a moving box moving), and a kinetic torque `τ_k = 0.9 τ_s` was added (without it `k − 1` agents kept a jolted box turning) (§2.4).
4. **Decided.** Agent dynamics (§2.5): reverse at half thrust, a first-order lag on turning, strong lateral damping.
5. **Decided.** Frictionless contacts between bodies, and the corner-penetration model for box–rectangle (§2.6). Added at M1a: a separate, stiffer contact for the box against walls and obstacles, and damping as a ratio of critical.
6. **Decided.** Input normalisation (§3), **rays do not see teammates** and report 1 for the box, 0 otherwise (§3.2), **the ray range equals the sensing range** (§3.3).
7. **Decided as `(sin 2θ, cos 2θ)`.** The box's relative heading (§3.3): the box is symmetric under a half turn, so `(sin θ, cos θ)` would give two different inputs for the same situation. The observation size stays 43. Implemented at M1b (§3.8).
8. **Decided as squashed.** Teammate density (§3.4): each sector's value is `1 − e^(−x)` of the closeness-weighted count `x = Σ (1 − d / r)`, in `[0, 1)` whatever the team size, so "the same brain for any team size" holds for the values as well as the shape. Implemented at M1b (§3.8).
9. **Decided.** Scenario params hold the whole layout (§4.1), and **no noise** in the world in v1 (sensor or actuator noise may help generalisation to unseen layouts).
10. **Decided.** Full-difficulty box–target distance 6–12 units, **the training mix** 8 easy + 24 full, **64 held-out scenarios**, **target zone radius** 1.5 (§4).
11. **Decided at M2a.** Progress is clipped at −1; the approach term (the mean over agents of `1 − min(1, g / r)`, with `g` the closest gap to the box, §5.2) counts at weight 0.1 for the whole run and does not fade (an aggregator is fixed for a run; a staged run could fade it later). A team that only gathers scores at most 0.1, less than one whose push has got going, but more than one that only nudges the box while half of it hangs back (§5.3).
12. **Decided at M1b.** `max_pushers` is a recorded measurement (§5.2), for the "cooperation emerging" chart: cheap, and built. It counts agents touching the box, not agents pushing through a queue.
13. **Open (M4).** CMA-ES at 1,474 dimensions (§6): full-covariance CMA-ES learns its covariance at a rate of about `1/d²`, so in a few hundred generations it barely adapts it, and its `O(d²)` state and eigendecompositions run on the host. The separable variant (pycma's `CMA_diagonal` option, through `PycmaStrategy(options=...)`) is the fairer comparison. `PycmaStrategy` writes no checkpoints, so a resumed comparison run replays from its start.
14. **Decided.** `contacts.py` and `config.py` as modules (§8); M1a also added `scripted.py` and `demos.py`.

### 10.7 Pilot run (M2a)

`uv run box-pushing train configs/pilot.toml`: the full sizes (40 agents, a population of 100, 16 training scenarios, 4 easy and 12 full, 900 steps), 25 generations, torch float32 on Metal (the M4 Pro), seed 1. **12 minutes, 29 s per generation** (the start line's simulation-only estimate said 23 s; recording, aggregation and the GA add the rest). The recording is 32 MB for 2,500 evaluations, so the overnight run's 40,000 will be about 0.5 GB (§10.4). From `runs/pilot/progress.csv`:

| generation | best so far | generation's best | generation's mean | mean success | mean progress | mean `max_pushers` | champion: success, progress |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.111 | 0.111 | 0.034 | 0.000 | +0.000 | 2.0 | 0 %, +0.06 |
| 4 | 0.332 | 0.332 | 0.043 | 0.001 | +0.001 | 2.7 | 6 %, +0.12 |
| 6 | 0.411 | 0.411 | 0.052 | 0.003 | +0.003 | 3.1 | 13 %, +0.05 |
| 9 | 0.652 | 0.652 | 0.055 | 0.002 | +0.004 | 3.3 | 19 %, +0.12 |
| 11 | 0.772 | 0.772 | 0.072 | 0.006 | +0.010 | 3.6 | 19 %, +0.24 |
| 13 | 0.892 | 0.892 | 0.080 | 0.009 | +0.011 | 4.0 | 25 %, +0.20 |
| 20 | 0.892 | 0.650 | 0.093 | 0.013 | +0.016 | 3.7 | 25 %, +0.20 |
| 23 | 0.892 | 0.847 | 0.141 | 0.023 | +0.036 | 4.2 | 25 %, +0.20 |
| 25 | **0.912** | 0.912 | 0.104 | 0.013 | +0.024 | 3.8 | **25 %, +0.22** |

**Evolution is making progress.** The champion's score rose from 0.11 (the best random brain, gathering at best) to 0.91 in 25 generations, in steps (generations 2, 4, 6, 9, 11, 13, 25), and the population follows: its mean score tripled (0.034 to 0.10), its mean `max_pushers` doubled (2.0 to 3.8, "cooperation emerging", §1) and its mean approach rose from 0.34 to 0.49. **The champion delivers all 4 easy scenarios** (at steps 242 to 673, with 9 to 12 agents touching the box) **and none of the 12 full ones**, though it moves the box up to 41 % of the way in some (a mean progress of 0.22 over all 16). The clip (`uv run box-pushing watch runs/pilot`) shows it: in the easy scenarios the team gathers into a crowd behind the box and pushes it into the target; in the full ones the agents spread across the arena, many along the walls, and the few that find the box push it, slowly, with 5 to 7 touching. On the held-out scenarios it does the same: it finds and moves some boxes, delivers none.

**What it suggests:**

- **The easy scenarios are learned first, as designed** (§4.2). The full ones are where the run has to go: finding the box from scattered starts (§10.1) and pushing it far enough in 900 steps. The champion's progress on full scenarios (up to 0.41) says it gets partway; whether 400 generations get it the rest of the way is what the overnight run will tell.
- **The champion stalled from generation 13 to 24** while the population's mean kept rising: the GA was filling in behind its best, not stuck. A plateau of a few dozen generations in the overnight run is no cause for alarm; one of a hundred or more would be.
- **The approach term did not trap it.** The population's approach rose with its progress, and the champions moved the box from generation 1 on (§5.3's valley between gathering and nudging did not show up as a plateau at 0.1).
- **On Metal, a clip of one candidate is close to, not identical with, the batch that scored it** (in one full scenario `watch` gave progress +0.26 against +0.24 recorded; the easy one was identical): float32 on a GPU depends slightly on the batch's shape (§2.11, §7). The clips show the brain's behaviour faithfully; exact scored numbers come from the recording.

**What to watch for in the overnight run** (`configs/full.toml`):

- **Ramming** (§10.2): a box that moves in jolts, with few agents touching it, while `max_pushers` stays low and progress rises; a lever for M2b if it appears (a lower top speed, a heavier box, a cost for collisions).
- **Oblique pushes** (§2.6, §10.2): teams pushing at a face at an angle, sliding off and losing the box; the clip shows the box turning instead of moving.
- **Agents trapping each other** (§10.1): clusters of agents circling one another far from the box, in full scenarios; the M1b senses-only controller fell into it when it turned toward teammates.
- **Wall-hugging searchers**: in the pilot's clip many agents ended along the walls in full scenarios; if the champion's full-scenario progress stays low, the search, not the push, is the bottleneck (the levers of §10.1: a longer sensing range, rays that see the box farther).
- **The step limit**: full-scenario episodes that end with the box moving toward the target (progress rising until step 900) mean 900 steps are still tight (§10.2); a longer limit is M2b's lever.
- **The success rate on full scenarios** (`champion_success_rate` above 0.25 in `progress.csv` means a full scenario delivered) and **`max_pushers_mean`**, the cooperation chart.

### 10.8 Overnight run and held-out results (M3)

*These runs predate R1: their directories are dropped, generator 1 is removed (§2.2), and the commands named below (`box-pushing report`, `showcase`, `watch`, …) are replaced by `scenarios box-pushing train` and `deploy` (§6.1, §7.2). Their numbers stay here as history.*

**The run.** `configs/full.toml` on the owner's M4 Pro: 40 agents, a population of 100 with 100 children per generation, 4 easy
and 12 full training scenarios, 900 steps, **400 generations**, torch float32 on Metal, seed 1 (`runs/full-20261017-201610`).
**3.2 hours, 29.2 s per generation** (the pilot's rate), 40,000 evaluations, a **507 MB** recording (§10.4 estimated 0.4 GB).

**The learning curve** (`progress.csv`):

| generation | best so far | generation's best | mean | median | mean success | mean `max_pushers` | champion: success, progress, `max_pushers` |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.111 | 0.111 | 0.034 | 0.031 | 0.000 | 2.0 | 0 %, +0.06, 5.8 |
| 10 | 0.652 | 0.554 | 0.075 | 0.046 | 0.006 | 3.4 | 19 %, +0.12, 6.4 |
| 25 | 0.912 | 0.912 | 0.104 | 0.050 | 0.013 | 3.8 | 25 %, +0.22, 6.1 |
| 50 | 1.287 | 1.241 | 0.183 | 0.068 | 0.036 | 4.7 | 38 %, +0.33, 6.6 |
| 100 | 1.614 | 1.329 | 0.229 | 0.068 | 0.044 | 4.8 | 44 %, +0.53, 8.8 |
| 200 | 1.967 | 1.909 | 0.473 | 0.160 | 0.106 | 5.5 | 56 %, +0.59, 6.8 |
| 300 | 2.301 | 1.823 | 0.439 | 0.124 | 0.094 | 5.2 | 69 %, +0.62, 8.2 |
| 400 | **2.509** | 2.293 | 0.713 | 0.467 | 0.170 | 5.8 | **75 %, +0.66, 8.2** |

The champion improved 27 times, in steps, the last at generation 384, so the run was **still improving when it stopped**. Its
plateaus lasted up to 76 generations (273 to 349). The population followed: the mean score rose twentyfold (0.034 to 0.71),
the median from 0.03 to between 0.16 and 0.47 over the last 50 generations (it is noisy: most children are much worse than their
parents, §6), and the mean `max_pushers` from 2.0 to 5.8. The champion delivers **12 of 16 training scenarios: all 4 easy and 8
of the 12 full**.

**The held-out report** (`box-pushing report`, 64 full-difficulty scenarios never seen in training, Metal):

| | training score | easy delivered | full delivered | held-out delivered | held-out progress | held-out `max_pushers` | delivery step (median) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generation 1 | 0.111 | 0/4 | 0/12 | 0/64 | +0.00 | 4.8 | | +0.00 |
| generation 2 | 0.247 | 1/4 | 0/12 | 0/64 | +0.02 | 3.1 | | +0.00 |
| generation 5 | 0.332 | 1/4 | 0/12 | 0/64 | +0.06 | 4.9 | | +0.00 |
| generation 10 | 0.652 | 3/4 | 0/12 | 0/64 | +0.04 | 4.1 | | +0.00 |
| generation 20 | 0.892 | 4/4 | 0/12 | 0/64 | +0.04 | 4.8 | | +0.00 |
| generation 50 | 1.287 | 4/4 | 2/12 (17 %) | 4/64 (6 %) | +0.16 | 5.0 | 778 | +0.10 |
| generation 100 | 1.614 | 4/4 | 3/12 (25 %) | 6/64 (9 %) | +0.46 | 7.6 | 768 | +0.16 |
| generation 200 | 1.967 | 4/4 | 5/12 (42 %) | 13/64 (20 %) | +0.48 | 5.5 | 734 | +0.21 |
| **generation 400** | **2.509** | **4/4** | **8/12 (67 %)** | **27/64 (42 %)** | **+0.59** | 6.2 | 708 | **+0.24** |
| *20 random brains* | 0.033 | 0/80 | 0/240 | 0/1,280 | +0.00 | 0.9 | | |
| *senses-only controller* | 2.156 | 4/4 | 5/12 (42 %) | 28/64 (44 %) | +0.59 | 3.2 | 374 | −0.02 |

- **It generalises, partly.** On unseen layouts the final champion delivers 27 of 64 (42 %) and moves the box 59 % of the way
  on average. It keeps 63 % of its full-training success (67 %). It never pushes a box away (no held-out progress below 0).
- **The generalisation gap grows with training**: zero while nothing full is delivered (to generation 20), then 0.10, 0.16, 0.21
  and **0.24 at generation 400**, at decision 7's threshold of about 0.25. With only 12 full training layouts, part of what
  the later generations learned is the maps.
- **The baselines.** Random brains deliver nothing (0 of 1,280). **The hand-written senses-only controller (§10.1) does as well
  on unseen layouts (28 of 64) and delivers twice as fast** (a median step of 374 against 708). From random brains, evolution
  reached the level of a controller written by hand for these senses, in 400 generations, but not beyond it. The controller's
  speed says the step limit is not the only reason long pushes fail: the evolved team pushes slowly.
- **Distance decides.** The final champion's held-out deliveries by the box's starting distance to the target: 6–8 units
  15 of 22 (68 %), 8–10 units 9 of 22 (41 %), 10–12 units 3 of 20 (15 %). The undelivered ones end 43 % of the way on average.
- **Five held-out layouts never moved** (`held-0014`, `0019`, `0024`, `0038`, `0040`; and `train-0009` in training). In all six
  the box starts close to a wall or beside an obstacle, mostly in the arena's upper half; the clips (`--showcase worst-held`)
  are where M2b should start.

**What the clips show** (milestone clips, the montages and `watch`):

- **The clueless start, as designed** (generation 1): agents wander and crowd; some gather at the box but nothing moves it.
- **Cooperation emerges**: by generation 10 the team crowds the box in the easy scenarios, at first **overshooting** (it pushed
  `train-0000`'s box past the target and against the wall, progress −0.48); by generation 50 it delivers it.
- **No ramming** (§10.2): the box moves only with a crowd touching it (4 to 15 agents), never in jolts from a lone agent.
- **No agents trapping each other** (§10.1): no spinning clusters away from the box.
- **A crowd, not a line.** The final team surrounds the box on all sides rather than lining up behind it, so some agents push
  against the direction of travel: the likeliest reason it pushes at half the hand-written controller's speed.
- **Wall- and obstacle-hugging searchers.** In full scenarios, rows of agents sit along the top wall and along obstacles' edges
  for the whole episode: searchers that never find the box. The search, not only the push, limits the full scenarios.
- **Time running out**: undelivered full episodes end with the box partway (43 % on average).

**The recommendation** (the report's, decision 7): **M2b, on what stops the team on full scenarios**, before more training
scenarios: held-out success 42 % (below half), full-training success 67 %, a gap of +0.24 (not above 0.25). The gap is at the
threshold, so **more full training scenarios are indicated too**. Concretely for M2b, in order: how the team pushes (a crowd
around the box against a line behind it; the hand-written controller shows a faster push exists), the search (wall-hugging,
the layouts never moved), the step limit (long pushes), and then the training mix (e.g. 4 easy + 28 full, which costs about
twice the time per generation).

**How long it took** (`box-pushing showcase`, on the M4 Pro, Metal for the simulation; matplotlib on the CPU, while the test
suite also ran): the report **114 s** (the nine champions on 64 scenarios 28 s, the baselines on both sets most of the rest),
the nine milestone clips **570 s** (about a minute each, 1036 × 1216 px), the sequence montage **334 s** (1920 × 1080, 9
segments, 20 MB), the grid montage **64 s** (1920 × 1080, 3 MB), the charts under a second: about **18 minutes** in all.

### 10.9 Scenarios with clearance (M2b)

*As §10.8: these runs predate R1, and their commands (`box-pushing compare`, `showcase`, …) are replaced by `train` and `deploy`. The held-out set they were compared on (generator 2, seed 2, 64 layouts) is the held-out set of any new run with the default `scenarios.seed = 1`, so `deploy --seed 2` judges a new brain on the same layouts.*

M2b changes only the scenarios (§2.2, §4.2): fitness, sensing, the brain, the physics, the step limit and the GA settings are
M2a's, so the second overnight run says what better data alone does.

**The bridge: the overnight champion on solvable layouts.** Until a generator-2 run exists, the overnight run's final champion
(generation 400, trained on generator 1's 4 + 12) is compared on a generator-2 held-out set built from `pilot.toml` (64 full
scenarios, seed 2, the same seed as its own held-out set but drawn with the clearance):
`box-pushing compare runs/full-20261017-201610 --held-out configs/pilot.toml` (Metal, 2 minutes).

| held-out set | delivered | progress | never moved | undelivered end at | delivery step (median) | 6–8 units | 8–10 | 10–12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generator 1 (its own, §10.8) | 27/64 (42 %) | +0.59 | 5 | 43 % of the way | 708 | 15/22 | 9/22 | 3/20 |
| **generator 2** | **30/64 (47 %)** | **+0.66** | **0** | 52 % of the way | 690 | 17/24 | 11/22 | 2/18 |
| *senses-only controller*, generator 2 | 31/64 (48 %) | +0.65 | | | 392 | 15/24 | 8/22 | 8/18 |

- **The unpushable layouts were part of the 42 %, not most of it.** On layouts where agents can always get behind the box,
  the same champion delivers 30 of 64 (47 %), and **none is left unmoved** (5 were). It moves the box further (+0.66 against
  +0.59), so clearance helps the episodes it does not finish too.
- **Distance still decides**, now more sharply: 17 of 24 short pushes (6–8 units), 2 of 18 long ones (10–12). The hand-written
  controller, at the same success overall, delivers 8 of the long ones in about half the time: the evolved team's slow push,
  not the layouts, is what fails long distances (§10.8).
- So **the second overnight run has two things to show**: whether more full layouts (28) close the generalisation gap, and
  whether training on solvable layouts makes a better pusher. The bridge sets the bar: 47 % on generator-2 layouts for a
  champion of generator 1's 4 + 12.

**The new pilot** (`configs/pilot.toml`: 4 + 28 from generator 2, 25 generations, Metal) against M2a's pilot (§10.7, 4 + 12
from generator 1):

**23.4 minutes, 56 s per generation** (M2a's: 29 s), so 400 generations take about 6.2 hours; the start line's
simulation-only estimate said 40 s (a generation is 1.4 times its simulation at 32 scenarios, 1.26 at 16: the start line's
run estimate now uses 1.4). The recording is 41 MB for 2,500 evaluations, about 0.7 GB for 40,000.

| generation | best so far (M2a → M2b) | generation's mean | champion's training success | mean `max_pushers` |
|---:|---|---|---|---|
| 1 | 0.111 → 0.079 | 0.034 → 0.026 | 0 % → 0 % | 2.0 → 1.6 |
| 4 | 0.332 → 0.244 | 0.043 → 0.032 | 6 % → 6 % | 2.7 → 2.2 |
| 9 | 0.652 → 0.365 | 0.055 → 0.053 | 19 % → 9 % | 3.2 → 3.4 |
| 13 | 0.892 → 0.365 | 0.080 → 0.050 | 25 % → 9 % | 4.0 → 3.7 |
| 25 | 0.912 → 0.648 | 0.104 → 0.065 | 25 % → 16 % | 3.8 → 3.5 |

The scores do not compare directly: the easy scenarios are a quarter of M2a's training set and an eighth of M2b's, and they
are where a young team scores. In deliveries, **learning starts as before, and further on the full scenarios**: at generation
25 M2a's champion delivered its 4 easy scenarios and none of its 12 full ones (mean full progress +0.10); M2b's delivers its
4 easy ones and **1 of its 28 full ones** (+0.15). The population's cooperation (`max_pushers`) rises at the same pace.

**The second overnight run.** `configs/full.toml` on the owner's M4 Pro: 40 agents, a population of 100 with 100 children per
generation, **4 easy and 28 full training scenarios from generator 2**, 900 steps, **400 generations**, torch float32 on Metal,
seed 1 (`runs/full-20261018-232911`). **6.2 hours, 55.6 s per generation** (the pilot's 56 s; the start line's estimate held),
40,000 evaluations, a **625 MB** recording (the pilot's estimate: about 0.7 GB).

**The learning curve** (`progress.csv`):

| generation | best so far | generation's best | mean | median | mean success | mean `max_pushers` | champion: success, progress, `max_pushers` |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.079 | 0.079 | 0.026 | 0.023 | 0.000 | 1.6 | 0 %, +0.03, 4.2 |
| 10 | 0.365 | 0.238 | 0.056 | 0.041 | 0.003 | 3.3 | 9 %, +0.09, 6.9 |
| 25 | 0.648 | 0.341 | 0.065 | 0.040 | 0.006 | 3.5 | 16 %, +0.20, 6.8 |
| 50 | 0.934 | 0.693 | 0.178 | 0.087 | 0.030 | 4.6 | 25 %, +0.27, 5.4 |
| 100 | 1.308 | 1.241 | 0.167 | 0.066 | 0.024 | 4.4 | 34 %, +0.44, 6.8 |
| 200 | 1.689 | 1.227 | 0.293 | 0.161 | 0.056 | 4.7 | 47 %, +0.50, 5.4 |
| 300 | 1.887 | 1.475 | 0.328 | 0.176 | 0.061 | 4.9 | 53 %, +0.57, 7.9 |
| 400 | **1.887** | 1.467 | 0.423 | 0.230 | 0.087 | 5.4 | **53 %, +0.57, 7.9** |

The scores do not compare with the first run's (2.509): an eighth of the training set is easy, not a quarter. The champion
improved 17 times, **the last at generation 288**: unlike the first run, this one **had stopped improving**, with a plateau of
112 generations to the end (the first run's longest was 76). The population kept moving (the mean score 0.33 to 0.42, the
median between 0.10 and 0.33 over the last 50 generations, the mean `max_pushers` 4.9 to 5.4) without finding a better
champion. The champion delivers **17 of 32 training scenarios: all 4 easy and 13 of the 28 full**.

**The held-out report** (`box-pushing report`, 64 full-difficulty scenarios of generator 2 never seen in training, Metal):

| | training score | easy delivered | full delivered | held-out delivered | held-out progress | held-out `max_pushers` | delivery step (median) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generation 1 | 0.079 | 0/4 | 0/28 | 0/64 | +0.01 | 3.4 | | +0.00 |
| generation 2 | 0.244 | 2/4 | 0/28 | 0/64 | +0.03 | 5.3 | | +0.00 |
| generation 10 | 0.365 | 3/4 | 0/28 | 0/64 | +0.03 | 6.2 | | +0.00 |
| generation 20 | 0.426 | 4/4 | 0/28 | 0/64 | +0.04 | 6.1 | | +0.00 |
| generation 50 | 0.934 | 4/4 | 4/28 (14 %) | 1/64 (2 %) | +0.17 | 4.7 | 831 | +0.13 |
| generation 100 | 1.308 | 4/4 | 7/28 (25 %) | 5/64 (8 %) | +0.37 | 6.5 | 877 | +0.17 |
| generation 200 | 1.689 | 4/4 | 11/28 (39 %) | 21/64 (33 %) | +0.43 | 5.3 | 730 | +0.06 |
| **generation 400** | **1.887** | **4/4** | **13/28 (46 %)** | **26/64 (41 %)** | **+0.56** | 7.6 | 746 | **+0.06** |
| *20 random brains* | 0.025 | 0/80 | 0/560 | 0/1,280 | +0.00 | 0.9 | | |
| *senses-only controller* | 2.027 | 4/4 | 13/28 (46 %) | 31/64 (48 %) | +0.65 | 3.3 | 392 | −0.02 |

- **The gap closed; the success did not rise.** With 28 full layouts the generalisation gap peaks at +0.17 (generation 100)
  and falls to **+0.06** by generation 200, where it stays: the champion keeps 88 % of its full-training success on unseen
  layouts (the first run: +0.24, 63 %). More layouts stopped the champion learning the maps, as decision 7 expected. But its
  training success fell with it (46 % of 28 full against 67 % of 12), and on unseen layouts it delivers **26 of 64 (41 %)**,
  below half.
- **The baselines.** Random brains deliver nothing. The hand-written senses-only controller delivers 31 of 64 at a median step of
  392; the evolved team delivers 26 at 746: **close to the controller, still half its speed**, as in the first run.
- **Distance decides, more sharply than before**: 6–8 units **20 of 24 (83 %)**, 8–10 units 6 of 22 (27 %), 10–12 units **0 of
  18**. The undelivered ones end 40 % of the way on average. Short pushes are now nearly solved; long ones are not done once.
- **Two held-out layouts barely moved** (`held-0055`, `held-0059`, progress under 5 %): 4 agents touch the box in both, so they
  are reached but not pushed, not unpushable layouts.

**Both overnight champions on one set** (`box-pushing compare runs/full-20261017-201610 runs/full-20261018-232911`: the new
run's held-out set, generator 2, seed 2, the same 64 layouts as the bridge; Metal, 2.6 minutes):

| run | trained on | delivered | progress | `max_pushers` | delivery step (median) | 6–8 units | 8–10 | 10–12 | its own training: full delivered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first overnight (§10.8) | generator 1, 4 + 12 | **30/64 (47 %)** | +0.66 | 6.3 | 690 | 17/24 | 11/22 | 2/18 | 67 % |
| second overnight | generator 2, 4 + 28 | 26/64 (41 %) | +0.56 | 7.6 | 746 | **20/24** | 6/22 | 0/18 | 46 % |
| *senses-only controller* | | 31/64 (48 %) | +0.65 | 3.3 | 392 | 15/24 | 8/22 | 8/18 | 46 % |

- **The new champion does not beat the bridge's 47 %.** On the very layouts it was built for, the old champion (trained on
  fewer, partly unpushable layouts) still delivers four more. The bridge's number reproduces exactly (30 of 64, median step 690
  against 689.5 here), so the comparison is like for like.
- **They differ by distance**: the new champion is better on short pushes (20 against 17) and worse on everything longer (6
  against 11, 0 against 2). It gathers more of the team at the box (`max_pushers` 7.6 against 6.3) and pushes slower (746
  against 690). Better data alone made a team that finds the box and generalises, not one that pushes faster.

**What the clips show** (the eight milestone clips, both montages):

- **No wall- or obstacle-hugging**, at any milestone: the rows of agents along the top wall and the obstacles' edges in the
  first run's clips (§10.8) are gone, and so are the layouts never moved.
- **The team goes to the target, then to the box.** From generation 50 to 200 a large part of the team gathers **at the
  target** and waits there, while a smaller group pushes the box (the clips' `train-0004` and `held-0001` at step 720). By
  generation 400 nearly the whole team is at the box, a few agents still circling the target. The search is solved: every
  layout is reached.
- **The push is still a crowd, and a tighter one.** The final team **rings the box on all sides** (the clips' `held-0000` at
  steps 600 to 900: some 30 agents in a closed ring, 2 or 3 touching at a time, the box creeping), not a line behind it. Agents
  on the leading side push against the travel, which is why the push is slow and the long ones run out of time.
- **No ramming and no traps** (§10.1, §10.2), as before: the box moves only with a crowd at it.
- **Time running out**: the undelivered episodes end with the box partway (40 % on average), none of the long pushes finishing.

**How long it took** (`box-pushing showcase`, on the M4 Pro, Metal for the simulation): the report **137 s** (eight champions
and the baselines on twice as many training scenarios as the first run's), the eight milestone clips **520 s**, the sequence
montage **305 s**, the grid montage **62 s**, the charts under a second: about **17 minutes** in all; `compare` 2.6 minutes.

**The recommendation.** The report's rule (decision 7) says *limited*: held-out success 41 % (below half), gap +0.06 (well within
0.25), so **not more scenarios: what stops the team on full scenarios**. (Its text still names that step "M2b", the name it had
when the report was written at M3; the step is now the next lever.) §9's rule for this run says the same: below half, so **not
M4 yet; the next lever, one at a time**. The clips choose which: the search no longer limits the team (no wall-hugging, every
layout reached), the push does (a ring around the box, half the controller's speed, no long push delivered). So the next lever
is **the time-bonus weight for the push** (`w_t`, §5.1, `[fitness] time`), which pays a team for delivering sooner and so for pushing from behind; the
sensing range for the search is no longer indicated. The plateau from generation 288 says that a longer run of the same
setup would not get there: the change has to be in what is rewarded.


### 10.10 Open questions after R1

- **"Milestone" names two things.** Decision 5 calls the brains a run saves at log-spaced generations *milestone champions*, and decision 12 forbids the word in code (it names this project's milestones). The code calls them **snapshots** (`snapshots.py`, `brains/gen-<k>.safetensors`); this document uses both until a reader is confused, then snapshots only.
- **A run is reproducible bit for bit only on the platform it was trained on.** The layouts use `math.cos` and `math.sin`, whose last bit differs between macOS's libm and glibc, so a run trained on the Mac rebuilds its scenarios with other last bits on Linux: Auxein then refuses to resume it there (the scenario set's fingerprint is in the evaluator's description), rightly. Deploying a brain anywhere is unaffected (it rebuilds nothing of its run). Even on one OS the last bit can differ between machines, since numpy picks its SIMD kernels by CPU (AVX2, AVX-512): the pinned smoke run is exact on macOS arm64 only, to 9 digits everywhere (§6.1). A run is resumed on the machine that trained it.
- **An interrupted run that is never resumed has no `held_out.csv` or charts**, only its `final.safetensors`: the held-out judging takes minutes, too long for Ctrl-C. Resuming it, even with no generation to add, writes them.
- **On a GPU, a deployment's video can differ a little from its `results.csv`**: the video replays each brain on its first few scenarios in a smaller batch than the 64 judged, and float32 on Metal depends slightly on the batch's shape (§10.7). The video's captions are its own episodes' outcomes, so they never contradict what is shown; on numpy they match the CSV exactly.
- **Brain files record the run's directory** (`provenance.run_dir`, an absolute path), useful to find the run again but a detail of the owner's machine in a file meant to be shared; the run's name is there too, so the path could be dropped.
