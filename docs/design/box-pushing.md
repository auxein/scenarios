# Cooperative box-pushing: design

**Status:** Draft for review · **Date:** 2026-10-16 · **Auxein:** `v0.3.2` · **Milestone:** M2a (fitness, the training config and command, a pilot run)

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
- **Validity, with rejection.** A layout is accepted only if the free space is connected and the box can reach the target; otherwise it is redrawn from the same scenario stream. The check rasterises the arena on a grid of 0.25 units, inflates the obstacles and walls by the **box's circumscribed radius** (half its diagonal, so the check holds whatever the box's orientation) and flood-fills from the box's cell to the target's; a second flood fill, with obstacles inflated by an agent's radius, checks that every agent's spawn point is connected to the box (and no agent overlaps an obstacle, the box, a wall or another agent). New obstacles also keep clear of the box's starting footprint and of the target zone. This is conservative (it may reject a layout the box could cross by turning), which is the safe side. Generation happens once per scenario set, on the host, so it is plain numpy and may loop; a scenario records in `attempts` how many draws it took (most take one; dense difficulties take several).

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

`run_batch(agents, scenarios, rng)` returns an `EpisodeBatchResult` of **raw measurements** (§5.2), each an `(n, s)` array on the backend, as Auxein's environments do: the score is the aggregator's (§5.1). For rendering (§7), the same simulation can be run for one genome on a few scenarios with a hook that records the states at every step (positions, headings, the box's pose); one code path serves training and rendering, so the clips show what was scored. Built at M1a: `simulate(..., hook=TrajectoryRecorder(), every=k)` copies the state to host numpy arrays every `k`th step, and the renderer draws the resulting `Trajectory`, for the scripted controllers that read the true state. **Built at M1b** (§2.11): `BoxPushingWorld.run`, the step loop of `run_batch` and `run_episode`, takes a hook that sees each step's state with what the agents sensed in it, and `BoxPushingWorld.watch` records both for clips of brains (`WatchRecorder`), so a clip of a brain is the episode that would be scored.

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

**The observation layout is defined once**, in `sensors.observation_layout(sensing)`: the named parts of the vector, their slices and sizes (derived from `SensingParams`: `rays = 12`, `range = 5`, `ray_range = None` (the same as `range`), `sectors = 8`), and the range of each. The environment, the senses-only controller, the overlay and the smoke test read it from there.

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

**A training run uses 16 scenarios: 4 easy and 12 full** (decided at M2a; in that order, ids `train-0000`...), the training config's default (`[scenarios]`, §6), **fixed for the whole run**: parents and children are judged on the same scenarios, so a better score is a better brain, not easier scenarios (no rotating subsets). Sixteen rather than M1's 32 is what fits a population of 100 with 40 agents in about half a minute per generation on Metal (§10.3). The generator's own default without a mix (`scenarios.training_set`) is still M1's 32 (8 easy, 24 full), which M1's measurements in this document refer to. Easy scenarios give a clueless team something to stumble into (agents start near the box, close enough to see it, §10.1); the full ones keep evolution pointed at the real task.

### 4.3 Held-out sets

**Fresh layouts never seen in training, at full difficulty**, evaluated after a run with `auxein.driver.evaluate_held_out` (or every few generations for the early stop, §6): the evidence for the "deployment to unseen situations" claim. 64 scenarios by default (ids `held-0000`...), generated from a seed and a **stream name** of their own, so that they are disjoint from the training set by construction, even with the same seed; `check_disjoint` verifies that no layout repeats. The strategy never sees them.

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
- **The score is a named, parameterised source**, `EpisodeScore(weights, max_steps)`, whose label lists the weights and `T`: `mean(<episode_score(progress=1.0, delivery=2.0, time=1.0, approach=0.1, progress_floor=-1.0, max_steps=900)>)`. The run's description, which Auxein compares on resume, therefore says which weights scored it, and a resume with a changed weight is refused (§6.1). Auxein `v0.3.2` labelled a callable source by its `__qualname__` alone, hiding the weights, so a changed weight resumed silently: an Auxein bug, fixed on a branch; until it is released a marked workaround gives the label (§10.5).
- **Gathering against pushing** (a test, with scripted teams in one scenario, 900-step scoring): twelve agents that only gather around the box, all touching it, score 0.100 (approach 1). The push team in the same scenario, after 50 steps (the box moved a third of a unit, 4 % of the way), scores 0.111 and wins. After 40 steps (0.11 units, 1.4 %) it scores 0.079 and **loses**: its column's back agents are still far from the box (approach 0.65 against 1). In general a push beats gathering once its progress exceeds `0.1 × (a_gather − a_push)`, at most 10 % of the way. So the approach term cannot dominate a push that gets going, but a team that only nudges the box while half of it hangs back scores less than one that just gathers: a small valley between "gather" and "push" for the first generations. M2b may lower `w_a` if the pilot or the overnight run shows teams settling for gathering (§10.7).

---

## 6. Training

- **Strategy.** **`GeneticAlgorithm`** is the main strategy: a population evolving is the clearest story. **`PycmaStrategy`** (CMA-ES) on the same setup is a comparison run (M4); §10.6 notes its limits at 1,474 dimensions.
- **Driver settings.** Generation delivery, deterministic mode, recording on (`run_dir`), progress printed with `auxein.callbacks.print_progress`. **The run stops on its budget only** (decided at M2a): no held-out evaluation and no early stop during the run (reading the recording of a running run for it is unchecked, §10.5); held-out results come after the run, in M3's report.
- **The evaluator.** `EpisodeEvaluator(policy, world, training_set, aggregator)`, which takes Auxein's batched path because the world has `run_batch`, `MLPPolicy` has `decode_batch` and the GA's batches are arrays.
- **Sizes** (decided at M2a, from the owner's Metal probe, §10.3): **40 agents, a population of 100** (`GeneticAlgorithm(population_size=100, offspring_size=100)`), **16 training scenarios** (4 easy, 12 full, fixed for the run), 900 steps. About 29 s per generation at most on Metal.
- **Backend.** **torch float32 on Metal** (`mps`) by default; torch on the CPU when Metal is not available, with a message saying so (and several times slower). numpy stays the backend of tests.
- **A command-line tool** trains from a TOML config, reproducible from its seed: the config fixes the world, the team, the sensing, the network, the scenario sets, the fitness weights, the strategy, the budget and the backend, and is copied into the run directory.

### 6.1 Training as built (M2a)

**The config** (`config.TrainingConfig`, TOML through `tomllib`) is the run's definition. Sections: `[world.<section>]` (any field of `WorldParams`, e.g. `[world.simulation] max_steps`), `[sensing]`, `[team] agents`, `[brain] hidden, scale` (`MLPPolicy.space(scale)`), `[scenarios] easy, full, seed, held_out, held_out_seed`, `[fitness]` (the weights), `[strategy] population, offspring, crossover_probability, tournament_size, initial_step`, `[budget] generations`, `[run] seed, backend, device, precision, checkpoint_every` (generations), and a top-level `name` (default: the file's name). Every field has this document's default, so a config lists only what it changes; unknown keys and bad values are errors naming the key (`unknown key 'strategy.populaton'`, `team.agents must be an integer, got 'forty'`). The budget in generations is `population + (generations − 1) × offspring` evaluations: the initial population is the first generation.

**Presets** (`configs/`): `smoke.toml` (10 agents, population 10, 4 scenarios, 200 steps, 3 generations, numpy float64: a few seconds; used by the tests), `pilot.toml` (the full sizes, 25 generations: the start of the learning curve) and `full.toml` (the full sizes, 400 generations, a checkpoint every 10: the overnight run, about 3 hours on Metal).

**The run directory** (`runs/<name>-<timestamp>/` by default, ignored by git): `config.toml` (the effective config, every default filled in), `progress.csv`, `clips/` (from `watch`) and **`recording/`, Auxein's run directory** (event log, metadata, checkpoints; `auxein.open_run(".../recording")`). Auxein's recording has a directory of its own because Auxein starts a run only in an empty or new directory, and so that the two never mix.

**Commands:**

- **`box-pushing train CONFIG [--run-dir DIR]`** builds everything from the config (the training set, the world, the brain, the aggregator, `EpisodeEvaluator`, `GeneticAlgorithm`) and runs Auxein with recording on, deterministic mode, generation delivery and a checkpoint every `run.checkpoint_every` generations (and Auxein's every 5 minutes). It prints one line at the start (backend, device, sizes, and the seconds per generation, measured on two steps), a progress line per generation (`print_progress`), and a summary at the end (the best score, its success rate on the training set, where the run is).
- **`progress.csv`** is rewritten after every generation **from the recording**: Auxein's `generation_history()` (generation, evaluations, the best score so far, the generation's score best, mean, median and worst) plus, per generation, the means of the descriptors over its candidates (`success_rate_mean`, `progress_mean`, `approach_mean`, `max_pushers_mean`), the champion's descriptors (`champion_*`) and the active seconds. Decision 6 asked for `ProgressHistory`; it is built from the recording instead, because `ProgressHistory` holds no descriptors and, after a resume, only the generations since the last checkpoint (a resume replays from there), while the recording has the whole run. It can be charted while the run goes on.
- **Ctrl-C stops a run resumably**, after the generation in progress, with a checkpoint (a second Ctrl-C stops at once, losing that generation). With Auxein `v0.3.2` one Ctrl-C did not reach a batched run (an Auxein bug, fixed on a branch, §10.5); `train` handles it itself until the fix is released.
- **`box-pushing train --resume DIR [--generations N] [--config CONFIG]`** continues a run from its recording and its `config.toml`: after Ctrl-C or a crash (Auxein replays from the last checkpoint; deterministic mode, so the result is the run that never stopped), or with another budget (`--generations`: more to extend it, fewer to end it sooner, never fewer than it has). **Only the budget may change**: a `--config` that differs in anything else is refused, listing every difference (`fitness.approach: recorded 0.1, given 0.2`), and so is an edited `config.toml` in the run directory, by Auxein, whose comparison of the evaluator sees the world's parameters and the score's weights. The new budget is written to the run's `config.toml`.
- **`box-pushing watch DIR [--generation K] [--scenarios train:0 held:1 ...] [--overlay]`** renders the best brain up to generation `K` (`RunReader.best_up_to(K)`, the last told generation by default) on showcase scenarios (by default the first easy and the first full training scenario and the first two held-out ones), **with the run's backend, precision and device** read from the recording (§7), through the environment's step loop (M1b's brains rendering), into `DIR/clips/best-gen<K>.mp4`, and prints each scenario's measurements. It reads the recording read-only (SQLite in WAL mode), so **it is safe while the run trains**: a test watches a run that another process is training. On the training scenarios the clip's measurements are the recorded ones (numpy float64, a test). It is a quick look, not M3's milestone clips.
- **The checks it passes** (`tests/box_pushing/train_test.py`, on numpy, CI-sized): `smoke.toml` trains, records, and writes `config.toml` and `progress.csv`; a run interrupted after its second generation (Ctrl-C as a callback would raise it) and resumed has the event log (events, candidates and genomes, lineage, evaluations, episodes) and the best of the run that was never stopped; a run of two generations extended to three has the event log of a three-generation run; a resume with a changed fitness weight or team size is refused, and one with an edited weight in the run's `config.toml` is refused by Auxein; a new run never overwrites one; `watch` renders a smoke run's best with its recorded measurements; and one Ctrl-C stops a run that another process is training, while `watch` reads it, and the run resumes.

---

## 7. Videos and charts

- **Rendered offline from the recording**, never during training (except an optional progress clip, rendered from a callback at a chosen interval, which blocks the run while it renders). Auxein's reader gives the best genome at any generation (`RunReader.best_up_to(k)`, `champions(every=n)`), so any generation can be rendered after the fact.
- **The same backend and precision as the run.** Penalty contacts with many bodies are chaotic, so a float32 champion replayed in float64 (or on another device) may behave visibly differently and even fail to deliver. Clips are rendered with the run's backend, precision and device, read from the recording's metadata, so that they show what was scored.
- **Default milestones:** log-spaced generations (1, 2, 5, 10, 20, 50, 100, 200, 500, …, and the last), **skipping a milestone whose champion is unchanged** from the previous one (the same candidate id): about 8–10 clips for a run of a few hundred generations. Any generation, or every `n`th, on request. The log spacing is computed here, with one `best_up_to(k)` per milestone, since `champions(every=n)` is arithmetic.
- **Showcase scenarios.** Each clip shows one or more fixed showcase scenarios, from the training set and from a held-out set, **the same across milestones** so that they can be compared.
- **A montage:** the milestone clips in sequence or side by side, with the generation number and a small fitness-over-generations curve with a moving marker.
- **Look and feel** (decided at M1a, `common/theme.py`): every video, gallery and chart shares one theme, **monochrome silver on near-black**, with no accent colour.
  - **State is shown with brightness, never with hue.** The box is outlined in steel at rest, filled bright white while moving, silver once delivered, when the target zone fills. Agents touching the box are bright white.
  - **Agents** are silver chevrons along their heading, inside a faint outline of their true disc, and agents close to each other are joined by faint lines (in M1b these can show which teammates each agent senses).
  - **Type:** a small tracked uppercase label, a light headline (Inter Tight Light) and a muted lead (Inter Light) above the panels; small tracked captions in each panel. Inter and Inter Tight are bundled (`common/fonts/`, SIL Open Font License), so videos look the same on every machine, CI included.
  - **Depth:** the floor is lit faintly at its centre and dithered so that the shallow gradient does not band.
  - **No branding** in the videos.
- **The sensing overlay** (M1b, `render.py`), optional per clip, for one or a few chosen agents, recorded by the same step loop that scores an episode (`WatchRecorder`), in the same monochrome theme: the watched agent bright white inside its **sensing circle** (dashed, with ticks at the sector boundaries); its **rays**, faint lines ending in a dot where they hit, **bright white on the box** and steel on a wall or an obstacle (no dot when nothing is in range), rays on the box drawn brighter; the **box cue**, a line to the box's centre while the box is in range; and the **teammates it senses**, joined to it by lines as bright as their weight `1 − d / r`: the faint links of M1a, now meaning "sensed", replacing the proximity links while the overlay is on. The overlay is clipped to the arena.
- **Panels are cells**, a candidate on a scenario: the demos show one candidate on several scenarios, the brains clip several candidates (random brains) on one scenario.
- **Format:** MP4, drawn with matplotlib and encoded by ffmpeg. ffmpeg comes from the `imageio-ffmpeg` wheel, which ships a static binary for macOS and Linux; matplotlib's `FFMpegWriter` is pointed at it, so no system ffmpeg is needed and `uv sync` alone is enough on a Mac and in CI.
- **Charts:** fitness over generations from `RunReader.generation_history()` (best, mean and median per generation), and the success rate and `max_pushers` from the recorded descriptors.

---

## 8. Repository layout

```
auxein_scenarios/
  common/                 # helpers shared by the scenarios
    geometry.py           # 2-D geometry on batched arrays: frames, rectangle contacts, corners, ray casts
    rendering.py          # matplotlib + ffmpeg: figures, frame timing, MP4 and PNG (overlays, montage: M3)
    theme.py              # the look: colours, fonts, agent glyph, floor (§7)
    fonts/                # Inter and Inter Tight, with their licences
    runs.py               # run directories and opening their recordings
    cli.py                # config loading, backend choice, common options
  box_pushing/            # this scenario
    world.py              # the physics core: layout, state, step, runner, trajectories (§2.10)
    contacts.py           # the penalty contact model, every pair (§2.6)
    environment.py        # BoxPushingWorld: run_batch, run_episode, the step loop, measurements, failures (§2.11); the brain
    scripted.py           # scripted controllers (true state) for tests and demos; the senses-only controller (§10.1)
    sensors.py            # the observation layout and batched sensing (§3.8)
    scenarios.py          # the scenario generator, difficulties, training and held-out sets (§4)
    fitness.py            # measurements to score: the episode score and the Aggregator (§5.3)
    config.py             # the world's and the sensing parameters (§2, §3); the TOML training config (§6.1)
    train.py              # building a run from its config; train, resume, watch; progress.csv (§6.1)
    render.py             # clips, galleries and the sensing overlay; milestones, montage and charts from a recording (M3, §7)
    demos.py              # the demo scenes; clips of random brains and of the senses-only controller (M1b)
    cli.py                # box-pushing demo ..., scenarios ..., brains ..., senses ..., train ..., watch ...
configs/                  # training presets: smoke.toml, pilot.toml, full.toml (§6.1)
benchmarks/               # scripts run by hand: physics_timing.py (M1a), probe.py (M1b, §10.3)
tests/                    # smoke_test.py, common/, box_pushing/, benchmarks/
runs/                     # run directories (ignored by git): config.toml, progress.csv, clips/, recording/
docs/design/              # one design document per scenario
docs/auxein-issues.md     # the log of Auxein bugs and requests found here
```

`contacts.py` and `config.py` were added to the module list this design was given (decided, §10.6): the contact model is the largest and riskiest part of the physics, and the parameters are shared by the world, training, rendering and the held-out report. M1a added `scripted.py` (test and demo tools, not part of training) and `demos.py`. **M1b added `environment.py`**: the environment is more than the physics (sensing, the brain's loop, measurements, failures), and keeping it out of `world.py` keeps the physics core free of sensors and networks, as M1a built it.

**Commands added at M1b** (clips to `out/`): `box-pushing brains [--count C] [--seed S] [--difficulty easy|full] [--agents N] [--overlay]` (random brains, one panel each, on one scenario: the clueless start), `box-pushing senses [...]` (the senses-only controller, one scenario per panel) and `box-pushing demo sensing` (one agent turning on the spot beside the box, an obstacle and moving teammates, with its overlay). Each prints every panel's measurements.

**Added at M2a:** `fitness.py` and `train.py` filled in, the training config in `config.py`, the presets in `configs/`, and the commands `box-pushing train CONFIG`, `box-pushing train --resume DIR [--generations N]` and `box-pushing watch DIR` (§6.1). The common helpers listed for `common/runs.py` and `common/cli.py` stay empty: the run directory and the backend choice live in `train.py` until a second scenario needs them.

**Adding a second scenario:** a subpackage `auxein_scenarios/<name>/`, its design document `docs/design/<name>.md`, its tests `tests/<name>/`, and an entry in the README's scenario list. A helper starts in its scenario and moves to `common/` when a second scenario needs it, not before.

---

## 9. Milestones

- **M0** (done): the design document and the repository scaffold.
- **M1** is split in two, each ending with something to run and watch:
  - **M1a** (done): the physics core (bodies, contacts, friction, integration, delivery), the scenario generator, and a viewer: MP4 demo clips driven by scripted controllers that use the true state, and PNG galleries of layouts (`uv run box-pushing demo all`, `uv run box-pushing scenarios`); physics tests (§2.10); a quick timing of the physics (§10.3).
  - **M1b** (done): sensors and the observation layout (§3.8), the Auxein environment `BoxPushingWorld` with both paths, the layout cache, measurements and episode failures (§2.11, §5.2), random `MLPPolicy` brains and a senses-only controller acting on their own senses through the same step loop (`uv run box-pushing brains`, `box-pushing senses`), the sensing overlay (`--overlay`, `box-pushing demo sensing`), determinism end to end, and the full performance probe (§10.3).
- **M2** is split in three:
  - **M2a** (done): the real fitness (§5.3), the TOML config and presets, `box-pushing train` with resume and extension, `box-pushing watch` for a run's current best (§6.1), the probe's fixes and the Metal numbers (§10.3), and a pilot run (§10.7).
  - **The owner's overnight run** on the M4: `configs/full.toml`, 400 generations.
  - **M2b:** tuning from what that run shows: the ramming exploit, oblique pushes, the weights, the step limit (§10.2, §10.7).
- **M3:** videos and the montage from the recording; the held-out report.
- **M4:** the CMA-ES comparison and polished clips for the website.

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

- **`v0.3.1` reported itself as version `0.3.0`** (fixed in `v0.3.2`). The `v0.3.1` tag's `pyproject.toml` was not bumped, so `auxein.__version__` was `"0.3.0"` when installed from it. Logged in `docs/auxein-issues.md`; the scenarios now depend on `v0.3.2`, and the smoke test checks the tag, the commit (PEP 610 metadata) and the version.
- **Auxein's test fixtures are not installed with it** (`tests/support/` is outside the wheel), so the minimal reproductions the protocol asks for are written in a clone of Auxein, not here.
- **M1b drove Auxein's evaluator and driver for real and found no bug.** `EpisodeEvaluator` took the batched path on its own (`MLPPolicy.decode_batch` and `run_batch`) and the per-episode path with a decoder without `decode_batch`; the two agree; a recorded two-generation `auxein.run` with `RandomSearch` completed on numpy float64 and torch float32, with the six measurements per episode in the recording; a NaN action became a failed episode and a failed candidate, as core design §6.6 says.
- **Feature request: a batched environment cannot learn the batch's size or backend from the protocol** (logged in `docs/auxein-issues.md`). `run_batch` must build its state for `n` candidates on the run's device, but `agents` is opaque and `BatchPolicy` has only `act_batch`. The world reads `MLPAgentBatch`'s `weights` and `n` (as Auxein's own team fixture does) or a policy's `backend` and `n`, a convention for the scenarios' own policies. `n` and `backend` on `BatchPolicy`, or a documented guarantee that `run_batch`'s `rng` is on the run's backend, would remove the convention.
- **Held-out evaluation during a run** (the early stop, §6) would call `evaluate_held_out` on a run directory that the driver is still writing. M2a decided not to (the run stops on its budget only), so it stays unchecked; held-out results come after the run (M3).
- **M2a drove the GA, checkpoints and resume for real, and found two bugs** (both logged in `docs/auxein-issues.md`, fixed on branches of Auxein, and worked around here until a tag has them; suggested `v0.3.3`):
  - **A parameterised aggregator source hid its parameters.** A reduction labelled a callable source by its `__qualname__`, so a callable object or a `functools.partial` showed its class's name whatever its parameters (`mean(<EpisodeScore>)`), and resuming a run with changed fitness weights was accepted. Fix: `fix/parameterised-source-labels` (a source is labelled by its `describe()`, its own `__repr__`, or a partial's function and arguments). Workaround: `EpisodeScore` and `EpisodeProgress` carry their description as an instance `__qualname__`, the label the fix gives, so runs recorded now stay resumable after the upgrade; the training command also compares the whole config on resume.
  - **One Ctrl-C did not stop a batched run.** `asyncio.run` turns Ctrl-C into a cancellation of the driver's task, delivered only when the task yields to the event loop, and the live loop yielded only while replaying, so a batched evaluator ran on to the end of the budget and only a second Ctrl-C stopped it, in the middle of a generation. Auxein's interrupt tests raised `KeyboardInterrupt` from code, never a real SIGINT. Fix: `fix/ctrl-c-between-batches` (the driver yields after every tell). Workaround: `train.StopOnCtrlC`, a SIGINT handler of the scenario's own and a callback that raises `KeyboardInterrupt` after the generation's tell.
- **What worked as documented:** the GA in generation delivery with `offspring_size` fixed (so a budget extension is exact); interrupting after a generation and resuming, and extending a finished run, both give the event log of the run that never stopped; resume refuses a changed world or (with the fix or the workaround) changed weights; `RunReader` reads a recording that another process is writing (SQLite in WAL mode), so `watch` and `progress.csv` work during a run; a callback raising `KeyboardInterrupt` after a tell gives an interrupted run with a checkpoint.
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
