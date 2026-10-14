# Cooperative box-pushing: design

**Status:** Draft for review · **Date:** 2026-10-14 · **Auxein:** `v0.3.2` · **Milestone:** M0 (design and scaffold)

---

## 1. Purpose and story

**A team of agents, whose shared brain is a small neural network evolved by Auxein, learns to push a box too heavy for any one of them to a target, in an arena with obstacles.** The scenario exists to show three things, in this order:

1. **A clueless start.** The first generation's teams wander, spin and bump into walls.
2. **Cooperation emerging.** Over generations, teams find the box, gather on one side of it and push it together, because nothing less moves it.
3. **Deployment to unseen situations.** The evolved team, unchanged, delivers the box in layouts it has never seen.

**The outputs are videos and simple charts**, for a website and for talks: clips of the champion at chosen generations, a montage of them, and fitness over generations (§7). **Training runs on a laptop** (Apple Silicon M4) in hours, not days, which bounds every size in this document (§10.3).

**Box-pushing specifics stay in this repository.** Anything generic that turns out to be missing from Auxein is described as a request to Auxein, separately, and never built here on the side (`CLAUDE.md`, "Working on Auxein from this repository").

**Numbers in this document are starting defaults**, all configurable and expected to be tuned once the world runs (M1) and the first runs are in (M2). A default marked *(proposed)* is not one of the decisions this design was given: it was needed to make the document coherent and is listed in §10.6 for review.

---

## 2. The world

### 2.1 Why its own physics

**The world is its own vectorised 2-D physics, implemented as an Auxein `BatchedEnvironment`.** One `run_batch` call simulates every candidate of a batch on every scenario of the set, and every agent of every team, at once: arrays shaped `(n, s, ...)` (candidates, scenarios) and `(n, s, N, ...)` (agents), with **no Python loops over candidates, scenarios or agents**; the only Python loop is over time steps.

- **Why not an existing engine.** A general physics engine (Box2D, pymunk, MuJoCo) steps one world at a time from Python, so a generation of 100 candidates × 32 scenarios would be 3,200 worlds stepped in a loop. Batched GPU engines exist but bring their own array library, while Auxein's numeric layer is the Array API standard on numpy and torch (core design §7). A small, purpose-built model written against the same namespace runs where the genomes already are, with no host round trip (the pattern of Auxein's team fixture, `tests/support/team.py`).
- **Array API, as Auxein.** The code uses the namespace of its inputs (`array-api-compat`): numpy or torch, float64 or float32, CPU, CUDA or Metal. It follows Auxein's rules for numeric code (core design §7.2): no in-place mutation (so updates are `xp.where`, never masked assignment), no global random state, arrays stay on their device, and everything works in float32, the only precision Metal has.
- **Deterministic.** For a given scenario seed, backend, precision and device, an episode gives the same numbers every time. Contacts and sensing are written as **dense pairwise arrays reduced with sums**, never as scatters (`index_add`), whose order on a GPU is not fixed. The scope is Auxein's (core design §8): float32 results are not portable between CPU architectures or between a CPU and a GPU.
- **Integration.** A fixed time step `dt` and semi-implicit (symplectic) Euler: velocities are updated from forces first, then positions from the new velocities. It is the cheapest integrator that is stable for the stiff springs of penalty contacts (§2.6). *(proposed)* The policy acts once per episode step, `dt = 0.05` s, and the physics runs `substeps = 2` steps of `dt / substeps` per episode step, so that contact stiffness and the control rate can be tuned apart.

### 2.2 Arena and obstacles

- **The arena** is a walled square, 20 × 20 units by default, with the origin at its centre.
- **Obstacles are random per scenario**: a few circles and axis-aligned rectangles (by default 2 to 6 in all at full difficulty, §4.2). *(proposed)* Circle radii 0.5 to 1.5 units; rectangle sides 0.5 to 4 units, so no obstacle is thinner than an agent's diameter.
- **Validity, with rejection.** A layout is accepted only if the free space is connected and the box can reach the target; otherwise it is redrawn from the same scenario stream. *(proposed)* The check rasterises the arena on a grid of 0.25 units, inflates the obstacles and walls by the **box's circumscribed radius** (half its diagonal, so the check holds whatever the box's orientation) and flood-fills from the box's cell to the target's; a second flood fill, with obstacles inflated by an agent's radius, checks that every agent's spawn point is connected to the box. This is conservative (it may reject a layout the box could cross by turning), which is the safe side. Generation happens once per scenario set, on the host, so it is plain numpy and may loop.

### 2.3 The box

- **A rectangle**, 2 × 1.2 units by default, with mass `M` and moment of inertia `I = M (w² + h²) / 12` (a uniform rectangle).
- **It translates and rotates.** Contact forces applied off its centre create torque, so pushing at a corner turns the box, and a team must push near the middle (or balance its pushes) to move it straight.
- **It moves only on the floor's terms** (§2.4); it has no actuator of its own.

### 2.4 Cooperation is necessary: friction

The box rests on the floor with **static friction**, so it moves only when the combined push exceeds a threshold, and the threshold is set so that it **needs at least `k` agents pushing together** (`k = 4` by default). One agent alone cannot move it; neither can `k − 1`.

- **The threshold from `k`.** With `F_a` an agent's maximum thrust, the static threshold is set between what `k − 1` and `k` agents can push: *(proposed)* `F_s = (k − 0.5) · F_a`. Four agents pushing exactly in line produce `4 F_a`, enough with a margin of `0.5 F_a`; they move the box as long as each push is within about 29° of the direction they move it in (`4 cos 29° ≈ 3.5`). This margin is a lever: smaller makes cooperation stricter and the first generations' search harder.
- **At rest.** While the box is at rest, it stays at rest as long as the net horizontal force on it is below `F_s` **and** the net torque below `τ_s`; *(proposed)* `τ_s = F_s · ρ`, with `ρ` a friction radius of the order of the box's size (`ρ = (w + h) / 6` by default, about the mean distance of the box's footprint from its centre). Treating force and torque with two separate thresholds is a simplification of a real friction limit surface, chosen because it is two comparisons on arrays.
- **Moving.** Once moving, **kinetic friction** opposes the velocity with a constant magnitude `F_k = μ · F_s` (`μ = 0.8` by default *(proposed)*, so that a moving box is a little easier to keep moving, as with real friction), and **angular damping** opposes rotation. The box comes to rest again when its speed and angular speed both fall below small thresholds.
- **Agents pushing agents.** Contact forces pass through agents, so a line of agents pushing each other into the box adds up like agents pushing it side by side. This is allowed: it is cooperation too.

### 2.5 Agents

- **Discs** of radius 0.25 by default, mass `m` (1 by default).
- **Differential drive.** Each agent has a heading; its actions are a **forward thrust** (a force along the heading, negative for reverse, *(proposed)* reverse limited to half the forward thrust) and a **turning rate**. Linear and angular drag, and maximum speeds. *(proposed)* The commanded turning rate is reached through a first-order lag (angular drag), and the velocity across the heading is damped strongly, as wheels resist sliding sideways; contacts can still push an agent sideways.
- **Scaling.** Actions arrive in `[-1, 1]` (§3.6) and are scaled to the agent's limits: thrust to `[-F_a / 2, F_a]` and turning to `[-ω_max, ω_max]`.

### 2.6 Collisions: penalty contacts

Agents collide with each other, with walls, with obstacles and with the box; the box collides with walls and obstacles. **Contacts are penalty-based**: an overlap `δ > 0` between two bodies produces a spring–damper force `f = k_c δ + c_c δ̇` along the contact normal (only pushing, never pulling), applied at the contact point, so it creates torque on the box. Penalty contacts are chosen **because they vectorise cleanly**: every pair is an independent formula evaluated on arrays, with no constraint solver and no data-dependent iteration. *(proposed)* Contacts are frictionless (agents slide along walls and along the box): friction between agents and the box is a possible later refinement.

The contact pairs:

| Pair | Bodies | Note |
|---|---|---|
| disc–disc | agent–agent, agent–circular obstacle | the `N²` pair (§2.7) |
| disc–axis-aligned rectangle | agent–rectangular obstacle | closest point by clamping |
| disc–oriented rectangle | agent–box | the same, in the box's frame |
| oriented box–circle | box–circular obstacle | the circle against the box, in the box's frame |
| oriented box–axis-aligned rectangle | box–rectangular obstacle | **the hardest one** |
| everything–wall | agents and the box against the four walls | half-planes; the box by its four corners |

- **The rotating box against rectangles** is the hardest pair: two rectangles at an angle meet corner-to-edge or edge-to-edge. *(proposed)* It is computed as corner penetration both ways: the box's four corners against the obstacle (in the obstacle's frame) and the obstacle's four corners against the box (in the box's frame), each penetrating corner contributing a spring force along the face it is closest to. It is a separating-axis-like approximation, exact for the corner–edge contacts that dominate and good enough for the rare edge-to-edge ones (two corners penetrate). §10.2 lists its risks.
- **Stiffness against the time step.** A penalty contact is a spring: stiffer means less overlap but needs a smaller step to stay stable (for semi-implicit Euler, `ω dt < 2` with `ω = √(k_c / m)`, and comfortably below for accuracy). The stiffness is chosen so that the overlap at full push stays small against the agent's radius (a few hundredths of a unit), and the substep from it.
- **No tunnelling.** At maximum speed an agent moves much less than its radius in one substep (for example 1.5 units/s × 0.025 s ≈ 0.04 units against 0.25), and walls are half-planes, so nothing passes through a wall; a test checks it at maximum speed (M1).

### 2.7 Cost: agent pairs grow as agents²

Agent–agent contacts and teammate sensing (§3.5) are computed for all pairs, `(n, s, N, N)` arrays. This is fine at 20–40 agents; at 100 agents the pairs are 10,000 per world, and **the torch path on Metal is expected**. A spatial grid (agents binned into cells, pairs only between neighbouring cells) could replace the dense pairs later if needed; it is not in v1, because sorting and binning are where the Array API is weakest and determinism on a GPU is hardest. §10.3 estimates the cost.

### 2.8 Episodes in the batch

Episodes end early on delivery (§4.4), but a batched world cannot drop a scenario from its arrays. **Finished scenarios keep their array shape but stop changing and scoring**: a `done` mask of shape `(n, s)` freezes their state through `xp.where`, and their measurements are kept from the step they finished. The batch stops early only when every scenario of every candidate is done. Frozen scenarios still cost compute, which is the price of no Python loops.

### 2.9 What `run_batch` returns, and rendering

`run_batch(agents, scenarios, rng)` returns an `EpisodeBatchResult` of **raw measurements** (§5.2), each an `(n, s)` array on the backend, as Auxein's environments do: the score is the aggregator's (§5.1). For rendering (§7), the same simulation can be run for one genome on a few scenarios with a hook that records the states at every step (positions, headings, the box's pose); one code path serves training and rendering, so the clips show what was scored.

---

## 3. Agents: sensing and acting

**All sensing is local and in the agent's own frame** (bearings relative to its heading), **except the target**, which every agent knows. *(proposed)* Every input is normalised to roughly `[-1, 1]` or `[0, 1]`, since the network's initial weights assume inputs of unit scale (`MLPPolicy.space`, core design §6.2).

### 3.1 Target (3 values)

Distance to the target (normalised by the arena's diagonal) and bearing to it as `(sin, cos)`. Always available.

### 3.2 Range rays (12 × 2 = 24 values)

12 rays evenly spaced around the agent (by default), each cast to a maximum range (*(proposed)* the sensing range `r`, §3.3) and returning:

- the **normalised distance** to the nearest wall, obstacle or box along the ray (1 when nothing is within range);
- **what it hit**: *(proposed)* 1 for the box, 0 for a wall or an obstacle (and for nothing).

*(proposed)* Rays do not see teammates; teammates are sensed through sector densities (§3.4), so that a crowd around an agent does not blind it to the box.

### 3.3 Box cue in range (6 values)

Within the sensing range `r` (5 units by default): a **flag** that the box is in range (1), the **distance** to the box's centre (normalised by `r`), the **bearing** to it `(sin, cos)`, and the **box's heading relative to the agent's** `(sin, cos)`. Beyond `r`, all six are 0 and the flag is off. The box is also visible to rays beyond `r` only if the ray range is longer; with both at `r`, an agent farther than `r` from the box knows nothing of it (§10.1).

### 3.4 Teammates: sector densities (8 values)

The space around the agent is divided into 8 sectors (by default); each value is the **closeness-weighted count** of teammates in that sector within the sensing range: *(proposed)* `Σ (1 − d / r)` over the teammates at distance `d < r`. The size is **fixed whatever the team size**, so the same brain works for 20 or 100 agents. §10.6 notes that the values themselves grow with the team's density.

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
| box cue: flag, distance, bearing (sin, cos), relative heading (sin, cos) | 6 |
| teammates: 8 sector densities | 8 |
| proprioception: forward speed, turning rate | 2 |
| **observation size** | **43** |

With one hidden layer of 32 units and 2 outputs, the genome is `43 × 32 + 32` (hidden layer) `+ 32 × 2 + 2` (output layer) = **1,474 weights** (`MLPPolicy(43, (32,), 2, output="tanh").dim`; checked by the scaffold's smoke test). It is laid out as Auxein's `MLPPolicy` lays it out (core design §6.2), and searched in `policy.space(scale)`, whose fan-in–scaled bounds make the GA's uniform first generation a Glorot-like initialisation.

---

## 4. Scenarios, episodes and the training set

### 4.1 A scenario

A scenario is drawn from a seed: the **obstacle layout**, the **box's position and orientation**, the **target position** (at least a minimum distance from the box), and the **agents' starting positions and headings, scattered across the arena**, free of obstacles (and of each other). It is an Auxein `Scenario` in a `ScenarioSet`, so **common random numbers** apply: every candidate faces the same scenarios, and differences in fitness are differences between brains.

- *(proposed)* **The generated layout is stored in the scenario's `params`** (obstacles, box pose, target, agent poses, the difficulty), not regenerated from the seed in the world. The world then has no randomness of its own, the scenario set's fingerprint covers the actual layouts (so a change to the generator cannot silently change a recorded run's scenarios: resuming is refused, core design §6.4), and the set can be saved and inspected as JSON. A layout of 100 agents is a few kilobytes.
- **The world has no noise in v1** (no sensor or actuator noise): the episode's randomness is entirely the scenario's. Noise is a possible lever for robustness on unseen layouts (§10.6).

### 4.2 Difficulty and the training set

**Difficulty is a parameter of the generator**: the number of obstacles, the box–target distance, and the spawn spread (how far from the box the agents start). **The training set mixes difficulties**, so the first generations get a gradient:

| Parameter | easy | full |
|---|---|---|
| obstacles | 0 to 2 | 2 to 6 |
| box–target distance | 3 to 5 units | 6 to 12 units *(proposed)* |
| spawn | within 3 units of the box | scattered across the arena |

*(proposed)* The default set is **32 scenarios: 8 easy and 24 full**, with the mix configurable (fractions per difficulty, or a list of difficulty levels). Easy scenarios give a clueless team something to stumble into (agents start near the box, close enough to see it, §10.1); the full ones keep evolution pointed at the real task.

### 4.3 Held-out sets

**Fresh layouts never seen in training, at full difficulty**, evaluated after a run with `auxein.driver.evaluate_held_out` (or every few generations for the early stop, §6): the evidence for the "deployment to unseen situations" claim. *(proposed)* 64 scenarios by default, generated from a seed of their own, so that they are disjoint from the training set by construction; the generator checks that no layout repeats. The strategy never sees them.

### 4.4 Episodes and delivery

- **A fixed limit**, 600 steps by default (30 s of simulated time at `dt = 0.05`), **ending early on delivery** (§2.8).
- **Delivery:** the **box's centre enters the target zone**, a circle (*(proposed)* of radius 1.5 units); the box's orientation does not matter.
- **Team size** is a parameter: 20–40 typical, up to 100. One run uses one team size; the brain does not depend on it (§3.4), so a brain evolved with 20 agents can be shown with 40.

---

## 5. Fitness

### 5.1 One shaped score

**A single score per episode, maximised** (`Objective("score", "maximise")`), aggregated over scenarios with the mean by default (the aggregator is configurable: a lower quantile or `cvar_lower` would reward teams that rarely fail). Its terms:

- **Progress** of the box toward the target, the main term: `(d₀ − d_end) / d₀`, the initial minus the final distance of the box's centre to the target's, normalised by the initial one (1 when delivered at the centre, 0 when untouched, negative when pushed away; *(proposed)* clipped at −1).
- **A delivery bonus**, large, and **a time bonus** for delivering early: `delivered · (w_d + w_t · (1 − t / T))`, with `t` the delivery step and `T` the step limit.
- **An approach term**, small: how close the agents get to the box, *(proposed)* the mean over agents of `1 − min(1, d_min / r)`, with `d_min` the agent's closest distance to the box's surface during the episode. It helps the first generations get started, when no team moves the box, and **must stay small so it cannot dominate**: a team that only gathers around the box must score less than one that moves it.

**Default weights**, to be tuned after the first runs:

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
| `max_pushers` *(proposed)* | the largest number of agents touching the box at once: a direct measure of cooperation, for the charts |

The aggregator's **descriptors** carry the per-run summaries the charts and the early stop need: the success rate (mean of `delivered`), the mean progress, the mean approach and the mean of `max_pushers`.

---

## 6. Training

- **Strategy.** **`GeneticAlgorithm`** is the main strategy: a population evolving is the clearest story. **`PycmaStrategy`** (CMA-ES) on the same setup is a comparison run (M4); §10.6 notes its limits at 1,474 dimensions.
- **Driver settings.** Generation delivery, deterministic mode, recording on (`run_dir`), progress printed with `auxein.callbacks.print_progress`, and an optional **early stop when the held-out success rate passes a threshold**: a callback that, every few generations, evaluates the current best on the held-out set and returns `StopRun` (§10.6 notes what has to be checked first).
- **The evaluator.** `EpisodeEvaluator(policy, world, training_set, aggregator)`, which takes Auxein's batched path because the world has `run_batch`, `MLPPolicy` has `decode_batch` and the GA's batches are arrays.
- **Backend.** numpy float32 or torch float32 on Metal; the choice is made at M1 from the measured performance (§9).
- **A command-line tool** trains from a TOML config, reproducible from its seed: the config fixes the world, the team, the sensing, the network, the scenario sets, the fitness weights, the strategy, the budget and the backend, and is copied into the run directory.

---

## 7. Videos and charts

- **Rendered offline from the recording**, never during training (except an optional progress clip, rendered from a callback at a chosen interval, which blocks the run while it renders). Auxein's reader gives the best genome at any generation (`RunReader.best_up_to(k)`, `champions(every=n)`), so any generation can be rendered after the fact.
- **The same backend and precision as the run.** Penalty contacts with many bodies are chaotic, so a float32 champion replayed in float64 (or on another device) may behave visibly differently and even fail to deliver. Clips are rendered with the run's backend, precision and device, read from the recording's metadata, so that they show what was scored.
- **Default milestones:** log-spaced generations (1, 2, 5, 10, 20, 50, 100, 200, 500, …, and the last), **skipping a milestone whose champion is unchanged** from the previous one (the same candidate id): about 8–10 clips for a run of a few hundred generations. Any generation, or every `n`th, on request. The log spacing is computed here, with one `best_up_to(k)` per milestone, since `champions(every=n)` is arithmetic.
- **Showcase scenarios.** Each clip shows one or more fixed showcase scenarios, from the training set and from a held-out set, **the same across milestones** so that they can be compared.
- **A montage:** the milestone clips in sequence or side by side, with the generation number and a small fitness-over-generations curve with a moving marker.
- **Format:** MP4, drawn with matplotlib and encoded by ffmpeg. ffmpeg comes from the `imageio-ffmpeg` wheel, which ships a static binary for macOS and Linux; matplotlib's `FFMpegWriter` is pointed at it, so no system ffmpeg is needed and `uv sync` alone is enough on a Mac and in CI.
- **Charts:** fitness over generations from `RunReader.generation_history()` (best, mean and median per generation), and the success rate and `max_pushers` from the recorded descriptors.

---

## 8. Repository layout

```
auxein_scenarios/
  common/                 # helpers shared by the scenarios
    geometry.py           # 2-D geometry on batched arrays: frames, distances, closest points, ray casts
    rendering.py          # matplotlib + ffmpeg: writer, overlays, montage
    runs.py               # run directories and opening their recordings
    cli.py                # config loading, backend choice, common options
  box_pushing/            # this scenario
    world.py              # the BatchedEnvironment: state, integration, episodes, measurements (§2)
    contacts.py           # the penalty contact model, every pair (§2.6)
    sensors.py            # the observation vector (§3)
    scenarios.py          # the scenario generator, difficulties, training and held-out sets (§4)
    fitness.py            # measurements to score: the Aggregator (§5)
    config.py             # the TOML config (§6)
    train.py              # building the run and running Auxein (§6)
    render.py             # clips, montage and charts from a recording (§7)
    cli.py                # the command-line tool
tests/                    # smoke_test.py now; tests/box_pushing/ from M1
docs/design/              # one design document per scenario
docs/auxein-issues.md     # the log of Auxein bugs and requests found here
```

`contacts.py` and `config.py` are additions to the module list this design was given (§10.6): the contact model is the largest and riskiest part of the physics, and the config is shared by training, rendering and the held-out report.

**Adding a second scenario:** a subpackage `auxein_scenarios/<name>/`, its design document `docs/design/<name>.md`, its tests `tests/<name>/`, and an entry in the README's scenario list. A helper starts in its scenario and moves to `common/` when a second scenario needs it, not before.

---

## 9. Milestones

- **M0** (this): the design document and the repository scaffold.
- **M1:** the world, sensors and scenario generator, with an MP4 viewer driven by scripted and random policies; physics tests (conservation where it applies, the friction threshold with `k − 1` and `k` agents, rotation from off-centre pushes, no tunnelling through walls at maximum speed, determinism); a performance probe (steps per second for 20/40/100 agents × candidates × scenarios, on numpy and torch CPU, and Metal on the owner's Mac), from which the backend and the default sizes are chosen.
- **M2:** fitness, the training CLI and config, recording, first trained runs.
- **M3:** videos and the montage from the recording; the held-out report.
- **M4:** the CMA-ES comparison and polished clips for the website.

---

## 10. Risks and open questions

### 10.1 Scattered starts with a memoryless brain

At full difficulty agents start scattered across a 20 × 20 arena and sense the box only within `r = 5` units: many start out of range, **knowing where the target is but not the box**, and a feed-forward brain cannot remember where it has been. It must find a search behaviour that is a pure function of its senses (follow walls, spread out, drift toward teammates who are near something). The levers, in order: the **mixed-difficulty training set** (easy scenarios start agents in sight of the box, §4.2), the **sensing range** (larger `r`, or rays that see the box farther than the cue), the teammate densities (a crowd is a cue), and as a fallback **recurrent brains** (§3.6). The first runs (M2) will tell whether full-difficulty scenarios are ever solved.

### 10.2 Physics

- **The rotating box against rectangles** (§2.6): corner penetration can give a wrong normal near a corner–corner contact, and a box wedged between two obstacles can jitter or be pushed through a thin one. Mitigations: obstacles no thinner than 0.5 units, tests of the box pushed into a corner and along an edge (M1).
- **The friction threshold** (§2.4): with penalty contacts the push on the box is the contact spring force, which oscillates around the agents' thrust, so near the threshold the box may stick and slip. The test with `k − 1` and `k` agents (M1) pins the behaviour; a small hysteresis may be needed. Treating force and torque with separate thresholds can let a team spin the box in place more easily than it could slide it.
- **Stiffness against the time step:** penalty contacts need a small enough step; too soft and agents sink into the box, too stiff and the integration blows up (a NaN measurement fails the candidate, which Auxein records as `FAILED`: a canary, but one that would punish brains for the physics' fault). Substeps (§2.1) are the lever.

### 10.3 Cost

**Back-of-envelope** (to be replaced by M1's probe): agent pairs are `n · s · N²` per substep. With 100 candidates × 32 scenarios × 40 agents that is about 5 million pairs, each a few tens of operations on `(n, s, N, N)` arrays of 5 million elements (20 MB each in float32): with numpy's temporaries, of the order of 0.1 s per substep, so about 2 minutes for the 1,200 substeps of one generation, plus sensing (rays: `n · s · N · 12` casts against about 10 shapes). **A generation should take about a minute for hours-long runs** (300 generations in 5 hours), so the numpy path at these sizes is borderline, and **100 agents (6× the pairs of 40) needs Metal or smaller populations and sets**. The probe decides the defaults: population, scenarios per generation, team size and backend.

### 10.4 Recording size

Auxein records every evaluated genome and every episode's measurements. A genome of 1,474 values is 5.9 KB in float32 (11.8 KB in float64), above Auxein's genome-store threshold (4 KB), so each distinct genome is a blob in `events.sqlite`; the GA's children are all distinct, so the store does not shrink them. A run of 300 generations of 100 children is 30,000 genomes, about **180 MB** in float32, plus 30,000 × 32 = 960,000 episode rows of six measurements (about 200 bytes each with keys and index), about **200 MB**: **roughly 0.4 GB per run** (twice that in float64). It matters for disk on a laptop but not for speed (Auxein's reader indexes a 1 GB recording in 0.5 s, core design §10.2), and run directories are never committed (`.gitignore`). Fewer measurements or scenarios are the levers if it does.

### 10.5 What reading Auxein turned up

- **`v0.3.1` reported itself as version `0.3.0`** (fixed in `v0.3.2`). The `v0.3.1` tag's `pyproject.toml` was not bumped, so `auxein.__version__` was `"0.3.0"` when installed from it. Logged in `docs/auxein-issues.md`; the scenarios now depend on `v0.3.2`, and the smoke test checks the tag, the commit (PEP 610 metadata) and the version.
- **Auxein's test fixtures are not installed with it** (`tests/support/` is outside the wheel), so the minimal reproductions the protocol asks for are written in a clone of Auxein, not here.
- **Held-out evaluation during a run** (the early stop, §6) calls `evaluate_held_out` on a run directory that the driver is still writing. Whether that is supported (it reads the recording while the writer holds `writer.lock`; `write=False` avoids `held_out.json`) is to be checked at M2; if not, it is a request to Auxein, and the fallback is to stop on the **training** success rate with `stop_when` (cheap, reproducible, but not the unseen-scenarios claim).

### 10.6 Proposed details to review

Details this design needed but was not given, each marked *(proposed)* above:

1. **Time step and substeps** (§2.1): `dt = 0.05` s per episode step, 2 physics substeps.
2. **Obstacle sizes and the validity check** (§2.2): the grid flood fill with obstacles inflated by the box's circumscribed radius (conservative).
3. **The friction threshold** `F_s = (k − 0.5) F_a`, the torque threshold `τ_s = F_s (w + h) / 6`, kinetic `F_k = 0.8 F_s` (§2.4).
4. **Agent dynamics** (§2.5): reverse at half thrust, a first-order lag on turning, strong lateral damping.
5. **Frictionless contacts** between bodies (§2.6), and the corner-penetration model for box–rectangle.
6. **Input normalisation** (§3), **rays do not see teammates** and report 1 for the box, 0 otherwise (§3.2), **the ray range equals the sensing range** (§3.3).
7. **The box's relative heading** (§3.3): the box is symmetric under a half turn, so `(sin θ, cos θ)` gives two different inputs for the same situation, and the network must learn that they mean the same. `(sin 2θ, cos 2θ)` would remove the ambiguity at no cost in size. Recommended, but it changes what the decision says.
8. **Teammate density** `Σ (1 − d / r)` (§3.4): it is fixed in size whatever the team, but its values grow with the team's density, so a brain evolved with 20 agents sees larger inputs with 100. Normalising by the team size, or squashing (`1 − e^(−x)`), would make "the same brain for any team size" more than a shape guarantee.
9. **Scenario params hold the whole layout** (§4.1), and **no noise** in the world in v1 (sensor or actuator noise may help generalisation to unseen layouts).
10. **Full-difficulty box–target distance** 6–12 units, **the training mix** 8 easy + 24 full, **64 held-out scenarios**, **target zone radius** 1.5 (§4).
11. **The progress clip at −1** and **the approach term's definition** (§5.1); whether the approach term should fade out over generations (Auxein's aggregator is fixed for a run, so that would be a staged run or a request).
12. **`max_pushers`** as a recorded measurement (§5.2), for the "cooperation emerging" chart.
13. **CMA-ES at 1,474 dimensions** (§6): full-covariance CMA-ES learns its covariance at a rate of about `1/d²`, so in a few hundred generations it barely adapts it, and its `O(d²)` state and eigendecompositions run on the host. The separable variant (pycma's `CMA_diagonal` option, through `PycmaStrategy(options=...)`) is the fairer comparison. `PycmaStrategy` writes no checkpoints, so a resumed comparison run replays from its start.
14. **`contacts.py` and `config.py`** as modules (§8).
