# Held-out report: tiny

Run `/private/tmp/claude-501/-Users-dpalmisano-Documents-repositories-scenarios/ecb415e2-8af4-413a-b20d-fd3aeae39738/scratchpad/pre-m2b/smoke`. 10 agents, 2 easy + 2 full training scenarios, 200 steps, 2 generations. Held-out set: 4 full-difficulty scenarios from seed 2, never seen in training. Evaluated with numpy cpu float64.

Each champion is the run's best up to its generation (log-spaced milestones, unchanged champions skipped). Training numbers are the recording's; held-out ones are Auxein's `evaluate_held_out`. The gap is full-training success minus held-out success.

| | training score | easy delivered | full delivered | held-out delivered | held-out progress | held-out `max_pushers` | held-out delivery step (median) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generation 1 (candidate 1) | 0.048 | 0/2 | 0/2 (0%) | 0/4 (0%) | +0.00 | 0.2 |  | +0.00 |
| generation 2 (candidate 5) | 0.048 | 0/2 | 0/2 (0%) | 0/4 (0%) | +0.00 | 0.2 |  | +0.00 |
| *20 random brains* | 0.046 | 0/40 | 0/40 (0%) | 0/80 (0%) | +0.00 | 0.2 |  | +0.00 |
| *senses-only controller* | 0.185 | 0/2 | 0/2 (0%) | 0/4 (0%) | +0.01 | 2.0 |  | +0.00 |

**Unseen layouts.** The final champion (generation 2, candidate 5) delivers **0 of 4 held-out scenarios (0%)**, moving the box 0% of the way on average, against 0 of 2 full training scenarios (0%). The first generation's champion delivers 0.

**The generalisation gap** (full-training success minus held-out success, both full difficulty), by generation: 1: +0.00, 2: +0.00. At the end it is +0.00: the champion delivers no full training scenario, so there is nothing to carry over yet.

**Baselines** on the same held-out scenarios: the 20 random brains deliver 0 of 80 (0%), progress +0.00; the senses-only controller delivers 0 of 4 (0%), progress +0.01. From random brains that deliver nothing, evolution reached a team that matches the hand-written controller (0 each).

**Distance.** The final champion's held-out deliveries by the box's starting distance to the target: 6–8 units 0/1 (0%), 8–10 units 0/2 (0%), 10+ units 0/1 (0%). The undelivered ones end 0% of the way on average. Success that falls with distance, with the rest partway, points at the 200-step limit as well as the skill.

**Never moved** (progress under 5 %): held-0000, held-0001, held-0002, held-0003. Watch them to see why (`--showcase worst-held`).

**Recommendation: M2b: what stops it on full scenarios (the step limit, the search) before more scenarios.** Held-out success 0%, full-training success 0%, gap +0.00: below 50% on unseen layouts, but within 0.25 of training (case: limited).
