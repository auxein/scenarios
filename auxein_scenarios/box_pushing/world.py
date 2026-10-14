"""The world (design doc §2): a walled arena with obstacles, a box and a team of differential-drive agents (`BatchedEnvironment`).

Its own vectorised 2-D physics, written against the Array API namespace that Auxein gives (numpy or torch, float64 or float32,
CPU, CUDA or Metal): all candidates × scenarios × agents step at once, with no Python loops over them. A fixed time step,
semi-implicit Euler, and deterministic for a given scenario seed and backend. `run_batch` runs whole episodes: the agents act
through `act_batch` on observations shaped `(n, scenarios, agents, observation size)`, finished scenarios keep their shape but
stop changing (§4), and the raw measurements are returned as `(n, scenarios)` arrays for the aggregator (§5).
"""
