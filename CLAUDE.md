# CLAUDE.md

This repository holds **demonstrations built on [Auxein](https://github.com/auxein/auxein)**, a Python framework for evolving
agents that act in environments. Each scenario is a world plus everything needed to evolve agents in it, watch them and show
the results. Package: `auxein_scenarios` (`common/` for shared helpers, one subpackage per scenario).

## Read first

- **The scenario's design document** before changing anything in it: [`docs/design/box-pushing.md`](docs/design/box-pushing.md)
  for `auxein_scenarios/box_pushing/`. Decisions live there; a change of decision updates the document in the same PR.
- **Auxein's design document**, `docs/design/core.md` at the tag this repository depends on (see `pyproject.toml`), for the
  contracts the scenarios rely on: environments and scenarios (§6), backends (§7), randomness (§8), the driver and callbacks
  (§9), recording and the reader (§10).
- **The log of Auxein issues**: [`docs/auxein-issues.md`](docs/auxein-issues.md), including any open `AUXEIN-WORKAROUND`.

## Quality bar (the same as Auxein's)

- `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (strict on `auxein_scenarios`) and `uv run pytest` all
  pass, on the numpy-only installation (`uv sync`) and with torch (`uv sync --extra torch`). CI runs both.
- Docstrings explain *why*, not only what. No `Any` in public signatures unless a comment justifies it.
- Numeric code is written against the Array API namespace of its inputs and follows Auxein's rules (core design §7.2): no in-place
  mutation, no global random state, arrays stay on their device, correct in float32. No Python loops over candidates, scenarios
  or agents in anything that runs per step.
- Small commits, one PR per milestone on its own branch. Never push to `main`, never tag, never publish.

## Working on Auxein from this repository

The scenarios have a second purpose besides the demo: **they are Auxein's first real user, so they are how Auxein's bugs get
found and fixed.** Every milestone follows this protocol.

1. **Confirm it's Auxein.** When something misbehaves and the cause looks like Auxein rather than the scenario, reproduce it
   **with a minimal example that doesn't use the scenario** (a few lines with Auxein's own test fixtures; `tests/support/` is not
   installed with the package, so this is done in a clone of Auxein). If it can't be reproduced without the scenario, it's
   probably a scenario bug: keep going here.
2. **Announce the switch.** Say clearly, in a separate, headed part of the conversation and of the report, **"Working on Auxein,
   not the scenario"**, with the symptom and the minimal reproduction. Never mix Auxein changes into a scenario commit or PR.
3. **Fix it in Auxein, test first.**
   - Work in a separate clone of Auxein (e.g. `../auxein`), on a branch off Auxein's `master` named `fix/<short-description>`.
   - **First write a regression test in Auxein's test suite** that fails, using Auxein's fixtures and **nothing
     scenario-specific**. Then fix it, so the test passes.
   - Follow Auxein's own rules (`docs/design/core.md`, its quality bar, its backend test matrix); update its design document if
     behaviour changes, and its `CHANGELOG.md` `Unreleased` section.
   - **One PR per bug, in the Auxein repository.** Never push to Auxein's `master`, never tag, never change its version.
   - If the fix isn't clear or would change Auxein's design, stop and report instead of fixing.
4. **Keep the scenario on the released tag.** To check the fix against the scenario, point `[tool.uv.sources]` temporarily at the
   local clone (`auxein = { path = "../auxein", editable = true }`), but **never commit that**: committed code always depends on
   a tag. If the scenario must keep moving before the fix is released, a workaround is allowed only if it's small, marked with a
   `# AUXEIN-WORKAROUND: <log entry>` comment, and removed once the fixed tag is in use.
5. **Log it** in [`docs/auxein-issues.md`](docs/auxein-issues.md): date, symptom, minimal reproduction, the Auxein branch or PR,
   status (open / fixed in PR / released in tag), workaround if any.
6. **Tell the owner whether a new tag is needed.** At the end of every session's report, a section **"Auxein: new tag needed?"**
   with either "no" or: which Auxein PRs must be merged, the suggested tag (the next patch version for fixes, a minor version for new
   features), and what the scenario will change once it's tagged (the `tag =` line, removing workarounds). The owner merges and
   tags; Claude Code never does.
7. **Missing features aren't bugs.** If Auxein lacks something generic that the scenario needs, don't implement it in Auxein on
   the side: describe it in the report and the log as a request, and the owner decides.

**Moving to a new Auxein tag:** change `tag =` under `[tool.uv.sources]` in `pyproject.toml`, run
`uv lock --upgrade-package auxein`, update the tag, commit and reported version in `tests/smoke_test.py`, remove the workarounds
the new tag fixes (search for `AUXEIN-WORKAROUND`) and mark their log entries as released.

## End-of-session report

Every session ends with: what was built; anything in the decisions that looks wrong, risky or underspecified (also added to the
design document's open questions); open questions; and **"Auxein: new tag needed?"**.
