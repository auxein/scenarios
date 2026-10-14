# Auxein issues found from the scenarios

The scenarios are Auxein's first real user. Every Auxein bug they turn up, and every generic feature they turn out to need, is
logged here, newest first, following the protocol in [`CLAUDE.md`](../CLAUDE.md) ("Working on Auxein from this repository").

Each entry has this format:

```markdown
## YYYY-MM-DD: <short title>

- **Kind:** bug | feature request
- **Symptom:** what went wrong, as seen from the scenario.
- **Minimal reproduction:** a few lines that use Auxein (and its own test fixtures) only, nothing from this repository.
- **Auxein branch / PR:** `fix/<short-description>`, and the PR link once opened (none for a feature request: the owner decides).
- **Status:** open | fixed in PR <link> | released in tag `vX.Y.Z`
- **Workaround:** none, or where the `# AUXEIN-WORKAROUND: <this entry's title>` comments are, and what removing them takes.
```

---

## 2026-10-14: v0.3.1 reports version 0.3.0

- **Kind:** bug (release packaging)
- **Symptom:** installed from the tag `v0.3.1`, `auxein.__version__` is `"0.3.0"`, so the scaffold's smoke test cannot check
  for `0.3.1`. The tag (a lightweight tag on `5636796`, the head of `master`) carries `version = "0.3.0"` in `pyproject.toml`, and
  its `CHANGELOG.md` still lists the release's changes under "Unreleased". A run's `metadata.json` records the versions it ran
  with, so recordings made with `v0.3.1` will say `0.3.0` too.
- **Minimal reproduction:**

  ```
  uv venv /tmp/v && uv pip install --python /tmp/v "auxein @ git+https://github.com/auxein/auxein@v0.3.1"
  /tmp/v/bin/python -c "import auxein; print(auxein.__version__)"    # prints 0.3.0
  ```

- **Auxein branch / PR:** none. The fix is a version bump (and the changelog heading), which only the owner makes; a new tag is
  needed for it to reach this repository. A CI check in Auxein that a tag matches `pyproject.toml` would prevent a repeat (a
  feature request for the owner to decide).
- **Status:** open
- **Workaround:** `tests/smoke_test.py` (`AUXEIN_REPORTED_VERSION = "0.3.0"`); the same test pins the tag and the commit through
  the installed package's PEP 610 metadata, which is the real check. Remove it once a tag with a correct version is in use.
