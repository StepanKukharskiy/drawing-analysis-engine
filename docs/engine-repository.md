# Engine repository boundary

`deploy/source/engine-manifest.json` is the explicit engine development selection.
Use `stage_source.py --profile engine --destination /absolute/new/directory` under
the artifact-job wrapper shown in the root README. Existing `source` and `runtime`
profiles keep their workspace and CLI-only meanings. A stage never copies folders
recursively, provisions private fixtures implicitly, or overwrites a destination.

The engine profile retains all `src/drawing_engine/` implementations and resources,
the CLI, test-required developer tools and engine tests. App/research implementations,
private fixture inventories, historical source maps, outputs and customer files
are excluded. Their original workspace files remain available.

The manifest records excluded test modules and dependency reasons. Tests belonging
to research/workspace packaging stay with those owners. The obsolete pipeline test
still imports the retired `rebar_pdf_pipeline` prototype; it remains an explicit
workspace blocker, not an engine pass. No historical expectations are rewritten.

## Gates

`python -B deploy/source/engine_repository.py test --list` previews the gate.
The engine gate uses the existing `config/test_gates.json` classifications to
exclude private replay/live-sheet/research and render tests from the default
fast selection. Explicitly listed synthetic CLI/audit cases supplement it.
`private_test_selectors` and `private_test_requirements` additionally identify
workspace-fixture reads measured during clean-stage qualification, including
cases that the workspace currently calls fast. This is a separate repository
gate: the existing workspace gate memberships and assertions are unchanged.
These cases remain unqualified here, not skipped successes or repaired history.
Discovery errors and test failures fail the gate. Test modules retain their
original private cases and assertions even when those cases are not in this gate.

`python -B deploy/source/engine_repository.py smoke` generates its own PDF and runs
structural delivery, inspection and frozen audit replay from a temporary input
folder. It checks the standard SQLite/audit/result artifacts and preserves the
source bytes. It makes no real-drawing coverage or quantity claim. Both commands
measure output and scratch disk usage and clean their owned scratch directories.
Reports remain under ignored `data/operations/`.

Private qualification continues in the full workspace using the existing hashed
fixture provisioning, source revision bindings and replay/live-sheet gates.
The four missing historical inputs and the mismatched historical audit recorded
in `docs/validation-blockers.md` there remain open. Provisioning is never inferred
from a passing synthetic engine gate. To run an excluded case, use that workspace
lane with its exact inputs and history; do not copy customer drawings into Git.

## Git publication check

Initialize Git only in the new stage, then add its source files. Run
`python -B deploy/source/engine_repository.py check-index` before committing.
It checks exact path membership, staged bytes, executable modes, alias targets,
unmerged entries and unignored untracked files. The generated `source-stage.json`
receipt is ignored and does not belong in the index. Ignore rules are a fallback;
the manifest and index check define the publication boundary.

CI runs the index check, installs the pinned Python 3.14 dependencies, builds
Tesseract 5.5.2 from the structural deployment's verified source archive, installs
host language data/fonts, and executes the same test and smoke commands. An actual hosted CI run
requires publishing the repository; local verification does not claim one ran.
No distribution license is selected automatically. Decide licensing before a
public release; private source preparation does not grant third-party rights.
