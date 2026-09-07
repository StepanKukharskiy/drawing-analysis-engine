# Drawing engine

A callable PDF drawing-understanding engine with a CLI for concrete,
reinforcement, detail/cage and bounded MEP/HVAC workflows. Native drawing
evidence determines geometry and interpretation. Declared schedule values,
calculated results and engineer approval remain separate.

## Setup

Use Python 3.14. From the checkout root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r deploy/source/requirements.lock
export PATH="$PWD/bin:$PATH"
estimation --help
```

Install Tesseract and its English/Russian language data for OCR fallback.
The qualified OCR version is 5.5.2; CI builds that version from a verified source
archive. Other versions may change recognition and require requalification.
On macOS use `brew install tesseract tesseract-lang`. On Ubuntu follow the
pinned OCR build and language/font installation in the [CI workflow](.github/workflows/engine.yml).
The optional VLM models and their environment are not engine dependencies.
Keep the virtual environment active when running the launcher from another folder.

## Run

```sh
estimation audit /path/to/drawing.pdf --task structural --output /path/to/new-project
estimation detail /path/to/detail.pdf --assembly "EXACT DRAWING MARK" --output /path/to/new-detail-project
estimation inspect /path/to/new-project/result.json --artifact comparison
estimation export /path/to/new-detail-project/result.json --format obj --output /path/to/new-obj-export
```

In a project/drawing folder, `estimation export --format obj` discovers its saved
project and writes to a new `obj-export` folder (numbered if one already exists).

Use a fresh output directory. A delivery preserves the source PDF, `project.sqlite`,
marked `audit.pdf`, applicable exports such as DXF, and the hashed `result.json`
index. Unsupported exports and unresolved geometry retain explicit reasons.
Execution or geometry export never grants fabrication or quote approval.
See the [CLI contract](docs/estimation-cli.md) for commands and scope limits.

## Validate and package

```sh
python -B deploy/source/engine_repository.py test
python -B deploy/source/engine_repository.py smoke
```

The engine gate runs the manifest's synthetic/core tests without private fixture
provisioning. The smoke check creates its own PDF and runs CLI delivery, inspection
and audit replay. It requires no customer drawings or historical output folders.
Neither gate establishes complete real-drawing coverage. Private replay/live-sheet
qualification remains separate; see [repository preparation](docs/engine-repository.md).

The source lives in `src/drawing_engine/`; `bin/` owns the launcher, `tools/`
contains required developer commands, and `tests/` contains engine tests.
Resources and the small shipped model live inside the engine. The explicit
`deploy/source/engine-manifest.json` defines the repository contents.

From the full development workspace, prepare a new source-only directory:

```sh
python3 -B src/drawing_engine/operations/run_artifact_job.py \
  --output /absolute/path/to/new-engine-repo --reserve-gib 0.05 \
  --max-growth-gib 0.05 -- python3 -B deploy/source/stage_source.py \
  --profile engine --destination /absolute/path/to/new-engine-repo
```

Initialize Git in that staged directory when ready. Before committing, run
`python -B deploy/source/engine_repository.py check-index` to verify that the
index contains exactly the manifest's files, bytes and executable/alias modes.
Customer data, private fixtures, app source, research and operational receipts
are outside this repository selection.
