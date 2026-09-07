# Estimation, detail and MEP delivery CLI

Install dependencies using the root README. With the virtual environment active,
run this from the checkout root to make the launcher available in input folders:

```sh
export PATH="$PWD/bin:$PATH"
```

## Fresh folder and page audits

```sh
cd /path/to/drawings
estimation audit                       # every top-level PDF, every page
estimation audit page 2                # only original page 2 in each PDF
estimation audit drawing.pdf --page 2  # one PDF, one page
estimation audit --task concrete
estimation audit --task rebar
estimation audit --task mep
estimation audit page 2 --task hvac
```

Without `--output`, every invocation creates a new timestamped folder under
`./estimation-output/`. Discovery is nonrecursive and accepts `.pdf` and `.PDF`;
previous output subfolders are excluded naturally. A single input produces one
standard project. Multiple inputs produce one standard project per PDF and a
batch `result.json` linking their hashed manifests. Every input is attempted;
failed PDFs and out-of-range pages are reported, successful projects are kept,
and any failure gives a nonzero exit status. Batch execution does not combine
physical identities or aggregate quantities across PDFs.

Page selection runs extraction and independent declaration reading only on the
requested source page. The full original PDF remains preserved, native page
numbers/IDs remain unchanged, and excluded-page content supplies no constraints.
MEP retains metadata-only placeholders explicitly marked not processed for other
pages. A structural source page yields its source, interpretation and status
audit pages; MEP/HVAC produce one static overlay per source page. `page 2` selects the source
page, not the second page of the resulting audit.

The default `--task auto` reads native titles only on the selected pages before
starting extraction. When every selected page has a supported M1 mechanical
title, it selects `mep`, whose runner also builds the HVAC inventory. Unknown or
mixed page scopes prompt for `structural`, `concrete`, `rebar`, `mep` or `hvac` in
an interactive terminal. Noninteractive runs stop with an instruction to supply
`--task`; they never silently fall back to structural. This is bounded native
title detection, not a general discipline classifier. Filenames, colours and
schedule quantities do not select the pipeline. An explicit `--task` bypasses
detection and the prompt. One chosen task applies to every PDF in the invocation.
Ctrl+C exits with status 130 and a short cancellation message; existing disk
accounting and worker termination still run.

The `structural` task uses the shared concrete/rebar engine. `concrete`
and `rebar` select the consumer while retaining both channels and all shared
evidence. MEP and HVAC use fresh M0 annotation extraction, M1 registration,
native text, the existing bounded automatic discovery/binding runner and HVAC
inventory. They require no prebuilt database or historical extraction inputs.
The runner's bounded interpretation is preserved, including complete retained
native JSON search records in SQLite. Exhaustive source-denominator route
certification, full network completion, installed lengths, fitting counts,
declaration comparison and fabrication DXF remain explicitly unsupported by
this fresh MEP adapter. Running the CLI grants no new engineering authority.
Unbound HVAC equipment and sizes remain observations.
MEP/HVAC audits contain one static overlay per processed source page: a 120 dpi
background at 47% opacity and opaque extracted route strokes, with no links,
cover or close-up pages. Known systems use solid strokes; unknown systems use
dashed strokes. The original vector PDF remains in `source/`; extraction still
uses native evidence.
Overlay colours preserve matching native source strokes; colour does not identify
a system. Missing or conflicting stroke colours use an accepted system's palette,
or neutral grey when the system is unresolved.

These commands always start fresh. To redraw an existing project:

```sh
estimation replay-audit /path/to/project/result.json --output /path/to/new-audit.pdf
```

The historical explicit `audit result.json --output new.pdf` invocation remains
compatible. It cannot accept a new page selection: replay uses its frozen scope.
`python3 -B -m src.drawing_engine.cli` remains available from the repository root.

## Standard delivery

The primary store is `project.sqlite`, using the existing compact **schema v3**.
The delivery contains source PDFs, that database, `audit.pdf`, eligible DXFs and
a small `result.json`. JSON records and HTML are exported only on request.
The existing structural runners still serialize native JSON in private job-local
staging; it is imported and removed after packaging. They never generate an HTML
preview during standard delivery.

Native JSON bytes are compressed once in v3's content-addressed chunk tables.
There is no legacy v2 store, storage envelope, or duplicated per-node payload.
Original schemas, primitive IDs, page coordinates, nulls and frozen comparison
hashes survive exactly. These delivery snapshots support artifact and JSON-pointer
queries; they do not claim the independently built MEP record/edge search indexes.
The active MEP database and its indexes remain unchanged.

CLI inspection and audit generation read the pinned database in read-only mode.
Each read checks the published database hash, snapshot/source/scope binding and
artifact index; artifact streaming also checks chunk and whole-artifact hashes.
PDF source paths are portable, relative and hash-bound. Historical paths inside
native records stay historical. The source PDF, SQLite and manifest are required
for audit replay; intermediate JSON files and extraction code execution are not.

Success means the bounded project was delivered. Complete takeoff, physical
identity, installed length, fabrication release and quote approval retain their
native eligibility states. Calculated, conditional, projected, physical,
declared and approved channels remain separate.

## Concrete and reinforcement

```sh
python3 -B -m src.drawing_engine.cli compare 1.pdf --output output/cli/new-client
python3 -B -m src.drawing_engine.cli compare 1.pdf \
  --frozen-bundle output/object_agnostic/1.object-agnostic-bundle.json \
  --output output/cli/new-replay
```

The existing engine runs before independent declaration extraction. Explicit
replay preserves supplied graph bytes and re-extracts declarations; it never
upgrades a historical result to a current one. Missing or mismatched evidence
fails closed.

The audit reuses `render_object_agnostic_audit`: each source sheet receives a
clean reference, the interpretation overlay demonstrated by page 2 of the user's
reference PDF, and its solid/quantity status. PDF layers retain views, contours,
dimension chains/formulas, cutting planes, section matches, leaders/detail links,
classifications, rejected candidates, unresolved alternatives and independent
schedule evidence. Regenerating the audit reads frozen declarations too.

## Detail assembly and DXF

```sh
python3 -B -m src.drawing_engine.cli detail \
  /path/to/detail.pdf \
  --assembly EP14 --page 1 --output output/cli/new-ep14
```

The detail audit marks native plate/bar profiles, dimension chains and terminals,
part/weld leaders, named face/section evidence and unresolved source references.
Click drawing annotations for native IDs. The selected scope and incomplete
metric cutting transform stay explicit.

EP14 plate 35 closes at 210 × 240 × 15 drawing units; 0.000756 m³ remains
conditional on millimetres. Six bar-end projections and three shaft profiles
remain observations; physical bar count, diameter and fabrication length are
unresolved. Independent declarations cannot fill them.

Plate DXFs export the closed native outer face using uniquely attached width
and height dimensions with consistent face scale. Thickness, a second section
height annotation, a watertight solid and internal-symbol classification are not
outline prerequisites. Native contour and dimension provenance remain required.
Default DXF
units follow native evidence, otherwise they remain unitless. `--dxf-units mm`
explicitly assumes millimetres for the export only. Internal contours and bar
ends are excluded. This supplies no material, machining, fabrication or
quotation approval. The ASCII AC1015 DXF uses explicit INSUNITS and a closed
LWPOLYLINE. Unsupported parts retain reasons in the SQLite `dxf_report`.
The report contract is `detail_dxf.v2`, scoped to one outer face outline per file.

## One-page MEP audit

```sh
python3 -B -m src.drawing_engine.cli mep \
  'M&P mark-up against shop systems piping.pdf' \
  --database output/projects/mep/project-v3.sqlite \
  --project mep-coordination --document coordination-set \
  --page 2 --output output/cli/new-mep-page2
```

This pins the selected existing snapshot (or the active snapshot at invocation;
supply `--snapshot` for an exact historical revision). It runs no extraction.
The delivery retains the complete source PDF, a compact v3 store and **one audit
page corresponding to source page 2**. Selecting another non-divider page uses
the same path.

The existing drawing renderer preserves native PDF content and authored markup,
shows projected route records, equipment observations, recovered local
attributes and frozen outcomes, and keeps unknown geometry visually distinct.
Native record/primitive IDs are attached to marks. Projected lengths remain
separate from installed lengths. A one-page presentation cannot claim complete
network coverage or physical connectivity.

SQLite retains the frozen `mep_review` projection and exact native registry,
network, route-observation, cross-sheet, attribute, equipment, catalog and outcome
artifacts behind it. `upstream_snapshot` lists included artifacts and explicitly
identifies deep discovery query packs still available only in the original MEP
project. This bounded review delivery does not impersonate a complete database
clone or enlarge its extraction coverage.

## Inspect, replay and optional exports

```sh
python3 -B -m src.drawing_engine.cli inspect output/cli/new-ep14/result.json \
  --artifact assembly --pointer /child_parts/0
python3 -B -m src.drawing_engine.cli inspect output/cli/new-mep-page2/result.json \
  --artifact mep_review --pointer /processing_scope
python3 -B -m src.drawing_engine.cli replay-audit output/cli/new-ep14/result.json \
  --output output/cli/replayed-ep14.pdf
python3 -B -m src.drawing_engine.cli export output/cli/new-ep14/result.json \
  --format json --artifact assembly --output output/cli/ep14-json
python3 -B -m src.drawing_engine.cli export output/cli/new-ep14/result.json \
  --format html --output output/cli/ep14-html
python3 -B -m src.drawing_engine.cli export output/cli/new-ep14/result.json \
  --format obj --output output/cli/ep14-obj
```

Omit `--artifact` to export all stored JSON bytes. The optional export index maps
native artifact keys to safe filenames. HTML uses the existing detail renderer
and includes its linked source/JSON/preview assets; other modes use their richer
PDF audit. Exporting never changes the primary database.

`--format obj` exports existing frozen triangle solids into separate Wavefront
OBJ files. It defaults to `assembly` for detail projects and `engineering_graph`
for structural projects; those are the only supported OBJ artifact keys. It does
not rerun extraction. Detail filenames use the drawing's assembly and part marks,
for example `EP14-part-7.mm.obj`. Structural filenames use the source page number.
Open an exported `.obj` in your CAD/mesh application. OBJ has no standardized unit
field: choose millimetres when importing `.mm.obj`; `.unitless.obj` retains the
drawing's numeric coordinates with unresolved units, without assuming millimetres.

The export's `result.json` lists file hashes, units, source snapshot/hash and exact
geometry pointers, plus each omitted part and its reason. Its state is `exported`,
`partial` or `abstained`; a completed export command can have no OBJ files when no
supported solid exists. Missing solids, failed reprojection and invalid meshes
are never replaced with an extrusion of the DXF outline. The current adapters
cover resolved detail plate solids and resolved single structural meshes.
Structural collections with presentation-only component offsets, reinforcement
paths, preview overlays, MEP envelopes and cage assemblies are not exported.
Each file retains its own frozen relative frame; importing several files does
not establish their placement as an assembly. The export adds no quantity,
fabrication or approval authority. Keep the original source/SQLite/audit delivery
for evidence; OBJ is an optional derivative, not a replacement project delivery.

All output paths must be new. Failed jobs do not publish a success manifest.
The CLI monitors output and explicit sibling `<output>.temporary` staging with
a default 1 GiB reserve/growth budget and 3 GiB minimum free space; adjust
`--reserve-gib` and `--max-growth-gib` for larger jobs. Telemetry stays in
`data/operations/artifact-disk-usage.jsonl`. Private staging is removed on completion;
historical delivered projects are preserved.

## Validation boundary

Run the self-contained test and delivery smoke commands from the root README.
See [engine repository preparation](engine-repository.md) for the exact gate and
private qualification boundary. CLI execution, synthetic tests, real drawing
coverage and independent transfer qualification are separate results. Historical
validation checkpoints and private replay blockers remain in the full workspace;
a passing repository gate does not close them.
