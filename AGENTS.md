# Repository agent instructions

These instructions apply to the entire repository.

## Mission

Build a procedural PDF drawing-understanding engine serving concrete and
reinforcement estimators and MEP/HVAC teams. Shared evidence-backed understanding
supports discipline-specific geometry, identity, attributes, connectivity and,
where independently eligible, quantities. A bounded interpretation capability
is a valid product output without quantity authority. The drawing is the
calculation source of truth. Designer-entered tables such as `Ведомость бетона`
are declarations to verify, never prediction inputs.

Deliver the engine as callable modules with a CLI execution/query/export surface;
team-specific apps build on that contract. Capabilities must be testable without
an app. Reuse the existing runners and evidence/package contracts, and preserve
unsupported stages explicitly. Apps never own duplicate interpretation or
quantity logic. Detail/assembly extraction serves both estimation and downstream
fabrication candidates; production exports retain separate approval gates.

Every extraction task uses the standard project delivery contract in
`ROADMAP.md` ("Standard extraction project delivery"): preserved source PDFs,
the existing `ProjectKnowledgeStore` in `project.sqlite`, a marked `audit.pdf`,
applicable discipline exports such as DXF, and a hashed `result.json` index.
Native JSON remains inspectable. Reuse the shared CLI packaging; research JSON
alone is an intermediate, not a completed client delivery. An unsupported export
must retain an explicit reason. DXF unit assumptions never change drawing-derived
facts, and a geometry export never grants fabrication or quote approval. New
capabilities must reach the same project contract; document migration gaps for
existing runners instead of claiming they already comply.

The audit acceptance reference is page 2 of
`output/pdf/1_object_agnostic_audit.pdf`: show the engine's recorded interpretation
on the source drawing, including applicable views, contours, dimension chains,
cross-view/part/route relations, accepted versus unresolved/rejected candidates,
and the evidence behind results or abstentions. Reuse frozen SQLite/source IDs
and existing renderers. Totals, generic status pages or part boxes alone are
insufficient. Render only actual stored evidence and checks; never invent a
successful reasoning trace. See the audit requirement in `ROADMAP.md`.

The estimation consumer flow is:

1. Sales receives a PDF and creates an estimation job.
2. The pipeline calculates quantities from drawing geometry and annotations.
3. Declared schedule quantities are extracted independently after the drawing
   calculation is frozen.
4. The system compares calculated and declared values and presents evidence,
   deltas, severity, and unresolved facts to an engineer.
5. Only reviewed drawing-derived quantities are exported for a commercial
   proposal.

Never collapse `calculated`, `declared`, and `approved` into one value.

## Active architecture

- `src/drawing_engine/disciplines/detail/native_cage_assembly.py` handles one
  caption-bound native ladder paint with complete source coverage and unique
  child leaders. Its counts are projected outlines and parent annotation targets,
  never physical multiplicity or cutting lengths. `cage_document_links.py`
  reads independent ruled declarations after geometry freezes and binds one
  explicit reference page through actual sheet, document/change and child-mark
  evidence. The `cage` CLI uses the shared SQLite/source/audit delivery; unsupported
  DXF and quantity authority remain explicit.

- `src/drawing_engine/core/object_agnostic_understanding.py` builds generic views, contours,
  mutually exclusive text roles, claims, relations, and solid hypotheses.
- `src/drawing_engine/pipelines/generate_object_agnostic_bundle.py` writes three layers:
  immutable observation graph, compact engineering graph, and evidence store.
- `src/drawing_engine/audit/render_object_agnostic_audit.py` creates the marked human audit.
- `src/drawing_engine/disciplines/concrete/schedule_comparison.py` hashes the frozen engineering graph, extracts
  declared schedule totals independently, and computes discrepancies.
- `src/drawing_engine/pipelines/generate_estimate_comparison.py` and
  `tools/render_estimate_comparison_audit.py` create comparison records
  and marked schedule/discrepancy audits.
- `deploy/source/manifest.json` defines the curated source and CLI-only runtime
  staging profiles. `src/drawing_engine/` owns engine code by responsibility;
  `deploy/source/engine-manifest.json` separately defines the engine-only Git
  repository. Its canonical shared history is
  `https://github.com/StepanKukharskiy/drawing-analysis-engine`, branch `main`.
  See `docs/engine-repository.md` for publication and workspace boundaries.
  Use `stage_source.py --profile engine` and verify its Git index
  with `engine_repository.py check-index`; app/research and private inventories
  are excluded. Its synthetic test gate is not private replay qualification.
  `apps/review/` owns app source, `tools/` developer commands and `research/`
  experimental code. Flat `src/*.py`, `src/runtime/` and `experiments/` are
  retired; do not recreate them or add callers. Keep engine code in `src/drawing_engine/`.
  Historical source keys belong only to frozen replay records. New code and
  callers use the current owners directly.
  `src/drawing_engine/resources/structural_runtime_files.txt` defines structural fingerprint coverage.
  Private qualification inputs are provisioned separately with exact hashes;
  staged runtime code must not read unrelated workspace folders. See
  `deploy/source/README.md`. Data relocation and historical cleanup remain
  separate from source preparation.
- `src/drawing_engine/disciplines/concrete/semantic_3d_solver.py` is an optional constrained solid solver. Its
  failure must not invalidate the object-agnostic graph.
- `src/drawing_engine/disciplines/concrete/generic_prismatic_solver.py` closes a drawing-neutral subset when two
  disjoint metric native-vector profiles uniquely define a thin watertight
  rectangular prism. It must abstain on ambiguous profile matches.
- `src/drawing_engine/disciplines/concrete/profile_physical_scope_binding.py` binds a resolved scoped profile to a
  Step 2 relative physical scope only when the accepted cutting relation,
  shared-coordinate scope, passed reprojection, and exact title-scope (or
  unique unsplit source-view) membership all close. The binding is only a
  Step 5 reconstruction input: it never establishes an additive component,
  physical transform, solid, or quantity.
- `src/drawing_engine/core/dimension_attachment.py` uses native numeric text first and runs OCR
  only inside label zones gated by complete vector dimension chains. OCR
  confidence and coordinates must remain explicit evidence.
- `src/drawing_engine/core/metric_equation_graph.py` binds arithmetic dimension formulas to
  complete native chains before view-frame and solid solving. Equal numbers or
  correct arithmetic alone never create an object scope; accepted scopes need
  redundant shared extents across multiple views and remain ineligible for
  quantity aggregation until physical identity and reprojection close.
- `src/drawing_engine/core/vector_topology.py` assigns page-global native segment and vertex IDs,
  including compatible-style components spanning multiple PDF drawing paths.
  Dimension ownership, contour correspondence, and audit provenance must
  reuse these IDs rather than create independent geometry references.
- `src/drawing_engine/core/contour_correspondence.py` intersects parent-view native edges at
  accepted cutting planes and compares the resulting metric path coordinates
  with section contours. It accepts only a unique correspondence and
  preserves mirrored sign ambiguity; bounding-box or span agreement alone is
  never a physical transform.
- `src/drawing_engine/disciplines/rebar/rebar_path_graph.py` runs before solid/object solvers and preserves
  generic line/curve fragments, connections, repetition, local metric
  projected observations, and exact provenance links. A projected component
  is never a physical or cutting length without the required identity and
  dimensionality checks.
- `src/drawing_engine/disciplines/rebar/projected_bar_composite.py` groups only uniquely paired,
  near-coincident, parallel, style-compatible same-mark components as multiple
  native strokes of one projected bar representation. Every source fragment
  remains immutable. Distant or non-unique pairs remain review candidates;
  accepted composites do not establish physical placement identity, additive
  count, installed length, or cutting length.
- `src/drawing_engine/disciplines/rebar/physical_bar_family_solver.py` may fold paired outlined section legs
  into physical instances only when the exact-mark linked fabrication detail
  has a closed two-leg topology. Preserve the observed leg count separately;
  an odd leg count must abstain. Ordinary filled end projections and repeated
  placements are not divided by a shape convention.
- Evidence-backed compound reinforcement callouts outside table grids may seed
  physical families without a fabrication-detail table. Their diameter/length
  code and paired count/spacing fields remain explicit observations. Multiple
  same-mark occurrences must abstain until the graph proves whether they are
  additive placements or duplicate projections; a profile interpretation of
  the length code remains `estimated`, never strict drawing quantity.
- `src/drawing_engine/core/raster_fallback.py` quality-routes pages between native processing,
  text-only OCR augmentation, and raster primitive proposals. Pixel results
  are observations with explicit coordinates, confidence, transforms, and
  model provenance; they are never engineering facts by themselves.
- `src/drawing_engine/disciplines/mep/mep_sheet_registry.py` is the isolated M1 sheet/package and floor-grid
  registry. It preserves title-field and quality-route evidence, keeps
  one-sided OCR grid rows as candidates, and accepts a cross-sheet transform
  only from redundant named-axis groups with scale-consistent low-residual
  replay. Registration never establishes route identity, physical
  continuation, clash, or quantity.
- `src/drawing_engine/disciplines/mep/mep_sheet_region_ownership.py` certifies page-local drawing regions
  before MEP route promotion. It uses edge anchoring, repeated cross-sheet
  geometry, ruled-grid topology, native text roles, and view-title evidence;
  title blocks, borders, schedules, legends, and other excluded geometry are
  preserved as `non_route_drawing_content`. `src/drawing_engine/disciplines/mep/mep_region_owned_takeoff.py`
  requires accepted drawing-view ownership and accepted system identity for
  the published semantic centreline channel. Unknown geometry remains an
  unresolved valid-view candidate and never becomes identified pipe.
- `src/drawing_engine/disciplines/mep/mep_audit_trace_network.py` is the audit-only complete-occurrence trace
  adapter over the region-owned ledger. It renders every retained route
  occurrence, groups page-local observations under compact trace IDs, and
  supplies reciprocal mark-to-schedule links. A native stroke colour may
  recover a dashed engineer-review trace only when accepted semantic anchors
  establish a unique one-to-one colour/system mapping; that correlation never
  certifies system identity, physical connectivity, installed length, count,
  or quantity. Exact native shared endpoints may be reported as projected
  topology components while crossings and physical continuity remain open.
- `src/drawing_engine/disciplines/mep/mep_terminology_proposals.py` is the isolated M2 versioned terminology
  and graphical-proposal layer. It preserves method/pack provenance,
  alternatives, conflicts, and evidence channels. Ambiguous, conflicting,
  legend, and color-only interpretations abstain; proposals never establish a
  route, connection, elevation, clash, installed length, or quantity.
- `src/drawing_engine/disciplines/mep/mep_route_observations.py` is the isolated M3 routed-system observation
  graph. It preserves native/raster primitive provenance, page-scoped stable
  fragment and vertex IDs, endpoints, gaps, branches, unconnected crossings,
  repetition, style, M1 ownership, and local projected metric observations.
  It never establishes physical continuation or emits installed length,
  fitting count, or quantity.
- `src/drawing_engine/disciplines/mep/mep_source_primitive_denominator.py` is the compact, exhaustive native
  source denominator required before M3 route certification. It pairs every
  fixed-width native descriptor with exactly one primary disposition while
  preserving competing role bits and sheet-region ownership. M3 may expand
  route-evidence rows from this pack, but envelope or leader discovery never
  decides which native source geometry exists. Legacy M3 membership is retained
  only as `legacy_m3_route_candidate`; it cannot become the denominator's
  primary `route_evidence` disposition without independent source evidence.
  `src/drawing_engine/disciplines/mep/mep_precision_near_joins.py`
  publishes exact joins, strictly bounded per-view/style near joins, ambiguous
  candidates, and disconnected crossings without globally increasing the
  native snap tolerance or establishing physical continuation or quantity.
- `src/drawing_engine/disciplines/mep/mep_anchored_route_recovery.py` builds bounded projected component
  candidates from accepted outlined-route seeds using two-sided exact or
  mutually unique near-join evidence, compatible non-colour style, corridor
  width, and accepted page ownership. It freezes geometry before applying M4;
  colour never selects membership, and branches, crossings, body boundaries,
  one-sided contacts, and ambiguous transitions remain explicit stops.
- `src/drawing_engine/disciplines/mep/mep_native_subpaths.py` reconstructs exact native line subpaths before
  version-2 corridor/body partitioning. Authored paint membership remains intact;
  cross-paint serialized chains require exact unique incidence, verified
  single-segment consecutive paint records, matching native style and compatible
  direction. Only a bounded monotone transverse passage can explain internal
  microsegment vertices. Caps, closed bodies, branches, ambiguous contacts and
  unsupported cubic roles remain blockers. This is projected stroke handling,
  never pipe identity, a physical connection or permission to publish an audit.
- `src/drawing_engine/disciplines/mep/mep_outlined_route_composites.py` is the generic M3-to-M4 outlined-route
  certificate layer. It preserves both native outline strokes and publishes a
  derived, quantity-ineligible page-local centreline/corridor only when
  provenance, non-color style, ownership, persistent separation, synchronized
  turns/endpoints, explicit envelope closure, and mutual uniqueness all close.
  Parallelism or distance alone never merges adjacent routes.
- `src/drawing_engine/disciplines/mep/mep_component_evidence_classification.py` classifies frozen projected
  components as identified MEP routes, independently supported unidentified MEP
  candidates, unclassified view geometry, or non-route drawing content.
  Outlined-pair geometry alone remains unclassified. Applicable M4 annotations,
  accepted fitting/equipment interfaces, or accepted anchored-style correlation
  may support candidacy, while only a unique accepted M4 system binding names a
  system. Negative-promotion gates must contain measured source references or
  remain explicitly open; missing classifiers never receive literal zero counts.
- `src/drawing_engine/disciplines/mep/mep_cross_sheet_runs.py` is the M5B cross-sheet run-hypothesis layer. A
  continuation requires an accepted M1 registration, compatible M4 system,
  size, and elevation, explicit endpoint-bound evidence, transformed endpoint
  coincidence, transform-cycle consistency, and mutual uniqueness. Tiled
  overlap geometry is canonicalized through the same attribute and uniqueness
  gates before run assembly. Projected occurrences, canonical segments,
  resolved 3D centreline segments, and unresolved vertical spans remain
  separate and quantity-ineligible; M7 owns installed length and takeoff.
- `src/drawing_engine/disciplines/mep/mep_network_assembly.py` is the bounded M5C projected hierarchy adapter.
  It replays M5B and M3.5, retaining certified outlines independently of missing
  M4 attributes. Single strokes still require independent M4 eligibility;
  their joins require exact native endpoint topology and compatible non-color
  style. Missing attributes and semantic conflicts remain separate overlays.
  Branches split runs; outline endpoints, proximity clusters and crossings never
  supply centreline joins. `src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py` independently
  replays complete native corridor searches and certified endpoint interfaces
  for named 2D scopes. Scoped projected completion never establishes a complete
  physical run, installed length, fitting count or quantity.
- `src/drawing_engine/disciplines/mep/mep_hvac_inventory.py` is an HVAC review adapter over frozen M2/M4.
  Unbound equipment tags, duct sizes and accessories remain observations;
  existing M4 equipment port gates are never bypassed. Duplicate tags remain
  distinct, unsupported classes stay explicit, and the register creates no
  M7 occurrence, physical identity, count or quantity.
- `src/drawing_engine/disciplines/mep/mep_projected_identity_bindings.py` is the serial M4 companion for
  independently replayed page-local 2D equipment body/tag and inferred fitting
  connections. It never relaxes `equipment_endpoint`, propagates elevation,
  creates M7 occurrences or establishes physical identity. M5C may expose its
  geometry-only segments with unknown attributes after fitting replay; semantic
  conflicts, class alternatives and physical-continuity failures remain explicit.
- `src/drawing_engine/core/pdf_native_metadata.py` preserves exposed PDF layer, visibility and
  rendering-context evidence without changing paint-path IDs. Layer names,
  form resources and rendering groups never establish physical object identity.
- `src/drawing_engine/disciplines/mep/mep_document_links.py` binds native structured reference markers to
  unique actual detail/riser anchors. Reference identity and item applicability
  are separate; missing destinations abstain. Independent fixture projects and
  source snapshots never enlarge another document's coverage.
  Its duct station companion accepts a plan/section station transform only from
  one mutually unique metric relation, a uniquely sized native route interval,
  and at least two physically identified native landmark pairs spanning the
  route. Matching tag text alone remains a candidate.
- `src/drawing_engine/project/project_knowledge_store.py` persists content-addressed project snapshots,
  compressed immutable artifacts, record/reference indexes and exact snapshot
  review overlays in SQLite. Imports are transactional; context/input changes
  replace only the active pointer while retaining auditable history. Storage
  and queries never merge physical identities, accept candidates or grant quote
  approval. `tools/mep_project.py` provides the build/import/query CLI.
- `src/drawing_engine/disciplines/mep/mep_duct_evidence.py` keeps duct-specific evidence inside the existing
  M2/M4 flow. A size applicability certificate requires native `W x H` text, a
  complete direct leader or inline association, one accepted native interval,
  an explicit rectangular-duct unit, and an applicable outside-dimension
  convention. It never establishes a 3D envelope, continuation, or quantity.
- `src/drawing_engine/disciplines/mep/mep_bounded_local_3d.py` is the isolated M5A bounded-local-geometry
  layer. It promotes one M5-canonical projected segment only when accepted M1
  scale/registration, M3.5 measured outline geometry, compatible M4
  system/size/elevation, and every deduplicated occurrence agree. Nominal size
  never supplies physical outside diameter. Bottom elevation resolves a local
  pipe centreline only by adding half the drawing-derived outside width.
  Terminal boundaries remain unresolved analysis caps; the layer establishes
  no physical continuation or run, installed length, fitting count, clash,
  severity, takeoff, or quantity.
  Its relative rectangular-duct companion requires accepted native size
  applicability, typed bounded-route assembly, and plan/section station
  correspondence before sweeping evidence-backed portals. It emits only
  `resolved_relative_centerline_length_m`; installed, net-material, and purchase
  lengths remain downstream M7 authority. Verification faces at unresolved ends
  are explicitly nonphysical analysis boundaries.
- `src/drawing_engine/disciplines/mep/mep_attribute_binding.py` is the serially owned M4 page-local binding
  layer. It builds maximal projected route scopes that stop at M3 branches,
  binds M2 proposals only through unique page/scope and explicit geometric
  target evidence, and certifies system, size, elevation, equipment-port,
  valve, fitting, damper, riser/drop, service-zone, and continuation-endpoint
  relations independently. Size and elevation applicability split at accepted
  change points. Continuations remain endpoint relations only; cross-sheet
  joining, installed length, fitting counts, clash status, and quantities are
  prohibited.
- The existing `src/drawing_engine/disciplines/mep/mep_automatic_target_binding.py` and
  `src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py` share replayed native stroke-role
  evidence for dot leaders, ticked dimension chains and a bounded transverse
  two-anchor body motif. Complete source queries, current route boundaries and
  exact annotation scopes remain explicit. Identical geometry under different
  native IDs cannot hide dual use. Ambiguous roles, open arms and unexplained
  interior ink remain competitors; no source stroke is deleted. A role can
  explain an obstruction, never grant annotation applicability, physical
  support identity, a port, continuity, elevation propagation or quantity by
  itself.
- `src/drawing_engine/disciplines/mep/mep_declared_data.py` is the independent M7C document-region and
  declared-field layer. It classifies M1-bound native region observations and
  extracts manufacturer, model, specification, mounting, and declared quantity
  only from explicitly complete region bodies. It never reads M4, M5, M7A, or
  M7B calculated records; a heading-only negative cannot establish that a field
  is absent from the document or emit reconciliation.
- `src/drawing_engine/project/takeoff_intelligence.py` is the shared concrete, reinforcement, and MEP
  takeoff contract. Discipline adapters preserve native records while exposing
  common occurrence, physical-item, calculated-line, and declared-line roles.
  A comparison requires an explicit mutually unique scope-match certificate;
  empty channels never create comparisons, and approval is a separate exact
  engineer overlay. `src/drawing_engine/disciplines/concrete/concrete_takeoff_adapter.py`,
  `src/drawing_engine/disciplines/rebar/rebar_takeoff_adapter.py`, and `src/drawing_engine/disciplines/mep/mep_takeoff_adapter.py` only project
  authority already closed by their native discipline graphs.
- `src/drawing_engine/disciplines/mep/mep_claim_grounding.py` is the isolated M6A markup-grounding layer. It
  preserves immutable M0 claim provenance and may bind a claim only to one
  uniquely overlapping, page-local composite that has accepted M4 relations
  and an M5 partial 2.5D centreline. Multiple overlaps remain
  `2d_overlap_candidate` records and missing page-local targets or geometric
  overlap abstain explicitly. M6A never emits `confirmed_clash`, calculated
  severity, installed length, M7 takeoff, or quantity; full M6 verification
  requires independently resolved 3D geometry.
- `src/drawing_engine/disciplines/mep/mep_bounded_clash_verification.py` is the isolated bounded M6B layer.
  It may certify a local clash or bounded clear result only when one M6A
  two-target overlap maps uniquely to two independent M5A envelopes in the
  same accepted metric frame and the closest intersection is outside both
  unresolved analysis-cap zones. Measurement uncertainty remains explicit.
  Clearance and access require separate explicit zone geometry. M6B never
  establishes a whole run, physical continuation, installed length,
  calculated severity, M7 takeoff, or quantity.
- `src/drawing_engine/disciplines/mep/mep_item_catalog.py` is the isolated M7A structured-item adapter. It
  preserves all registered M1 pages and creates item occurrences only from
  accepted M4 route, equipment, or accessory relations. Optional M5A data may
  add a drawing-derived physical dimension and local elevation, never a
  projected or installed length. Manufacturer, model, specification, and
  mounting fields remain explicit `null` values when unobserved; observed,
  deduplicated, and physical counts and calculated, declared, reviewed, and
  approved values remain separate. M7A establishes no physical count,
  calculated quantity, installed length, or quote authority.
- `src/drawing_engine/disciplines/mep/mep_occurrence_coverage_audit.py` freezes the reviewed M7A-to-M7B
  coverage boundary across every registered page. Accepted M4 occurrences are
  distinguished from authored markup hints that lack an M4 target. A reviewed
  hint may record an unresolved equipment, valve, fitting, damper, or fixture
  category, but it cannot become an occurrence, establish absence, physical
  identity, or count.
- `src/drawing_engine/disciplines/mep/mep_bounded_discrete_counts.py` is the isolated M7B discrete identity
  and count layer for accepted M4 equipment, valve, fitting, and damper
  relations. It requires one M4 target per occurrence plus explicit physical
  identity and duplicate-projection evidence. Cross-sheet duplicates require
  accepted registration, matching target signatures, mutual uniqueness, and
  resolved repetition. Observed, deduplicated, physical, and calculated counts
  remain separate; abstentions keep every count null. M7B never emits projected,
  bounded-local, or installed route length, declared values, or quote approval.
- `src/drawing_engine/disciplines/mep/mep_marketplace_assembly.py` is the serial physical-run and marketplace
  assembly gate downstream of M5C. It keeps projected, resolved-centreline,
  net-material, and purchase lengths in separate channels; replays complete
  segment/occurrence/port coverage before physical promotion; and derives
  stock rounding or implicit connector items only from versioned rule packs.
  Catalog resolution is staged from generic requirement to exact SKU, and
  missing material, specification, size, connection, or manufacturer facts
  remain unresolved. The layer never mutates M5C, consumes declarations as
  geometry, or infers review/quote approval.
- `src/drawing_engine/disciplines/mep/mep_physical_run_evidence.py` is the fail-closed readiness and automatic
  physical-run evidence compiler between M4/M5A/M5B/M5C and marketplace
  assembly. Readiness states only prioritize investigation and never grant
  authority. A positive currently requires one bounded straight M5C segment,
  one accepted M5A XYZ centreline/envelope, complete projected topology, two
  uniquely typed M4 terminals, exact port/occurrence coverage, and a connection
  standard. It copies frozen geometry and must abstain on multi-segment joins,
  incomplete source searches, ambiguous terminals, or missing 3D coverage.
- `src/drawing_engine/core/leader_target_ranker.py` may rank existing evidence-backed
  leader-to-rebar candidates. Its learned score cannot accept identity or
  bypass the deterministic geometry and uniqueness gate.
- `research/leaders/synthetic_graph_pdf.py` and
  `research/generate_synthetic_leader_corpus.py` provide known-truth PDF
  plotting invariance cases. Conflicting schedules remain forbidden features.
- `src/drawing_engine/core/object_instance_assembly.py` groups repeated large/compact metric
  projection motifs into auditable physical-object hypotheses and scopes
  cross-view rebar matching so candidates cannot jump between object groups.
- `src/drawing_engine/core/shared_coordinate_system.py` propagates a stable relative XYZ gauge
  through accepted cutting planes and orthogonal view pairs, validates shared
  metric spans, and scopes cross-view rebar identity. A relative gauge is not
  a physical datum; unresolved axis signs and origins must remain explicit.
- `src/drawing_engine/disciplines/concrete/object_instance_extrusion_solver.py` is a generic candidate solver for a narrow, explicitly gated subset
  per object instance: a closed metric profile paired with a dimensioned
  three-sided depth outline, with reprojection, watertightness, and volume
  agreement gates before concrete aggregation. Do not describe this subset as
  universally validated merely because it contains no filename dispatch.
- `src/drawing_engine/core/native_vector_detail_linking.py` detects detail tables from repeated
  native grids plus non-grid polylines, OCRs only mark cells, and accepts every
  independently validated placement only when scale-tolerant mark identity
  closes through a native leader trace onto a unique or coherent projected
  rebar path. Drawing-specified bracket conventions may bind grouped marks to
  object-instance designations; printed schedule totals remain excluded.
- `src/drawing_engine/core/automatic_adjudication.py` deterministically accepts candidates only
  when typed graph, metric, ownership, exact-mark, or closed relation-triangle
  certificates succeed; it rejects hard contradictions and preserves all
  remaining ambiguity as explicit abstentions. Its compact delta is replayed
  over the frozen canonical graph and never substitutes a score for closure.
- `src/drawing_engine/project/review_feedback.py` binds engineer decisions to a frozen canonical
  graph hash and stable candidate IDs. Review deltas are separate overlays:
  they never rewrite observation evidence, convert a reviewed fact to
  `direct`, or consume schedule values.
- `src/drawing_engine/project/placement_pair_review.py` creates graph-bound review candidates from
  repeated normalized marks only when both occurrences already have unique
  projected-path targets. Equal text never decides additive placement,
  duplicate projection, split fragments of one projected bar, separate-object
  reuse, or invalid-mark outcomes. Disjoint targets in one view do not prove
  additive placement because one bar may be split into multiple vector paths.
  Shared membership in one uniquely accepted projected-bar composite may
  certify `same_projected_bar_fragments`; membership in distinct composites
  still does not prove additive placement.
- `src/drawing_engine/project/review_evaluation.py` measures semantic task performance from reviewed
  candidates, including wrong automatic accepts and unnecessary abstentions.
  Conflicting reviews remain excluded until resolved.
- `output/object_agnostic/` and `output/pdf/*_object_agnostic_audit.*` are the
  current generated examples.
- `output/estimates/` and `output/pdf/*_estimate_comparison_audit.*` are the
  independent declared-versus-calculated examples.
- `archive/` contains superseded experiments and outputs. Do not import active
  code from it or modify it unless the task explicitly concerns historical
  results.

## Engineering truth rules

- Preserve native primitive IDs and page coordinates through every claim.
- Use states consistently: `direct`, `observed`, `derived`, `inferred`,
  `convention_dependent`, and `unknown`.
- A schedule/table value may be extracted for comparison but must not constrain
  drawing-derived geometry, counts, lengths, diameters, or material quantities.
- Freeze drawing-derived results before schedule comparison.
- Never invent a unique solid, bar identity, diameter, grade, mass, or length
  when the drawing does not determine it.
- Graphically measured rebar diameter remains `convention_dependent` until the
  drafting convention or an explicit annotation confirms it.
- Automatic group labels are not bar marks. A mark requires an evidence-backed
  leader and cross-view identity chain.
- Concrete volume requires a watertight, constraint-valid solid. Rebar mass
  requires defensible centerline length, diameter, and material density/grade.
- Keep per-element results before job-level aggregation so repeated objects and
  cross-page references can be audited.

## Drawing-neutral implementation

- Do not branch on filenames, sheet IDs, K1/K7 names, reviewed coordinates, or
  fixed page columns.
- Prefer geometric relations, local scale consensus, and provenance-preserving
  graph operations.
- Native PDF vector/text extraction is primary. OCR/OpenCV is a fallback that
  adds observations to the same schemas; it is not a separate truth model.
- Object-specific solvers may rank hypotheses, but the observation and compact
  engineering graphs must remain object-agnostic.

## Ponytail principles

Apply the Ponytail decision ladder after tracing the real affected flow:

1. Does new code need to exist? If not, skip it (YAGNI).
2. Does the repository already contain the required helper or pattern? Reuse it.
3. Can the Python standard library solve it? Use that.
4. Can the native PDF/platform capability solve it? Prefer it.
5. Can an already-installed dependency solve it? Reuse it; do not add another.
6. Can the change be one clear line or a local edit? Keep it that small.
7. Only then write the minimum new code that completely solves the requirement.

Minimize code, abstractions, dependencies, files, and configuration—not
correctness. Never simplify away trust-boundary validation, provenance,
fail-closed behavior, data-loss protection, security, or accessible audits.

## Working practices

- Read the relevant flow before editing and keep diffs local.
- Use `rg` for code and file discovery.
- Use `apply_patch` for source and documentation edits.
- Preserve user data. Archive superseded material rather than deleting it when
  historical comparison may matter.
- Do not hand-edit generated JSON or PDF outputs; regenerate them from scripts.
- Put transient renders under `tmp/`; do not commit caches or `.DS_Store` files.
- Every artifact-producing job must record disk usage before and after (including
  failed/interrupted runs): logical and allocated output bytes, growth, retained
  job-local temporary bytes, and remaining free bytes on each involved filesystem.
  Use `src/drawing_engine/operations/run_artifact_job.py` for producers without complete built-in
  accounting; explicitly list output and job-local staging/render paths. Shared
  MEP JSON writes, direct-v3 artifact writes and diagnostic package stages also
  emit operational records through `src/drawing_engine/operations/artifact_disk_usage.py`.
- Disk telemetry stays outside canonical artifacts/hashes in
  `data/operations/artifact-disk-usage.jsonl` (override with `REBAR_DISK_USAGE_LOG`). Nested
  measurements overlap; never sum them as exclusive storage. Do not claim
  temporary coverage when its paths were not measured. Default free-space floor
  is 3 GiB; retain any stricter migration floor. Before a large job, declare
  expected peak output/WAL/temp reserve and an allocated-growth budget, then use
  the monitored wrapper. Sampling is not a filesystem quota or reservation.
- Report artifact size/growth and free disk at handoff. Flag unexpected retained
  intermediates for explicit cleanup; never automatically delete source PDFs,
  evidence, reviews, snapshots or user files. See `docs/artifact-disk-accounting.md`.
- Run focused tests while developing:

  The `tools/run_test_gate.py` commands below belong to the full workspace.
  In an engine-only checkout, use `python3 -B deploy/source/engine_repository.py test`
  for the self-contained engine gate and `smoke` for CLI delivery. Its gate and
  private qualification limits are in `docs/engine-repository.md`. Private replay
  remains a full-workspace operation; never report it passed from an engine-only
  checkout or recreate missing private inputs from current predictions.

  ```bash
  python3 -B tools/run_test_gate.py focused --match 'test_module.Class.test_case'
  ```

- Choose validation from the affected flow, not from the fact that a request
  is ending. Read-only audits, discussion, roadmap and documentation-only
  edits require no Python suites. For local code changes, run focused tests
  covering changed behavior and its relevant callers, including ambiguity,
  provenance and authority boundaries where affected.
- Run only affected `replay` cases when frozen evidence, contracts or their
  consuming logic changes;
  affected `live-sheet` fixtures for extraction, topology, dimensions, view
  segmentation, cross-view identity or end-to-end graph construction; and
  affected app/`render` checks for output changes. Every gate accepts repeated
  `--match` filters; preview the selected tests with `--list`. A CLI argument
  or documentation edit alone does not require rerunning drawing extraction.
- Run full `fast` plus `replay` for release/integration qualification, explicit
  requests, or changes to shared graph/schema/identity/geometry behavior whose
  impact cannot be bounded. Explain that trigger before starting a broad run.
  Reserve `full-corpus` for nightly/release validation or explicit requests;
  `full-sheet` remains a compatibility alias. Do not repeat passed checks
  without a new change, failure or unresolved concern that justifies them.
- At handoff, state what was tested and the limits of that selection. A scoped
  pass is not a full-suite pass. Preserve known blockers; do not hide them by
  weakening assertions, changing gate membership or treating missing fixtures
  as passes. See `docs/validation-policy.md` for examples and scope decisions.
- Every gate writes per-test timings and runtime warnings under
  `data/operations/test-timings/`. Do not move a real-sheet test into `fast` to hide a
  budget warning.

- For PDF output changes, render every final page and inspect it before handoff.

## Current product priority

Prioritize requested, independently qualified engine capabilities over isolated
detectors. Concrete, reinforcement, MEP and HVAC are product consumers; the
current execution order and remaining blockers live in `ROADMAP.md`. Each slice
must name its consumer output, evidence boundary and acceptance test. Distinguish
missing source facts or independent review truth from missing implementation;
do not retune exposed drawings indefinitely or claim transfer from neutral code
alone. Interpretation outputs retain their own value and eligibility boundaries.

For the estimation consumer, preserve the complete loop:

1. drawing-derived concrete and reinforcement calculation;
2. independent declared-schedule extraction;
3. line-level discrepancy detection and severity;
4. engineer review with marked evidence;
5. quotation-ready aggregation for sales.

For the detailed workflow and acceptance boundary, see
`docs/client-estimation-workflow.md`.
Product and research priorities are maintained in `ROADMAP.md`.
