# Working on WingSAXS

WingSAXS (`butterfly-saxs`, import `butterfly_saxs`, CLI `bsaxs`) helps users
extract butterfly-pattern SAXS trajectories and follow their evolution across
image sequences. The aim is useful, inspectable measurements from actual
intensity data, with uncertainty and model limitations attached to the results.

## How to work

For Astra and other coding agents, this document supplies project context and
decision principles, not a fixed sequence of approval gates. Read the relevant
code, tests and scientific definitions, reproduce the problem, and finish the
implementation and appropriate verification within the user's authorization.
Choose routine tools and implementation details independently. Parallelize
independent investigation where useful; coordinate edits to shared files.
Ask only when missing information changes the scientific question, scope,
acceptance criteria or permission and cannot be inferred reliably.

Prefer focused improvements to the real workflow: loading/calibration, selecting
the signal region, tracing observed intensity, inspecting local profiles,
fitting, and reviewing/exporting a sequence. Existing tests describe current
behavior; update a test when an authorized behavior change makes it obsolete,
while retaining a test for the underlying numerical or scientific failure.

## Measurements and confidence

- Fit observed intensity trajectories. Seeds, neighboring frames and explicit
  bounds guide optimization; they do not create pixels, mirror missing lobes or
  turn interpolated values into observations.
- Preserve finite fitted estimates and per-frame diagnostics when signal is weak,
  support is incomplete or an optional uncertainty assessment has not run.
  Distinguish estimates, constrained candidates, failed fits and missing data in
  the UI and exports. Explain the limitation and the next useful action.
- Quality thresholds are diagnostic heuristics unless supported by a stated
  measurement model. A warning should help the user refine the q window, mask,
  calibration or fit; it should not silently discard the result or whole frame.
- Initialization across frames is an optimization aid. Fit each new frame to its
  own data and retain source-frame information. Do not force smooth evolution
  or copy a previous fit into a failed measurement.
- Keep ring `q*` / `L = 2π/q*` separate from ellipse `a`, `b/a`, axis angle and
  conditional Ln/Lz estimates. A boundary solution, near-circle or extrapolated
  major axis can remain visible as a candidate, with its specific limitation.
- Physical lengths require calibrated physical q units; pixel-q cannot produce
  nm. Ellipse geometry alone does not determine a unique 3D lamellar structure.
  Use `docs/scientific_basis_zh.md` for the Grubb 2016/2021 definitions.
- Geometry analysis does not implicitly run `full2d`, which is a separate
  empirical intensity model. Engineering success and human review describe
  their respective operations, not automatic scientific acceptance.

Invalid array shapes, empty numerical domains and invalid calibration/weights
need actionable errors. Continue other frames where possible and retain the
failed frame's place and reason in the sequence.

## Practical reference

Python 3.11–3.13 is supported. Prefer an existing suitable environment. On
Windows use an explicit interpreter if `python` resolves to the Store stub.
For a new environment install `-e ".[all]"` with
`constraints/validation-py311-313.txt`. The desktop entry is
`启动_WingSAXS.cmd` (optional `--check`).

Useful commands, selected as appropriate:

```text
bsaxs doctor --json
bsaxs describe
bsaxs inspect INPUT --poni PONI --mask MASK
bsaxs analyze INPUT --ridge-method butterfly_curvature --butterfly-stage evaluate --butterfly-resamples 0
bsaxs batch "data/frame_*.edf" --poni PONI --mask MASK --mode warm_start -o results/batch
python -m ruff check src tests scripts
python -m pytest -q
```

CLI stdout is strict JSON (`allow_nan=False`, ASCII escaped); diagnostics go to
stderr. Exit 0 means completion without reported quality warnings; 1 preserves
results with warnings, partial failures or cancellation; 2 denotes an input,
configuration or overwrite error. Preflight and P3/P4 reports support evidence
review; they are not substitutes for fitting or scientific judgment.

Package code is in `src/butterfly_saxs/`. `cli.py` imports scientific modules
lazily; `service.py` is Qt-free; workers do not access widgets. Preserve these
boundaries and test behavior through the CLI/service/UI seam actually affected.

Keep raw inputs, PONI and masks unchanged. Write outputs separately and respect
explicit overwrite choices. `CHANGELOG.md`, `data_local/`, private literature
and local validation artifacts are not committed or uploaded. Review the staged
files before publishing.

More context: `docs/first_run_zh.md`, `docs/user_guide_zh.md`,
`docs/butterfly_arcs_zh.md`, `docs/validation/result_schema_v1.md`,
`docs/architecture_zh.md`.
