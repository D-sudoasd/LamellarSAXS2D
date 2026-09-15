# LamellarSAXS2D agent notes

This file is the machine/operator contract for coding agents. Human scientific
scope remains in `docs/scientific_basis_zh.md`. Do not commit `CHANGELOG.md` or
`data_local/`; both are local-only (see `.gitignore`).

## What this software is

LamellarSAXS2D (`butterfly-saxs`, import `butterfly_saxs`, CLI `bsaxs`)
identifies and parameterizes butterfly-pattern 2D SAXS. Typical experimental
frames support a **first-order ring** (`q*` and ring L = `2π/q*`). Apparent
ellipse `a`, `b/a`, `θ`, and unpublished Ln/Lz/major-axis candidates exist only
when an interior ellipse is actually supported.

`success=True` is an engineering status, not scientific acceptance.

## Environment

Python **3.11–3.13** (3.14+ unsupported).

```bash
python -m venv .venv-project
.venv-project/bin/python -m pip install -U pip
.venv-project/bin/python -m pip install \
  -c constraints/validation-py311-313.txt -e ".[all]"
bsaxs doctor --json
# equivalent: bsaxs-doctor --json
```

If `bsaxs` cannot import, start with `bsaxs-doctor --json` (stdlib-only). Core
analysis does not need Qt; the workbench does (`--require-ui`).

Windows desktop entry: `启动_LamellarSAXS2D.cmd` (optional `--check`).

## Discover commands

```bash
bsaxs describe          # default if you run `bsaxs` with no subcommand
bsaxs --help
```

Stdout is strict JSON (`allow_nan=False`, ASCII-escaped). Human diagnostics
belong on stderr. Failed commands also print a JSON envelope:

```json
{"schema_version": "lamellarsaxs2d.cli_error.v1", "ok": false, "exit_code": 2}
```

### Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Completed; required quality gates passed (or the command has none). |
| 1 | Completed with WARN, quality FAIL, partial batch failure, or cancel; keep evidence. |
| 2 | Usage/input/config/overwrite error. Do not treat stdout as a scientific result. |

## Safe agent workflow

1. `bsaxs doctor --json` — environment ready?
2. `bsaxs describe` — catalog, invariants, recommended flags.
3. Prefer synthetic or user-supplied fixtures. Never write into raw data dirs.
4. `bsaxs inspect INPUT [--poni PONI] [--mask MASK]` — read-only diagnostics.
5. `bsaxs analyze INPUT --ridge-method butterfly_curvature --ellipse-preset flat_ellipse --butterfly-stage evaluate --butterfly-resamples 0`
6. Real packages: `bsaxs preflight PACKAGE --manifest MANIFEST --poni PONI --mask MASK -o results/preflight` before fitting.
7. Series: `bsaxs batch "data/frame_*.edf" --poni PONI --mask MASK --mode independent -o results/batch`

Do not pass `--full2d` unless an empirical whole-pixel intensity model is
explicitly requested. It is not the butterfly geometry measurement.

Do not overwrite outputs without `--force`. `--force` never applies to raw
inputs, PONI, masks, or `data_local/`.

## Invariants you must not violate

- pixel-q is not a physical period. Load a PONI before reporting L in nm.
- Do not fabricate missing butterfly quadrants.
- A `flat_ellipse` bound hit or runaway major axis is **ring-only**; do not copy ring L into Ln.
- Identify (trace) then Evaluate. GUI order is the same.
- geometry-only analysis must not silently start full2d.
- P3/P4 gates are engineering evidence, not scientific acceptance.

The inspect/analyze/batch JSON includes an `agent` object with `next` / `do_not`
hints and `scientific_acceptance: false`. Those hints are operator guidance,
not a publication decision.

## Tests and layout

```bash
python -m ruff check src tests scripts
python -m pytest -q
```

Package code lives in `src/butterfly_saxs/`. CLI seam: `cli.py` (lazy scientific
imports). Qt-free service: `service.py`. Do not put QWidget work in workers.

## Docs

| Topic | Path |
| --- | --- |
| First run / Identify then Evaluate | `docs/first_run_zh.md` |
| CLI, TOML, batch, exports | `docs/user_guide_zh.md` |
| Result schema | `docs/validation/result_schema_v1.md` |
| Architecture | `docs/architecture_zh.md` |
