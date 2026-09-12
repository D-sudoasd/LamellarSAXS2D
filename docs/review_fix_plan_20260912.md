# Review remediation and delivery

Scope: repair the reviewed working tree without changing raw experimental data.

1. R01/R09: remove angular pixel-sampling artefacts from peak detection, preserve raw profiles and source pixels, and support the existing q-window aliases. Validate isotropic controls and genuine lobes independently.
2. R02: prevent a large fixed background from expanding a narrow coefficient's bound tolerance. Preserve tiny-unit and genuine-bound cases.
3. R03/R04: validate all resolved CLI output paths before writing; share the existing analysis-method normalization.
4. R05/R06/R07: resolve competing peak/ridge clicks and expose live warning/profile information to assistive technology.
5. R08/R10: move common geometry/figure helpers below the bundle orchestrator and remove the duplicated render branch, preserving output contracts.
6. R11: reuse only unchanged geometric bin assignments within an uncertainty run. Recalculate intensity statistics and invalidate geometry reuse on changed maps, masks, windows or finite support.
7. Run focused regression checks, independent review, real-data/Qt/CLI checks and the full suite. Keep scientific validity separate from software success.
8. Commit the reviewed changes, merge and push to main; verify local/remote agreement and remote CI. Remove only merged task branches and verified disposable build/cache directories. Preserve evidence, data, environments, active worktrees and deliverable figure bundles.

Each implementation surface has one writer; the lead owns integration, CLI, intensity-bound diagnostics, verification and Git delivery.
