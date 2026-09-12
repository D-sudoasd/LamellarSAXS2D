# Review fixes

The reviewed upgrade keeps geometry fits and empirical full-pixel models distinct. These repairs address scientific diagnostics, application interaction and export integrity.

- **R01/R09:** sample an isotropic radial reference on the same q grid and mask before finding angular lobes; preserve observed profiles and export the reference/detection difference separately. Keep the four-sigma rule, use a local angular noise estimator, and accept existing start/stop q-window aliases.
- **R02:** use contrast-aware bound diagnostics and check candidate intensity bounds against the same full weighted objective, preserving ties and fitted values. Evaluate soft-L1 stably for tiny residuals so distinguishable interior coefficients are not mistaken for bounds. Cover large pedestals, tiny units and near-exact zero solutions.
- **R03/R04:** normalize configured method aliases consistently and reject resolved analysis-file/figure-directory collisions before writing, including incompatible existing target types under force mode.
- **R05/R06/R07:** let the closest displayed ridge/peak win a click; expose changing warnings and workflow text to accessibility tools; provide a keyboard-accessible numeric profile table alongside plots.
- **R08/R10:** place shared candidate geometry and figure preparation below the export orchestrator; preserve compatibility imports and remove the identical rendering branches.
- **R11:** reuse only unchanged q-bin geometry within an analysis, with one bounded cache entry. Recompute intensity statistics every draw and bypass/invalidate reuse for changed calibration, centers, masks, finite support or windows.

Validation includes fixed synthetic negative/positive controls, real-data checks without rewriting inputs, offscreen Qt interactions, CLI export checks, figure/CSV/NPZ round trips, independent code review, and the test suite. Native Qt Quick3D checks and unavailable historical fixtures remain separate environment-dependent checks. Software validation does not establish beamline calibration, structural identifiability or scientific acceptance of an ellipse.

Local reports, source-array bundles and development evidence live under ignored `outputs/` directories. They are not included in source delivery.
