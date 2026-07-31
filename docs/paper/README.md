# CDVT Paper Artifacts

Status date: 2026-07-31

Read in this order:

1. `CDVT_Manuscript_Draft.md`
   Contains Introduction, Related Work, Method, completed Phase 2 Results,
   Discussion, Limitations, and a provisional Conclusion.
2. `CDVT_Experiment_Tables.md`
   Contains the paper-facing result tables, reporting rules, experiment
   registry, and remaining `TBD` cells.
3. `../experiments/CDVT_PHASE2_RESULTS.md`
   Gives the concise six-dataset gate decision and authoritative provenance.
4. `../experiments/results/CDVT_Phase2_Six_Dataset_Summary.json`
   Is the machine-readable Phase 2 record generated from final manifests.
5. `CDVT_Citation_Metadata_Audit.md`
   Tracks bibliographic verification and remaining claim-alignment work.
6. `CDVT_Followup_Deployment_Record.md`
   Records the exact follow-up source, portable commit, remote paths, gate,
   and workload.

Architecture assets:

- `cdvt_architecture.pdf`: paper-ready vector figure.
- `cdvt_architecture_render.png`: rendered preview.

The manuscript is not final. Paired three-seed results, missing core
ablations, no-relation and K sensitivity, and normal-only runtime benchmarks
are running from portable commit `f7209f2`. Do not replace a `TBD` cell with a
trajectory or log value; only final manifests and generated summaries are
authoritative.
