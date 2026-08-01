# CDVT Paper Artifacts

Status date: 2026-08-01

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
5. `../experiments/results/CDVT_Phase3_Interim_8of12.md`
   Records only completed same-seed Phase 3 pairs. It intentionally withholds
   aggregate mean and standard deviation values while five manifests are
   missing.
6. `CDVT_Citation_Metadata_Audit.md`
   Tracks bibliographic verification and remaining claim-alignment work.
7. `CDVT_Followup_Deployment_Record.md`
   Records the exact follow-up source, portable commit, remote paths, gate,
   and workload.
8. `CDVT_Code_Method_Alignment_Audit.md`
   Maps every substantive method claim to the frozen implementation and lists
   the remaining experimental evidence gates.
9. `CDVT_Additive_Control_Deployment_Record.md`
   Records the portable additive-fusion control and its hard dependency on a
   successfully completed unified follow-up queue.

Architecture assets:

- `cdvt_architecture.pdf`: paper-ready vector figure.
- `cdvt_architecture_render.png`: rendered preview.

The manuscript is not final. Paired three-seed results, missing core
ablations, no-relation and K sensitivity, and normal-only runtime benchmarks
are running from portable commit `f7209f2`. A three-dataset additive-fusion
control is staged at source commit `dac3c7b` and will start only after the
current unified queue completes. Do not replace a `TBD` cell with a trajectory
or log value; only final manifests and generated summaries are authoritative.
