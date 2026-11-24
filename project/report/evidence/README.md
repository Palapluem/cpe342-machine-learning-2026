# Evidence & Traceability Index (`project/report/`)

All metrics, tables, and 300-dpi figures in `CPE342_TermProject_Report.pdf` are traced directly to executed standalone notebooks and verified artifacts in `project/task1`–`project/task5` and `project/submission.csv`.

## Submission Checksums (`25,889` Test Rows)

| Artifact | Path | `sha12` | Source Version |
|---|---|---|---|
| Consolidated Submission | `project/submission.csv` | `7fceeedc7d15` | Run 30 (`P4a`, merged before Nov 24, 2026 freeze) |
| Task 1 Index | `project/task1/submit/index.csv` | `db458dfc73b4` | Analytical Cohort LR + 5-Profile Semi-Sup Tobit-MVN (`exp006`) |
| Task 2 Index | `project/task2/submit/index.csv` | `9adab5d3de0d` | `0.90 ExactParamEM + 0.10 CB_Clean28` + 4-Param Logit Bias + Affine Rule |
| Task 3 Index | `project/task3/submit/index.csv` | `3646bdd6dda1` | `ExactBayesL1V2` Generative Marginalization |
| Task 4 Index | `project/task4/submit/index.csv` | `8f0b9db4d663` | Run 30 (`P4a`) 4-Family Dual-Regime SOTA Fusion |
| Task 5 Index | `project/task5/submit/index.csv` | `a5005a73a89d` | 8 Exact Archetype Rules + Closed-Form `ExactBayesUB` |

## Figure Assets Included in `project/report/assets/`

| Report Asset | Origin in `project/` | What It Evidences |
|---|---|---|
| `task1_diagnostics.png` | `task1/assets/task1_diagnostics.png` | ROC progression (`0.7678` to `0.8968`), 5/5 per-fold $F_2$ uplift, OOF confusion matrix (`F2=0.84155`), top-15 LGBM features |
| `task1_test_alignment.png` | `task1/assets/task1_test_alignment.png` | Test probability density around $\tau^* = 0.1990$ and Train (`34.93%`) vs. Test (`58.98%`) positive rate |
| `task2_diagnostics.png` | `task2/assets/task2_diagnostics.png` | Top-15 Clean28+ZLR features, model OOF LogLoss, normalized confusion matrix, 5-fold Macro-$F_1$ consistency (`0.75980`) |
| `task2_test_alignment.png` | `task2/assets/task2_test_alignment.png` | Train vs. OOF vs. Test segment shares (`N=25,889`) and calibrated affine decision margins |
| `figure1-spending-eda.png` | `task3/assets/figure1-spending-eda.png` | Zero-inflation (`48.12%`), log10 spender distribution, segment spending boxplots, historical correlation |
| `figure2-hurdle-calibration-diagnostics.png` | `task3/assets/figure2-hurdle-calibration-diagnostics.png` | Stage-1 calibration & ROC (LR, LGBM, `ExactBayesL1V2`), actual vs. OOF log-spending, segment absolute error |
| `figure3-residual-error-distribution.png` | `task3/assets/figure3-residual-error-distribution.png` | OOF residual distribution and 1,000-bootstrap paired NMAE gain vs. Quantile Hurdle V3 |
| `figure4-test-spending-prediction-density.png` | `task3/assets/figure4-test-spending-prediction-density.png` | Test spending density (`49.6%` exact zeros, median `78.76 THB`, max `300,000 THB`) |
| `figure1-screenshot-eda.png` | `task4/assets/figure1-screenshot-eda.png` | Training class balance (`31,546` rows across 5 games) and representative gameplay thumbnails |
| `figure2-validation-diagnostics.png` | `task4/assets/figure2-validation-diagnostics.png` | Clean SGKF confusion matrix (`99.41%` raw / `0.994636` Unif-Prior F1) and per-class Clean vs. `ExtHard6` F1 |
| `figure3-regime-error-analysis.png` | `task4/assets/figure3-regime-error-analysis.png` | Stage 1–5 validation progression and Consensus Shift-Gate divergence routing (`795` test rows) |
| `figure4-test-class-distribution.png` | `task4/assets/figure4-test-class-distribution.png` | Train vs. Final Fusion test class shares across all `25,889` test images |
| `task5_anomaly_detection_diagnostics.png` | `task5/assets/task5_anomaly_detection_diagnostics.png` | Soft-Max Tail break (`18.35`), 9-cohort anomaly counts (`2,077` flags), security flag enrichment, `behavior_stability` gap |
| `erratic-combat-box.png` | `task5/assets/erratic-combat-box.png` | Joint 12D `erratic_combat` uniform box separation (`244` accounts) |
| `task5_bayes_calibration.png` | `task5/assets/task5_bayes_calibration.png` | `ExactBayesUB` Expected-$F_3$ cut optimization (`0.11513`, `+25` recovery) and known-label reliability curve |
| `task5_test_alignment.png` | `task5/assets/task5_test_alignment.png` | Final Test Normal (`23,812`, `91.98%`) vs. Anomaly (`2,077`, `8.02%`) split |

## Recorded Source Conflicts & Unknowns

1. **Task 4 Label Format:** `CPE342_Project_Instruction.pdf` (p. 24) illustrates string slugs (`freefried`, `fiveN`), whereas `task4/data/datacard.txt` and the Kaggle evaluator (`sample_submission.csv`) require integers `0..4`. Our submission uses integers `0..4`.
2. **Task Weights $w_1 \dots w_5$:** The official weighted formula $\sum w_i S_i / \sum w_i$ appears in `CPE342_Project_Instruction.pdf` (p. 26), but exact numeric weights $w_i$ are undisclosed by the competition host.
3. **`project/submit.md` vs. `project/submission.csv`:** `project/submit.md` lists Run 27 (`d5ce58837641`, Public LB `0.86680`), whereas `project/submission.csv` (`7fceeedc7d15`, Public LB `0.86678`) matches Run 30 (`P4a`) from `ml00x/submit.md` (#99). Both Run 27 and Run 30 were auto-selected as Kaggle Final Submissions; Private LB scores are pending release.
4. **Team Member Task Allocation:** Only Wisit Suwannao (`67070501042`) is verified across `AGENTS.md`, `project/README.md`, and git commits (`ml00x` & `cpe342`). Individual task assignments for the remaining four members from `AGENTS.md` (`003`, `027`, `045`, `067`) are marked `Unknown` where unverified.
