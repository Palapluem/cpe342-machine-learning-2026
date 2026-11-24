# ดัชนีตรวจสอบหลักฐานประกอบรายงาน (`project/report/evidence/`)

ข้อมูลตัวเลข กราฟ และผลการทดสอบแบบ Cross-Validation ทั้งหมดใน [report.tex](../report.tex) และ [CPE342_TermProject_Report.pdf](../output/pdf/CPE342_TermProject_Report.pdf) ตรวจสอบย้อนกลับได้จากไฟล์ผลลัพธ์ของแต่ละโมดูลดังนี้

| โมดูล / หัวข้อ | ไฟล์หลักฐานต้นทาง | ภาพประกอบใน `report/assets/` |
| :--- | :--- | :--- |
| ขอบเขตและกติกาโครงงาน | `project/CPE342_Project_Instruction.pdf`, `task1..5/data/datacard.txt` | — |
| Task 1: Anti-Cheat Pre-Filter | `project/task1/task1_anti_cheat.ipynb`, `task1/assets/{ablation_benchmark,per_fold_f2}.csv`, `task1/assets/eda/missing-rate-train-vs-test.csv`, `task1/models/experiments/006/metrics.json` | `task1_feature_distributions.png`, `task1_missing_rate_train_test.png`, `task1_diagnostics.png`, `task1_test_alignment.png` |
| Task 2: Player Segment Classification | `project/task2/task2_player_segment.ipynb`, `task2/assets/{ablation_benchmark,per_fold_macro_f1}.csv`, `task2/models/bayes-export/decision-rule.json` | `task2_diagnostics.png`, `task2_test_alignment.png` |
| Task 3: Monthly Spending Prediction | `project/task3/task3_monthly_spending.ipynb`, `task3/assets/ablation_benchmark.csv`, `task3/models/cv_metrics_summary.json` | `figure1-spending-eda.png`, `figure2-hurdle-calibration-diagnostics.png`, `figure3-residual-error-distribution.png`, `figure4-test-spending-prediction-density.png` |
| Task 4: Game Title Detection | `project/task4/task4_game_title_detection.ipynb`, `task4/assets/{ablation-benchmark,per-class-metrics}.csv`, `task4/assets/eda/{class-distribution,color-summary,image-integrity}.csv`, `task4/models/t4_sota_fusion_meta.json` | `figure1-screenshot-eda.png`, `figure2-validation-diagnostics.png`, `figure3-regime-error-analysis.png`, `figure4-test-class-distribution.png` |
| Task 5: Account Security Monitoring | `project/task5/task5_account_security.ipynb`, `task5/assets/{archetype-rule-counts,unstable-nan-calibration,unstable-nan-cut-expected-f3}.csv` | `task5_anomaly_detection_diagnostics.png`, `erratic-combat-box.png`, `task5_bayes_calibration.png`, `task5_test_alignment.png` |
| การรวมผลและตรวจสอบความถูกต้อง | `project/merge_submission.ipynb`, `project/submission.csv` | — |
