# CPE 342 Machine Learning

This repository contains coursework, assignments, lecture materials, and reference textbooks for the **CPE 342: Machine Learning** course (Semester 1/2024), Department of Computer Engineering, Faculty of Engineering, King Mongkut's University of Technology Thonburi (KMUTT).

---

## 📂 Repository Structure

```text
cpe342-machine-learning-2026/
├── assignment/
│   ├── Assignment 1_Training models/
│   │   ├── Assignment_1_OLS.ipynb             # Jupyter Notebook implementation
│   │   ├── CPE342_Assignment 1_Problem.pdf     # Problem specification
│   │   ├── CPE342_Assignment 1_main.tex       # LaTeX source report
│   │   ├── CPE342_Assignment 1_main.pdf       # Compiled report
│   │   └── CPE342_Assignment 1_Full Result.pdf # Complete merged submission
│   │
│   ├── Assignment 2_Training models via iterative approach/
│   │   ├── Assignment_2_GD.ipynb              # Jupyter Notebook implementation
│   │   ├── CPE342_Assignment 2_Data.csv       # Training dataset (n = 100)
│   │   ├── CPE342_Assignment 2_Problem.pdf    # Problem specification
│   │   ├── CPE342_Assignment 2_main.tex      # LaTeX source report
│   │   ├── CPE342_Assignment 2_main.pdf      # Compiled report
│   │   └── CPE342_Assignment 2_Full Result.pdf# Complete merged submission
│   │
│   ├── Assignment 3_Regression/
│   │   ├── Assignment_3_Regression.ipynb      # Jupyter Notebook implementation
│   │   ├── Telco-Churn.csv                    # Dataset (N = 7,043)
│   │   ├── CPE342_Assignment 3_main.tex      # LaTeX source report
│   │   ├── CPE342_Assignment 3_main.pdf      # Compiled report
│   │   ├── notebook_appendix_3.tex           # LaTeX appendix with executed notebook cells
│   │   └── plot_1_*.pdf to plot_5_*.pdf       # High-resolution vector figures
│   │
│   ├── Assignment 4_Tree-based and Ensemble Models/
│   │   ├── Assignment_4_Tree-Based_and_Ensemble_Models.ipynb # Jupyter Notebook implementation
│   │   ├── MBA.csv                            # Dataset (N = 6,194, Labeled = 1,000)
│   │   ├── CPE342_Assignment 4_main.tex      # LaTeX source report
│   │   ├── CPE342_Assignment 4_main.pdf      # Compiled report (31 pages)
│   │   ├── notebook_appendix_4.tex           # LaTeX appendix with executed notebook cells
│   │   └── plot_1_*.pdf to plot_6_*.pdf       # High-resolution vector figures
│   │
│   ├── Assignment 5_Neural Network/
│   │   ├── Assignment_5_Deep_Neural_Network.ipynb # Fully executed reference notebook
│   │   ├── 1027_1042_1067.ipynb               # Official course submission notebook
│   │   └── CPE342_Assignment 5_main.pdf      # Publication-grade academic report
│   │
│   ├── Assignment 6_CNN/
│   │   ├── 67_1027_1042.ipynb                 # Official course submission notebook
│   │   ├── ML_7_CNN_in-class.ipynb            # In-Class Lab: Evaluation, activations & discussions
│   │   ├── test_01.png                        # Sample 28x28 grayscale test digit
│   │   ├── CNN_Homework.ipynb                 # Original assignment template
│   │   ├── CPE342_Assignment 6_main.tex       # LaTeX source report
│   │   ├── CPE342_Assignment 6_main.pdf       # Publication-grade academic report (61 pages)
│   │   ├── notebook_appendix_6.tex            # LaTeX appendix with executed notebook cells
│   │   ├── benchmark_results.csv & .json      # 5-model benchmark and ablation metrics
│   │   └── plot_1_*.pdf to plot_8_*.pdf       # Publication-grade vector figures
│   │
│   ├── Assignment 7_Dimensionality Reduction/
│   │   ├── 67_1003_1027_1042_1045_1067.ipynb # Official course submission notebook
│   │   ├── [Lab] Dimensionality Reduction.ipynb # Original assignment template
│   │   ├── dimensionality-reduction.xlsx       # MTCARS dataset (32 cars, 11 features)
│   │   ├── CPE342_Assignment 7_main.tex       # LaTeX source report
│   │   ├── CPE342_Assignment 7_main.pdf       # Publication-grade academic report (41 pages)
│   │   ├── notebook_appendix_7.tex            # LaTeX appendix with executed notebook cells
│   │   ├── benchmark_results.csv & .json      # PCA spectral benchmarks and PVE metrics
│   │   └── plot_1_*.pdf to plot_7_*.pdf       # Publication-grade vector figures
│   │
│   └── Assignment 8_Clustering_Basics/
│       ├── 67_1003_1027_1042_1045_1067.ipynb # Official course submission notebook
│       ├── [Lab] Clustering Basics.ipynb      # Original assignment template
│       ├── clustering-basics.xlsx             # Course dataset (4 synthetic sheets + Wine)
│       ├── CPE342_Assignment 8_main.tex       # LaTeX source report
│       ├── CPE342_Assignment 8_main.pdf       # Publication-grade academic report (39 pages)
│       ├── notebook_appendix_8.tex            # LaTeX appendix with executed notebook cells
│       ├── benchmark_results.csv & .json      # Clustering metrics across geometries & Wine
│       └── plot_1_*.pdf to plot_8_*.pdf       # Publication-grade vector figures
├── lecture/                                   # Lecture slides and demo notebooks
│   ├── Lecture 1–12 PDFs                      # Course slide decks
│   ├── ML_2_Training_Models.ipynb             # Demo: Gradient Descent & Linear Regression
│   ├── ML_3_SVM.ipynb                         # Demo: Support Vector Machines
│   ├── ML_4_Regression.ipynb                  # Demo: Regression Analysis
│   ├── ML_5_Classification_Tree_Based.ipynb   # Demo: Tree-based Classification (Decision Trees)
│   ├── ML_5_Ensemble_Models.ipynb             # Demo: Random Forest, Gradient Boosting & XGBoost
│   ├── ML_5_hr_attrition.csv                  # Dataset: HR Employee Attrition
│   ├── ML_5_bank-data.csv                     # Dataset: Banking Marketing Churn (Lecture 5)
│   ├── ML_5_hr_attrition.data                 # Preprocessed training cohort pickle
│   ├── ML_6_1_Intro_to_NN.ipynb               # In-Class Lab: Scikit-learn & Keras MLP on Bank Data (Week 6)
│   ├── ML_6_2_Keras_Tutorial.ipynb            # Tutorial: Comprehensive Keras Sequential & MNIST (Week 6)
│   ├── ML_6_Keras_Tutorial_Quickstart.ipynb   # Quickstart: TensorFlow Keras Sequential Pipeline (Week 6)
│   ├── ML_6_bank-data.csv                     # Dataset: Banking Marketing Tabular Data (Lecture 6, N = 45,211)
│   ├── ML_7_CNN_in-class.ipynb                # In-Class Lab: Convolutional Neural Networks on MNIST & Dogs vs Cats (Week 7)
│   ├── test_01.png                            # Test image for digit evaluation (28x28)
│   ├── ML_8_Dimensionality_Reduction.mp4      # Lecture Recording: Dimensionality Reduction & PCA (Week 8)
│   ├── ML_8_dimensionality-reduction.xlsx      # Lecture Dataset: MTCARS Tabular Features (Week 8)
│   ├── ML_8_Tutorial_Dimensionality_Reduction.ipynb # Tutorial: PCA Implementation & Scree Analysis (Week 8)
│   ├── ML_9_Clustering_Basics.mp4             # Compressed Lecture Recording (27.9 MB, Week 9)
│   ├── ML_9_clustering-basics.xlsx            # Lecture Dataset: Clustering Basics (Week 9)
│   └── ML_9_Tutorial_Clustering_Basics.ipynb  # Tutorial: Clustering Basics & K-Means (Week 9)
├── project/                                   # CPE 342 Term Project (Karena Player Intelligence System)
│   ├── CPE342_Project_Instruction.pdf         # Term project guidelines & specification
│   ├── CPE342_Project_Instruction (Ver. 2025).pdf # Full project challenge slide deck (40 pages)
│   └── README.md                              # Term project overview, 5 tasks & system specifications
│
└── textbook/                                  # Reference textbooks (ISLR / O'Reilly)
```

---

## 📝 Coursework & Assignments

### [Assignment 1: Training Models (OLS Regression)](./assignment/Assignment%201_Training%20models/)
* **Topic:** Linear Regression using Ordinary Least Squares (OLS) via Closed-Form Analytic Solutions.
* **Key Tasks:**
  * Proof and derivation of Normal Equations ($\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^T\mathbf{y}$).
  * Solving for regression coefficients ($\alpha, \beta$) via multiple analytical methods (Cramer's Rule, Matrix Inversion, $S_{xx}/S_{xy}$ formulation).
  * Comprehensive regression diagnostics (Residual vs Fitted, Normal Q-Q, Scale-Location, Actual vs Predicted).
* **Deliverables:**
  * [`Assignment_1_OLS.ipynb`](./assignment/Assignment%201_Training%20models/Assignment_1_OLS.ipynb)
  * [`CPE342_Assignment 1_Full Result.pdf`](./assignment/Assignment%201_Training%20models/CPE342_Assignment%201_Full%20Result.pdf)

---

### [Assignment 2: Training Models via Iterative Approach (Gradient Descent)](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/)
* **Topic:** Non-linear Model Fitting using Batch Gradient Descent (GD).
* **Model Equation:** $\hat{y} = C_0 + C_1 e^{C_2 x}$
* **Key Tasks:**
  * Formulation of Mean Squared Error (MSE) loss function: $\mathcal{L}(C_0, C_1, C_2) = \frac{1}{2n}\sum (y_i - \hat{y}_i)^2$.
  * Mathematical derivation of partial derivatives/gradients ($\frac{\partial\mathcal{L}}{\partial C_0}, \frac{\partial\mathcal{L}}{\partial C_1}, \frac{\partial\mathcal{L}}{\partial C_2}$) using the Chain Rule.
  * Definition of simultaneous parameter update rules with learning rate $\eta$.
  * Python implementation from scratch using NumPy with early stopping.
  * Training diagnostic visualizations (Loss learning curve, parameter trajectories, fitted non-linear curve, residual distribution).
  * Mathematical evaluation via Coefficient of Determination ($R^2$).
* **Deliverables:**
  * [`Assignment_2_GD.ipynb`](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/Assignment_2_GD.ipynb)
  * [`CPE342_Assignment 2_Full Result.pdf`](./assignment/Assignment%202_Training%20models%20via%20iterative%20approach/CPE342_Assignment%202_Full%20Result.pdf)

---

### [Assignment 3: Regression — Survival Analysis (Telco Customer Churn)](./assignment/Assignment%203_Regression/)
* **Topic:** Time-to-Event Modeling and Non-Parametric Survival Analysis on Customer Churn Data ($N = 7,043$).
* **Key Tasks:**
  * Formulation of duration $T = \text{tenure}$ (months) and binary event indicator $E = \mathbb{I}(\text{Churn} == \text{'Yes'})$ with Right-Censoring handling.
  * Mathematical definitions and derivation of Survival Function $S(t)$, Hazard Function $\lambda(t)$, and Cumulative Hazard $\Lambda(t) = -\ln S(t)$ based on Lecture 4.
  * Non-parametric Kaplan-Meier estimation $\hat{S}(t) = \prod (1 - d_i/n_i)$ with Greenwood's 95% confidence intervals.
  * Comparative survival segmentation across payment methods (`Electronic check`, `Mailed check`, `Bank transfer (auto)`, `Credit card (auto)`).
  * Cumulative Hazard analysis via Nelson-Aalen estimator.
  * Hypothesis testing via Multivariate Log-Rank Test ($\chi^2 = 865.24, p = 3.07 \times 10^{-187}$) and Pairwise Log-Rank tests.
  * Milestone retention probabilities calculation ($t = 12, 24, 36, 48, 60, 72$ months) and business churn mitigation strategies.
* **Deliverables:**
  * [`Assignment_3_Regression.ipynb`](./assignment/Assignment%203_Regression/Assignment_3_Regression.ipynb)
  * [`CPE342_Assignment 3_main.pdf`](./assignment/Assignment%203_Regression/CPE342_Assignment%203_main.pdf)

---

### [Assignment 4: Tree-Based and Ensemble Models (MBA Admission Prediction)](./assignment/Assignment%204_Tree-based%20and%20Ensemble%20Models/)
* **Topic:** Tree-based supervised classification, Bagging, Random Subspace, and Gradient Boosting on MBA Admissions Data ($N = 6,194$, Labeled Cohort $N = 1,000$).
* **Key Tasks:**
  * Mathematical formulation of Entropy $H(S)$, Information Gain $IG(S, A)$, Gini Impurity, and Tree Pruning (CCP).
  * Variance reduction theory of Bagging ($\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$) and Random Subspace feature sampling in Random Forest.
  * Gradient Boosting formulation via additive pseudo-residuals $r_{im} = -\left[\frac{\partial L}{\partial F(x_i)}\right]$.
  * Implementation and benchmark of 3 models: **Decision Tree**, **Random Forest**, and **Gradient Boosting**.
  * Handling 9:1 class imbalance (`Admit: 90%`, `Waitlist: 10%`) via cost-sensitive weighting (`class_weight='balanced'`) and stratified sampling.
  * Comparative evaluation across Confusion Matrices, Precision, Recall, F1-Score, ROC Curves, and 5-Fold Stratified Cross-Validation.
  * Feature importance analysis identifying GMAT, GPA, Work Experience, and Industry as primary decision determinants ($>80\%$ importance).
  * Hyperparameter validation curves for tree depth (`max_depth`) and ensemble scaling (`n_estimators`).
* **Deliverables:**
  * [`Assignment_4_Tree-Based_and_Ensemble_Models.ipynb`](./assignment/Assignment%204_Tree-based%20and%20Ensemble%20Models/Assignment_4_Tree-Based_and_Ensemble_Models.ipynb)
  * [`CPE342_Assignment 4_main.pdf`](./assignment/Assignment%204_Tree-based%20and%20Ensemble%20Models/CPE342_Assignment%204_main.pdf)

---

### [Assignment 5: Deep Neural Networks (DNN) — MNIST Handwritten Digit Recognition](./assignment/Assignment%205_Neural%20Network/)
* **Topic:** Fully-Connected Multi-Layer Perceptrons (MLP), Forward Propagation, Backpropagation, Categorical Cross-Entropy, Optimizer Dynamics, and Regularization on the MNIST Dataset ($N = 70,000$).
* **Key Tasks:**
  * Mathematical formulation of Artificial Neurons, Perceptrons, Universal Approximation Theorem, and Multi-Layer Feedforward Networks.
  * Non-linear Activation Functions (Sigmoid, Tanh, ReLU, Softmax) and gradient preservation mechanics.
  * Rigorous derivation of Backpropagation via the Multivariable Chain Rule for Softmax with Categorical Cross-Entropy Loss.
  * Mathematical mechanics of Optimization Algorithms: First-order SGD, Momentum, Adagrad ($G_t$ accumulation), RMSprop ($s_t$ decaying average), and Adam ($m_t, v_t$ bias-corrected moments).
  * Data preprocessing pipeline: $28 \times 28 \to 784$ flattening, $[0, 255] \to [0.0, 1.0]$ normalization, one-hot categorical encoding, and 90/10 train-validation splitting.
  * Baseline 2-layer DNN architecture ($784 \to 512 \text{ ReLU} \to 10 \text{ Softmax}$) with 407,050 trainable parameters.
  * **Question 1 (Overfitting Diagnostics)**: Analysis of training vs. validation loss/accuracy across 10 epochs, explaining why baseline SGD does not overfit (active convergence regime).
  * **Question 2 (Confusion Matrix & Error Topology)**: Identification of top misclassified pairs (4 vs 9, 5 vs 3, 2 vs 8, 7 vs 9/2) based on stroke geometry; calculation of Macro Precision (92.36%), Macro Recall (92.31%), and Accuracy (92.41%).
  * **Question 3 (Model Tuning Suite)**:
    * Learning Rate Sensitivity ($\alpha \in \{0.005, 0.01, 0.05, 0.1, 0.2, 0.5\}$).
    * Optimizer Benchmark (SGD $\alpha=0.1$, Adagrad $\alpha=0.01$, RMSprop $\alpha=0.001$, Adam $\alpha=0.001$).
    * Architectural Depth & Dropout Ablation (Shallow vs. Deeper 512-256-128 vs. Deeper + Dropout 0.2, achieving **98.35% Test Accuracy** and **0.0662 Test Loss**).
  * Comparative methodological discussion with tabular MLP on Bank Marketing data (`lecture/ML_6_bank-data.csv`).
* **Deliverables:**
  * [`Assignment_5_Deep_Neural_Network.ipynb`](./assignment/Assignment%205_Neural%20Network/Assignment_5_Deep_Neural_Network.ipynb) (Course Submission File: [`1027_1042_1067.ipynb`](./assignment/Assignment%205_Neural%20Network/1027_1042_1067.ipynb))
  * [`CPE342_Assignment 5_main.pdf`](./assignment/Assignment%205_Neural%20Network/CPE342_Assignment%205_main.pdf)

---

### [Assignment 6: Convolutional Neural Networks (CNN)](./assignment/Assignment%206_CNN/)
* **Topic:** Deep Convolutional Neural Networks (CNN), Transfer Learning (MobileNetV2), Two-Phase Fine-Tuning, Data Augmentation, Feature Map Activations, and offline inference diagnostics for Dog vs. Cat Classification ($N = 25,000$).
* **Key Tasks:**
  * Mathematical formulation of 2D Cross-Correlation, Receptive Fields (Hubel & Wiesel visual cortex), Zero-Padding (Valid vs Same), Stride, and Subsampling / Max Pooling.
  * Architecture of MobileNetV2: Inverted Residual blocks, Depthwise Separable Convolutions, and Linear Bottlenecks (Sandler et al. 2018; Géron Ch. 14).
  * Canonical data protocol: 18,000 augmented training images, 4,500 unaugmented validation images, and 2,500 untouched final-test images; $160\times160$ RGB, `1./255`, Cat=0, Dog=1, seed 42.
  * Baseline Custom Scratch CNN (4 blocks, 6,944,449 total parameters) demonstrating sample inefficiency and overfitting (Final-Test Accuracy: 84.15%, Dog Precision: 83.50%, below the required 90–95% threshold).
  * Two-Phase Transfer Learning Strategy:
    * **Phase 1 (Feature Extraction):** Pretrained MobileNetV2 base frozen, training Global Average Pooling (GAP) + Dense(256) head with Adam ($\eta = 10^{-4}$), EarlyStopping, and ReduceLROnPlateau.
    * **Phase 2 (Fine-Tuning):** Unfreezing the top 40 MobileNetV2 layers with a low learning rate ($\eta = 10^{-5}$).
  * Final grading result on the untouched 2,500-image test split: **97.47% Accuracy**, **97.55% Dog Precision**, **97.38% Cat Precision**, and **0.9747 Macro F1**.
  * Validation-only diagnostics on 4,500 unaugmented images: confusion matrices, ROC/PR curves, model-derived activation maps, and actual TP/TN/FP/FN galleries.
  * 5-model ablation: M1 Scratch CNN, M2 Frozen MobileNetV2, M3 Top-40 + Augmentation, M4 Top-40 without Augmentation, and M5 Top-20 + Augmentation. INT8 is excluded because no evaluated quantized artifact is available.
  * Offline inference evidence: measured on Apple Silicon arm64 with N=170 runs (Mean: 18.4 ms, Median: 18.2 ms, P95: 20.8 ms). Real-time streaming pipeline is an architectural simulation (25.0 ms total latency, 40.0 FPS, 25.0% headroom vs. 33.33 ms deadline); no live-webcam hardware deployment or guarantee of zero dropped frames is claimed.
* **Deliverables:**
  * [`67_1027_1042.ipynb`](./assignment/Assignment%206_CNN/67_1027_1042.ipynb) (Official Course Submission File)
  * [`CPE342_Assignment 6_main.pdf`](./assignment/Assignment%206_CNN/CPE342_Assignment%206_main.pdf) (Publication-Grade Academic Report)
  * [`benchmark_results.csv`](./assignment/Assignment%206_CNN/benchmark_results.csv) & [`benchmark_results.json`](./assignment/Assignment%206_CNN/benchmark_results.json)

---

### [Assignment 7: Dimensionality Reduction (PCA)](./assignment/Assignment%207_Dimensionality%20Reduction/)
* **Topic:** Unsupervised Dimensionality Reduction via Principal Component Analysis (PCA), Spectral Theorem, Sample Covariance vs. Correlation Decomposition, Variance Scale Domination, and 2D Biplot Geometric Interpretation.
* **Key Tasks:**
  * Mathematical formulation of Data Centering ($\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu}$), Sample Covariance Matrix ($\boldsymbol{\Sigma} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$), Eigenvalue Decomposition ($\boldsymbol{\Sigma}\mathbf{p}_j = \lambda_j \mathbf{p}_j$), Orthogonal Subspace Projection ($\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{P}_r$), and Lossless/Truncated Reconstruction ($\hat{\mathbf{X}}_r = \mathbf{Z}\mathbf{P}_r^T + \boldsymbol{\mu}$).
  * **Part I (2D Synthetic Toy Data, $N=12$):** Exact numerical derivation and verification of eigenpairs ($\lambda_1 = 411.6218$, $\lambda_2 = 6.1812$), variance preservation ($\text{PVE}_1 = 98.52\%$), orthogonal 1D subspace projection, and machine-precision lossless reconstruction ($\|\mathbf{X} - \hat{\mathbf{X}}\|_2 = 1.21 \times 10^{-14}$).
  * **Part II (Motor Trend Car Road Tests - MTCARS, $N=32$, $p=11$):**
    * Rigorous comparative analysis between **Unstandardized PCA** (Covariance Matrix $\boldsymbol{\Sigma}$) and **Standardized PCA** (Correlation Matrix $\mathbf{R}$).
    * Mathematical proof of **Scale Domination**: In Unstandardized PCA, `disp` ($\sigma^2 = 15,360.80$, 75.15%) and `hp` ($\sigma^2 = 4,700.87$, 23.00%) account for **98.15%** of the total system variance ($20,440.40$), causing PC1 alone to explain **92.70%** of variance dominated solely by `disp` (loading $-0.900$) while masking the remaining 9 features ($|w| \le 0.038$).
    * Standardized PCA equilibrium: Scaling features to unit variance ($\sigma^2 = 1.0$) eliminates scale distortion; requires **4 PCs** to achieve $\ge 90\%$ total variance ($92.32\%$), balancing power/displacement factors against fuel efficiency/drivetrain factors.
  * **Section 5 (In-depth Q&A):**
    * **Question 2.1 (Unstandardized PCA):** Mathematical proof that 1 PC explains $92.70\% \ge 90\%$ total variance, dominated by `disp` and `hp`.
    * **Question 2.2 (Standardized PCA & Contrast):** Formal proof that 4 PCs are required for $92.32\% \ge 90\%$, with balanced loadings ($|w| \in [0.20, 0.37]$), and root-cause explanation of why standardization is mandatory for multi-scale engineering data.
    * Both questions formatted strictly to **exactly 1 full page each** without spillover.
  * **Geometric Interpretation via 2D PCA Biplot:** Clear projection of 32 car models onto the PC1-PC2 subspace with 11 loading vectors, revealing 3 semantic clusters: *Muscle & Luxury Heavyweights*, *Economy Compacts*, and *Agile Manual Sports Cars*.
  * **Reconstruction Error Validation:** Empirical verification that relative reconstruction error $\frac{\|\mathbf{X} - \hat{\mathbf{X}}_r\|_F}{\|\mathbf{X}\|_F} \times 100\%$ monotonically decreases from $r=1$ to $0.00\%$ at $r=11$, confirming the Spectral Theorem.
* **Deliverables:**
  * [`67_1003_1027_1042_1045_1067.ipynb`](./assignment/Assignment%207_Dimensionality%20Reduction/67_1003_1027_1042_1045_1067.ipynb) (Official Course Submission File)
  * [`[Lab] Dimensionality Reduction.ipynb`](./assignment/Assignment%207_Dimensionality%20Reduction/[Lab]%20Dimensionality%20Reduction.ipynb) (Original Lab Notebook)
  * [`dimensionality-reduction.xlsx`](./assignment/Assignment%207_Dimensionality%20Reduction/dimensionality-reduction.xlsx) (MTCARS Dataset)
  * [`CPE342_Assignment 7_main.pdf`](./assignment/Assignment%207_Dimensionality%20Reduction/CPE342_Assignment%207_main.pdf) (Publication-Grade Academic Report, 41 pages)
  * [`benchmark_results.csv`](./assignment/Assignment%207_Dimensionality%20Reduction/benchmark_results.csv) & [`benchmark_results.json`](./assignment/Assignment%207_Dimensionality%20Reduction/benchmark_results.json)
  * [`plot_1_linear_pca_geometry.pdf`](./assignment/Assignment%207_Dimensionality%20Reduction/plot_1_linear_pca_geometry.pdf) to [`plot_7_reconstruction_error.pdf`](./assignment/Assignment%207_Dimensionality%20Reduction/plot_7_reconstruction_error.pdf) (7 High-Resolution Vector Figures)

---

### [Assignment 8: Clustering Basics & Latent Subspace Pipeline](./assignment/Assignment%208_Clustering_Basics/)
* **Topic:** Unsupervised Partitioning via K-Means Clustering, Voronoi Tessellation Geometry, Lloyd's Heuristic Algorithm, Failure Modes on Non-Convex Geometries, Cluster Tendency (Hopkins Statistic), and High-Dimensional Latent Subspace Clustering on Chemical Wine Profiling.
* **Key Tasks:**
  * Mathematical formulation of K-Means optimization objective (Inertia/WCSS), Voronoi cell boundaries, and step-by-step Lloyd's convergence.
  * **Part I (4 Synthetic Geometries, $N=600\text{--}800$):**
    * **Dataset 1 (Gaussian Blobs with Extreme Outliers):** Quantifying outlier leverage effect on centroid drift ($K=3$ vs $K=4$). Proving that allocating a dedicated outlier cluster isolates anomalies at $(9.97, 4.07)$, restoring cluster cohesion ($s=0.6468$, $\text{CH}=1545.9$).
    * **Dataset 2 (Concentric Rings) & Dataset 3 (Two Interlocking Moons):** Geometric proof that Voronoi tessellation is constrained to convex polyhedra bounded by linear hyperplanes, causing transverse bisection across manifold structures; contrasting with density-based (DBSCAN) and graph spectral clustering paradigms.
    * **Dataset 4 (Anisotropic Blobs):** Analyzing cluster boundary distortion caused by unequal variance and density disparities under the isotropic Euclidean distance metric.
    * Multi-metric comparative benchmarking: Silhouette Width, Calinski-Harabasz Index, Davies-Bouldin Index, and Hopkins Statistic ($H$).
  * **Part II (Italian Wine Chemical Profiling, $N=178$, $p=13$):**
    * Mathematical justification for Z-Score Standardization (variance disparity $\sigma^2=99,166.7$ vs $0.0155$) and Correlation Matrix PCA.
    * Information preservation proof: $r=5$ Principal Components capture **$80.16\% \ge 80.00\%$** total system variance.
    * K-Means on 5D Latent PC Subspace: Global optimal $K=3$ verified via Elbow inflection and Silhouette peak ($s=0.3691$, $\text{CH}=109.2$).
    * Enological chemical profiling: Perfect semantic alignment with Piedmont Italian cultivars — *Barolo* ($n=62$), *Grignolino* ($n=65$), and *Barbera* ($n=51$), mapped onto 2D PCA Biplot with 13 chemical loading vectors and Polar Radar Charts.
  * **Section 5 (In-depth Q&A):**
    * **Question 1 (Outlier Diagnostics & Centroid Drift):** Analytical proof of outlier leverage and $K=4$ partition mechanics.
    * **Question 2 (Non-Convex Manifolds & Voronoi Hyperplane Failure):** Formal geometry of linear Voronoi boundaries and alternative paradigms.
    * **Question 3 (Standardization Mandate & PCA Latent Subspace):** Proof that 5 PCs achieve $80.16\% \ge 80\%$ and why PC clustering eliminates multicollinearity.
    * **Question 4 (Enological Interpretation & Italian Cultivar Alignment):** Chemical characterization across Barolo, Grignolino, and Barbera.
    * All 4 questions formatted strictly to **exactly 1 full page each** without spillover.
* **Deliverables:**
  * [`67_1003_1027_1042_1045_1067.ipynb`](./assignment/Assignment%208_Clustering_Basics/67_1003_1027_1042_1045_1067.ipynb) (Official Course Submission File)
  * [`[Lab] Clustering Basics.ipynb`](./assignment/Assignment%208_Clustering_Basics/[Lab]%20Clustering%20Basics.ipynb) (Original Lab Notebook)
  * [`clustering-basics.xlsx`](./assignment/Assignment%208_Clustering_Basics/clustering-basics.xlsx) (Course Dataset)
  * [`CPE342_Assignment 8_main.pdf`](./assignment/Assignment%208_Clustering_Basics/CPE342_Assignment%208_main.pdf) (Publication-Grade Academic Report, 39 pages)
  * [`benchmark_results.csv`](./assignment/Assignment%208_Clustering_Basics/benchmark_results.csv) & [`benchmark_results.json`](./assignment/Assignment%208_Clustering_Basics/benchmark_results.json)
  * [`plot_1_dataset1_kmeans_elbow_silhouette.pdf`](./assignment/Assignment%208_Clustering_Basics/plot_1_dataset1_kmeans_elbow_silhouette.pdf) to [`plot_8_wine_cluster_biplot_radar_profiling.pdf`](./assignment/Assignment%208_Clustering_Basics/plot_8_wine_cluster_biplot_radar_profiling.pdf) (8 High-Resolution Vector Figures)

---

## 🏆 [Term Project: Karena Player Intelligence System](./project/)

* **Industrial Setting:** Karena Thailand online gaming platform (publisher of *PUBE Mobile*, *Free Fried*, *7-11 Knights*, *Roblock*, *FiveN*).
* **System Mission:** Build an integrated end-to-end Machine Learning pipeline across 5 core operational tasks that feed into a central Player Lifetime Value (LTV) Prediction System.
* **The 5 Core Machine Learning Tasks:**
  1. **Task 1: Anti-Cheat Pre-Filter** (Binary Classification, Target: `is_cheater`, Metric: **$F_2$ Score** prioritizing recall, 32 tabular features).
  2. **Task 2: Player Segment Classification** (Multi-class Classification, Target: `segment` across Casual, Grinder, Social, Whale, Metric: **$F_1$ Score**, 44 tabular features).
  3. **Task 3: Player Monthly Spending Prediction** (Zero-Inflated Continuous Regression, Target: `spending_30d` in THB, Metric: **Normalized MAE**).
  4. **Task 4: Game Title Detection** (Deep Multi-Class Image Classification, Target: 5 game titles from $230\times 120$ centered-crop gameplay screenshots, Metric: **Macro $F_1$**, Pretrained CNN/ViT/Swin permitted).
  5. **Task 5: Account Security Monitoring** (Unsupervised Anomaly Detection, Target: `is_anomaly` across 4 time periods, Metric: **$F_3$ Score** heavily penalizing missed account takeovers/bot rings).
* **Grading & Competition Structure:**
  * **Kaggle Leaderboard:** 70% (Ground Baseline = $0.68$, Challenge Baseline = $0.75$, evaluated on consolidated 25,889-row `sample_submission.csv`).
  * **Technical Academic Report:** 20% (Methodology, EDA, Error Analysis, Business Impact).
  * **Source Code & Reproduction Quality:** 10% (LEB2 & GitHub repository).
* **Deliverables & Documentation:**
  * [`project/README.md`](./project/README.md) (Complete technical specifications & metric derivations)
  * [`CPE342_Project_Instruction (Ver. 2025).pdf`](./project/CPE342_Project_Instruction%20(Ver.%202025).pdf) (Official 40-page project challenge slide deck)
  * [`CPE342_Project_Instruction.pdf`](./project/CPE342_Project_Instruction.pdf) (Project guidelines & specification document)

---

## 🛠️ Environment & Prerequisites

* **Python:** 3.10+
* **Deep Learning Frameworks:** `tensorflow` (2.21+), `keras` (3.15+), `torch`, `torchvision`, `timm`
* **Core Machine Learning & GBDT:** `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `scipy`, `lifelines`, `statsmodels`, `optuna`
* **Visualization & Data Exploration:** `matplotlib`, `seaborn`, `jupyter`, `nbclient`, `nbformat`
* **Typography:** TH Sarabun New / Sarabun font support for Matplotlib charts
* **Report Compilation:** XeLaTeX / TeX Live / MiKTeX (Polyglossia + Sarabun font)

---

## 👤 Author

* **Wisit Suwannao (วิศิษฐ์ สุวรรณเนาว์)** — 67070501042
  * Department of Computer Engineering, Faculty of Engineering
  * King Mongkut's University of Technology Thonburi (KMUTT)

---

## 📄 License

Copyright © 2026 Wisit Suwannao. All rights reserved.
