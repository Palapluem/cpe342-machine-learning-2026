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
│   └── Assignment 5_Neural Network/
│       ├── Assignment_5_Deep_Neural_Network.ipynb # Fully executed reference notebook
│       ├── 3_DNN_Homework.ipynb               # Course template homework notebook
│       ├── 1042_xxxx_xxxx.ipynb               # Course student ID submission notebook
│       ├── bank-data.csv                      # Introductory tutorial dataset (N = 45,211)
│       ├── 1_Intro_to_NN.ipynb                # Introductory Lab: Scikit-learn & Keras MLP on Bank Data
│       ├── 2_Keras_Tutorial.ipynb             # Tutorial: Introduction to Keras & MNIST
│       ├── Keras_Tutorial.ipynb               # Quickstart: TensorFlow Keras Sequential Pipeline
│       ├── CPE342_Assignment 5_main.tex      # Academic LaTeX source report
│       ├── CPE342_Assignment 5_main.pdf      # Publication-grade academic report (33 pages)
│       ├── notebook_appendix_5.tex           # Auto-generated LaTeX appendix with executed cells
│       ├── benchmark_results.json             # Complete empirical benchmark metrics (11 models)
│       └── plot_1_*.pdf to plot_7_*.pdf       # High-resolution publication-grade vector figures
│
├── lecture/                                   # Lecture slides and demo notebooks
│   ├── Lecture 1–12 PDFs                      # Course slide decks
│   ├── ML_2_Training_Models.ipynb             # Demo: Gradient Descent & Linear Regression
│   ├── ML_3_SVM.ipynb                         # Demo: Support Vector Machines
│   ├── ML_4_Regression.ipynb                  # Demo: Regression Analysis
│   ├── ML_5_Classification_Tree_Based.ipynb   # Demo: Tree-based Classification (Decision Trees)
│   ├── ML_5_Ensemble_Models.ipynb             # Demo: Random Forest, Gradient Boosting & XGBoost
│   ├── ML_5_hr_attrition.csv                  # Dataset: HR Employee Attrition
│   ├── ML_5_bank-data.csv                     # Dataset: Banking Marketing Churn
│   └── ML_5_hr_attrition.data                 # Preprocessed training cohort pickle
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
  * Comparative methodological discussion with tabular MLP on Bank Marketing data (`bank-data.csv`).
* **Deliverables:**
  * [`Assignment_5_Deep_Neural_Network.ipynb`](./assignment/Assignment%205_Neural%20Network/Assignment_5_Deep_Neural_Network.ipynb)
  * [`3_DNN_Homework.ipynb`](./assignment/Assignment%205_Neural%20Network/3_DNN_Homework.ipynb)
  * [`1042_xxxx_xxxx.ipynb`](./assignment/Assignment%205_Neural%20Network/1042_xxxx_xxxx.ipynb)
  * [`CPE342_Assignment 5_main.tex`](./assignment/Assignment%205_Neural%20Network/CPE342_Assignment%205_main.tex)
  * [`CPE342_Assignment 5_main.pdf`](./assignment/Assignment%205_Neural%20Network/CPE342_Assignment%205_main.pdf) (33 pages)
  * [`benchmark_results.json`](./assignment/Assignment%205_Neural%20Network/benchmark_results.json)
  * Diagnostic Plots: [`plot_1_mnist_eda.pdf`](./assignment/Assignment%205_Neural%20Network/plot_1_mnist_eda.pdf), [`plot_2_dnn_architecture.pdf`](./assignment/Assignment%205_Neural%20Network/plot_2_dnn_architecture.pdf), [`plot_3_training_history.pdf`](./assignment/Assignment%205_Neural%20Network/plot_3_training_history.pdf), [`plot_4_confusion_matrix.pdf`](./assignment/Assignment%205_Neural%20Network/plot_4_confusion_matrix.pdf), [`plot_5_learning_rate_comparison.pdf`](./assignment/Assignment%205_Neural%20Network/plot_5_learning_rate_comparison.pdf), [`plot_6_optimizer_comparison.pdf`](./assignment/Assignment%205_Neural%20Network/plot_6_optimizer_comparison.pdf), [`plot_7_architecture_ablation.pdf`](./assignment/Assignment%205_Neural%20Network/plot_7_architecture_ablation.pdf)

---

## 🛠️ Environment & Prerequisites

* **Python:** 3.10+
* **Deep Learning Frameworks:** `tensorflow` (2.21+), `keras` (3.15+)
* **Core Machine Learning Libraries:** `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`, `lifelines`, `statsmodels`, `seaborn`, `jupyter`, `nbclient`, `nbformat`
* **Typography:** TH Sarabun New / Sarabun font support for Matplotlib charts
* **Report Compilation:** XeLaTeX / TeX Live / MiKTeX (Polyglossia + Sarabun font)

---

## 👤 Author & Team

* **Wisit Suwannao (วิศิษฐ์ สุวรรณเนาว์)** — *Lead Author*
  * **Student ID:** 67070501042
  * **Department:** Computer Engineering, Faculty of Engineering
  * **Institution:** King Mongkut's University of Technology Thonburi (KMUTT)
* **Team Members:**
  * [Student Name 2] ([Student ID 2])
  * [Student Name 3] ([Student ID 3])
* **Instructor:** ดร. บุญฤทธิ์ จันทร์ไกรวัล (Dr. Boonyarit Changaival)
