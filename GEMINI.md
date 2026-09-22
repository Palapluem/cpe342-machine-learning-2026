# CPE 342 Machine Learning — Workspace Rules & Operational Guidelines

This document serves as the authoritative, comprehensive Workspace Rules and System Prompt for the **CPE 342: Machine Learning** repository at King Mongkut's University of Technology Thonburi (KMUTT). All contributors and AI coding assistants must strictly adhere to these rules.

---

## 1. Repository Identity & Authorship Policy

### 1.1 Repository Ownership & `README.md`
- **Sole Repository Owner:**
  - **Wisit Suwannao (วิศิษฐ์ สุวรรณเนาว์)** — Student ID: `67070501042`
  - Department of Computer Engineering, Faculty of Engineering, KMUTT
- **Rule for `README.md`:**
  - In the `## 👤 Author` section of `README.md`, list **ONLY** Wisit Suwannao (67070501042). Do not include group partners here, as this is the personal coursework repository of the repository owner.

### 1.2 Assignment Reports & Notebook Submissions
- **Authorship by Assignment:**
  - **Assignments 1, 2, 3, 4:** 2 Group Members
    - `67070501027` นัธทวัฒน์ ปริมสิริคุณาวุฒิ (Natthawat Primsirikunawut)
    - `67070501042` วิศิษฐ์ สุวรรณเนาว์ (Wisit Suwannao)
  - **Assignment 5 (Deep Neural Networks):** 3 Group Members
    - `67070501027` นัธทวัฒน์ ปริมสิริคุณาวุฒิ (Natthawat Primsirikunawut)
    - `67070501042` วิศิษฐ์ สุวรรณเนาว์ (Wisit Suwannao)
    - `67070501067` พลวริษฐ์ วัฒนเหมรัตน์ (Polwarit Watthanahemmarat)
  - **Assignment 6 (Convolutional Neural Networks):** 2 Group Members
    - `67070501027` นัธทวัฒน์ ปริมสิริคุณาวุฒิ (Natthawat Primsirikunawut)
    - `67070501042` วิศิษฐ์ สุวรรณเนาว์ (Wisit Suwannao)
  - **Assignment 7 (Dimensionality Reduction):** 2 Group Members
    - `67070501027` นัธทวัฒน์ ปริมสิริคุณาวุฒิ (Natthawat Primsirikunawut)
    - `67070501042` วิศิษฐ์ สุวรรณเนาว์ (Wisit Suwannao)
- **Official Submission Notebook File Naming:**
  - Course convention: `(First 2 digits of ID)_(Last 4 digits of Member 1)_(Last 4 digits of Member 2)[_Member 3].ipynb`
  - Examples:
    - Assignment 5: `1027_1042_1067.ipynb`
    - Assignment 6: `67_1027_1042.ipynb`
    - Assignment 7: `67_1027_1042.ipynb`

---

## 2. Directory Hygiene & File Management Rules

### 2.1 The Zero-Script Rule for Python (`*.py`)
- **STRICT PROHIBITION:** Never save any `.py` script files directly in the repository root or inside any assignment directory (`assignment/Assignment *`).
- **Allowed Location for Scripts:** All helper scripts, migration tools, figure verification scripts, or LaTeX processors MUST be created and executed exclusively inside the assistant's temporary scratch directory:
  `<appDataDir>/brain/<conversation-id>/scratch/`
- **Pre-Commit Verification:** Verify zero `.py` files exist in the repository tree before staging (`git status`, `find . -name "*.py"` or `Get-ChildItem -Recurse -Filter *.py`). If any `.py` files exist in the repository tree, delete them immediately.
- **Strict Dataset / Artifact Cleanliness:** Never commit raw dataset folders (e.g. `Cat_Dog_data/`, `__MACOSX/`), uncompressed batches, or intermediate archive zip files to git.

### 2.2 Git Commit & Synchronization
- **Always Keep Up-to-Date:** Every set of edits must be tested, compiled, staged, committed, and pushed to `origin/main`:
  `https://github.com/Palapluem/cpe342-machine-learning-2026.git`
- **Semantic Commit Messages:** Use standard conventional commit format:
  - `feat(assignment-X): ...` for new features or sections
  - `fix(assignment-X): ...` for bug fixes, alignment corrections, or layout adjustments
  - `style(assignment-X): ...` for formatting, typography, or spacing improvements
  - `refactor(assignment-X): ...` for restructuring or code optimization
  - `docs(assignment-X): ...` for documentation updates

---

## 3. LaTeX Academic Report Standards

### 3.1 Compilation Engine & Typography Setup
- **Compiler:** `xelatex` (run 2 passes to resolve all internal page references, counters, and citations).
- **Language & Line Breaking:**
  ```latex
  \usepackage[no-math]{fontspec}
  \usepackage{polyglossia}
  \setdefaultlanguage{thai}
  \setotherlanguage{english}
  \XeTeXlinebreaklocale "th"
  \XeTeXlinebreakskip = 0pt plus 1pt
  \sloppy
  ```
- **Fonts:**
  - **Body Text:** `Sarabun` with explicit font mapping:
    ```latex
    \setmainfont{Sarabun}[
        BoldFont       = Sarabun-Bold,
        ItalicFont     = Sarabun-Italic,
        BoldItalicFont = Sarabun-BoldItalic,
    ]
    ```
  - **Monospace / Code Listings:**
    ```latex
    \setmonofont{Consolas}[Scale=0.92]
    \newfontfamily\thaifonttt{Consolas}[Scale=0.92]
    ```
- **Code Variables in Text:** Never use raw `\texttt{...}` in descriptive text. Always use:
  ```latex
  \definecolor{codeaccent}{HTML}{0969DA}
  \newcommand{\codevar}[1]{\textcolor{codeaccent}{\textbf{#1}}}
  ```
- **Listings Configuration:**
  ```latex
  \lstdefinestyle{pycode}{
      language=Python,
      backgroundcolor=\color{codebg},
      basicstyle=\ttfamily\small,
      keywordstyle=\color{keyword}\bfseries,
      stringstyle=\color{string},
      commentstyle=\color{comment}\itshape,
      numberstyle=\tiny\color{comment},
      numbers=left,
      numbersep=8pt,
      frame=single,
      rulecolor=\color{codeframe},
      breaklines=true,
      breakatwhitespace=false,
      showstringspaces=false,
      tabsize=4,
      xleftmargin=16pt,
      xrightmargin=4pt,
      columns=flexible,
      keepspaces=true
  }

  \lstdefinestyle{output}{
      basicstyle=\ttfamily\small\color{output},
      backgroundcolor=\color{codebg},
      frame=single,
      rulecolor=\color{codeframe},
      numbers=none,
      xleftmargin=8pt,
      xrightmargin=4pt,
      breaklines=true,
      breakatwhitespace=false,
      columns=flexible,
      keepspaces=true
  }
  ```

### 3.2 Page Budgeting & Section Flow Rules
1. **Section 5 (Q&A): Exactly 1 Page Per Question**
   - Each individual question (e.g., Question 1, Question 2, Question 3, Question 4) must strictly occupy **exactly one full page**.
   - No question may spill over onto a second page. Adjust font sizing, item margins (`itemsep`), and table padding if needed to enforce this.
2. **Seamless Section Transitions (Avoid Empty Pages):**
   - Do not insert blind `\clearpage` before small subsections that leave $>50\%$ of a page blank.
   - Let subsections and figures flow naturally. E.g., Section 4 follows Section 3.3 seamlessly; Section 7 concludes on the same page as Section 6.4.
3. **Appendix Formatting (Assignment 5 Gold Standard):**
   - **Main Report Entry:**
     ```latex
     \clearpage
     \appendix
     \raggedbottom
     \section*{Appendix: Full Jupyter Notebook Code \& Output}

     รายละเอียดต่อไปนี้คือโค้ด Python ที่ใช้แก้ปัญหาและวิเคราะห์แบบจำลอง [Model Domain] บนชุดข้อมูล [Dataset Name] พร้อมผลลัพธ์และกราฟทั้งหมด ซึ่งได้รันจริงจาก Jupyter Notebook (\texttt{<CourseNotebook>.ipynb} หรือ \texttt{<SubmissionNotebook>.ipynb})

     \input{notebook_appendix_X.tex}
     ```
   - **Appendix File Structure (`notebook_appendix_X.tex`):**
     - Starts with the Signature Blue Box:
       ```latex
       \begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,halign=left,title=\textbf{Jupyter Notebook: <CourseNotebook>.ipynb\\ (ไฟล์ส่งงานหลัก: <SubmissionNotebook>.ipynb)}]
       โค้ด ผลลัพธ์ และการพล็อตภาพทั้งหมดด้านล่างเป็นการรันจริงจาก Jupyter Notebook
       \end{tcolorbox}
       ```
     - **No Cell 0 Markdown Card:** Jump immediately after the blue box into `[In 1]`.
     - **Figure Sizing & Captions in Appendix:** Standardize all figures to `width=0.96\linewidth`. Always precede or accompany each plot output with its corresponding markdown explanation card (using `tcolorbox` with `halign=left, before upper={\sloppy\raggedright}`).
     - **Strict Byte-for-Byte Accuracy (Zero-Tolerance):** Every code `[In]` listing and plaintext `[Out]` listing in the LaTeX appendix must match the notebook cells byte-for-byte, including trailing newlines and exact strings.
     - **Unconditional Figure Generation:** In the notebook, all figures must be generated unconditionally using `plt.savefig(...)` without any `if not Path(...).exists()` caching bypasses or synthetic fallbacks.
     - **Overfull / Underfull Zero-Warning Rule:** Use `\sloppy` and `\raggedright` inside boxes containing code tokens or long URLs to guarantee zero XeLaTeX overfull `\hbox` warnings.

---

## 4. Jupyter Notebook Standards

### 4.1 Cell Structure & Quality
- **Cell 0:** Clean markdown header containing Course Title, Assignment Name, Authors with Student IDs, and Instructor (`Dr. Boonyarit Changaival`).
- **No Trailing Blank Lines:** Every code cell and markdown cell must be trimmed so there is zero trailing empty line at the bottom.
- **Strict Reproducibility:**
  ```python
  import random, numpy as np, tensorflow as tf
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  ```
- **Sequential Execution Counts:** Execution counts must strictly increment $1, 2, 3, \dots, N$ without gaps or missing outputs.
- **Embedded Visualizations:** All generated plots must be displayed inline and saved to vector `.pdf` files.

### 4.2 Matplotlib Chart Styling
- All figures must use the `Sarabun` font family:
  ```python
  import matplotlib.pyplot as plt
  plt.rcParams['font.family'] = 'Sarabun'
  plt.rcParams['font.sans-serif'] = ['Sarabun', 'TH Sarabun New', 'DejaVu Sans']
  ```
- Use publication-grade DPI (150–300), clean academic color palettes (`#1E3A8A`, `#2563EB`, `#16A34A`, `#DC2626`), transparent gridlines (`alpha=0.3`), and bilingual/English academic annotations.

---

## 5. Course Domain & Assignment Roadmap

| Assignment | Core Methodology | Primary Dataset | Key Architecture / Algorithms |
| :--- | :--- | :--- | :--- |
| **Assignment 1** | OLS Linear Regression | Synthetic / 1D | Normal Equations, Cramer's Rule, Matrix Inversion |
| **Assignment 2** | Batch Gradient Descent | Synthetic Non-linear ($N=100$) | $\hat{y} = C_0 + C_1 e^{C_2 x}$, Chain Rule Gradients, MSE Loss |
| **Assignment 3** | Survival Analysis / Regression | Telco Customer Churn ($N=7,043$) | Kaplan-Meier, Nelson-Aalen, Log-Rank Tests, Greenwood CI |
| **Assignment 4** | Tree-based & Ensemble Models | MBA Admissions ($N=6,194$) | Decision Tree, Random Forest, Gradient Boosting, Class Balancing |
| **Assignment 5** | Deep Neural Networks (DNN) | MNIST Digits ($N=70,000$) | MLP (Dense 512-256-128), Backprop Softmax+CCE, SGD/Adam/RMSprop, Dropout |
| **Assignment 6** | Convolutional Neural Networks | Dogs vs. Cats ($N=25,000$) | MobileNetV2, Inverted Residuals, Two-Phase Fine-Tuning, Real-Time Streaming |
| **Assignment 7** | Dimensionality Reduction (PCA) | MTCARS ($N=32, p=11$) \& Synthetic 2D ($N=12$) | Linear PCA, Spectral Decomposition, Standardization vs Covariance, Scree / PVE ($\ge 90\%$), 2D Biplot |
