# Bias, Fairness & Explainability Audit in Loan Approval AI Systems

> **When demographic parity is enforced as a hard constraint, does the optimizer achieve genuine fairness — or simply inflate approval rates?**

A complete Responsible AI audit pipeline applying bias detection, fairness-aware mitigation,
and SHAP explainability to credit scoring decisions — with direct EU AI Act compliance mapping.

---

## Why This Matters

Credit scoring models affect millions of lending decisions. When these models encode
demographic bias — even unintentionally — the consequences are:

- **Discriminatory outcomes** for protected groups (age, gender, ethnicity)
- **Regulatory liability** under EU AI Act Annex III (credit scoring = high-risk)
- **Hidden financial risk** when "fair" models over-approve unsuitable applicants

This project shows that **measuring fairness metrics is not enough**.
The over-approval paradox reveals a risk that demographic parity compliance can mask.

---

## Research Question

> When demographic parity is enforced as a hard constraint on a credit scoring model,
> does the optimizer achieve parity by genuinely improving decisions for the
> disadvantaged group — or simply by inflating approvals across the board?

If the latter, the model is *statistically fair but financially unsound* — a form
of hidden risk that fairness metrics alone cannot detect.

---

## Key Findings

| Finding | Result |
|---------|--------|
| **DP Difference** (before → after) | 0.3040 → 0.2002 (~34% reduction) |
| **EO Difference** (before → after) | 0.3367 → 0.1875 (~44% reduction) |
| **Accuracy** (before → after) | 0.640 → 0.675 (+3.5% — fairness improved accuracy) |
| **ROC-AUC** (baseline) | 0.636 test / 0.637 ± 0.024 cross-validated |
| **CV Accuracy** (5-fold stratified) | 0.673 ± 0.019 — stable across all folds |
| **Overall approval rate shift** | 69.0% → 81.5% (+12.5% post-mitigation) |
| **Top SHAP feature** | `amount_per_duration` (0.0747) — followed by `age` (0.0672) |
| **Core insight** | Fairness metrics can be satisfied through approval inflation, not genuine de-biasing |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | German Credit Dataset |
| Source | https://github.com/selva86/datasets/blob/master/GermanCredit.csv |
| Instances | 1,000 |
| Features used | 4 (`age`, `amount`, `duration`, `amount_per_duration`) |
| Target | Credit risk: Good (1) / Bad (0) |
| Class balance | 700 Good (70%) / 300 Bad (30%) |
| Train / Test split | 800 / 200 (stratified) |
| Sensitive attribute | Age group: Young / Adult |
| License | CC BY 4.0 |
| EU AI Act classification | **High-risk** (Annex III) |

---

## Methodology

### Pipeline Overview

```
German Credit Dataset (1,000 instances)
        ↓
Preprocessing → 4 features: age, amount, duration, amount_per_duration
        ↓
Stratified Train/Test Split (800 / 200)
        ↓
Baseline Random Forest → 5-Fold Stratified CV + Test Set Evaluation
        ↓
Baseline Fairness Audit (DP Difference: 0.3040, EO Difference: 0.3367)
        ↓
Exponentiated Gradient Reduction (Demographic Parity constraint)
        ↓
Post-Mitigation Audit → Over-Approval Analysis → SHAP Explainability
        ↓
EU AI Act Compliance Mapping
```

### 1. Baseline Model

- **Algorithm:** Random Forest Classifier (100 trees)
- **Validation:** 5-fold stratified cross-validation
- **Metrics:** Accuracy, ROC-AUC, classification report per class

### 2. Fairness Evaluation

| Metric | Question |
|--------|----------|
| **Demographic Parity (DP)** | Do both age groups get approved at equal rates? |
| **Equal Opportunity (EO)** | Are creditworthy applicants approved equally across groups? |
| **Demographic Parity Difference** | Absolute difference in approval rates between groups |
| **Equalized Odds Difference** | Maximum difference in TPR/FPR between groups |

### 3. Bias Mitigation

- **Method:** Exponentiated Gradient Reduction (Agarwal et al., 2018)
- **Constraint:** Demographic Parity
- **Mechanism:** Iteratively reweights training samples to satisfy the fairness constraint during training — not post-processing

### 4. Explainability (SHAP)

- Global feature importance (mean |SHAP| across all predictions)
- Feature importance before mitigation — identifies which features drive decisions
- Bar chart with age-related features highlighted in red

---

## Results

### Cross-Validation — Baseline Model

| Metric | Mean | Std | Fold Scores |
|--------|------|-----|-------------|
| Accuracy | **0.673** | ±0.019 | [0.655, 0.690, 0.685, 0.645, 0.690] |
| ROC-AUC | **0.637** | ±0.024 | — |

Low variance (±0.019) confirms stable, reproducible performance across all data splits.

### Model Performance — Test Set

| Metric | Before Mitigation | After Mitigation | Change |
|--------|------------------|-----------------|--------|
| Accuracy | 0.640 | **0.675** | **+3.5%** |
| ROC-AUC | 0.636 | N/A ¹ | — |
| CV Accuracy (5-fold) | 0.673 ± 0.019 | — (baseline only) | — |

¹ *Fairlearn's Exponentiated Gradient mitigator does not expose `predict_proba`,
so post-mitigation ROC-AUC cannot be computed directly.*

### Classification Report — Baseline Model

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bad Credit (0) | 0.40 | 0.42 | 0.41 | 60 |
| Good Credit (1) | 0.75 | 0.74 | 0.74 | 140 |
| **Weighted avg** | **0.64** | **0.64** | **0.64** | **200** |

### Fairness Metrics — Before vs After Mitigation

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| DP Difference | 0.3040 | **0.2002** | **−34%** |
| EO Difference | 0.3367 | **0.1875** | **−44%** |
| Overall Approval Rate | 69.0% | **81.5%** | **+12.5%** |
| Overall Rejection Rate | 31.0% | **18.5%** | **−12.5%** |

### SHAP Feature Importance — Baseline Model

| Rank | Feature | Mean \|SHAP\| | Note |
|------|---------|--------------|------|
| 1 | `amount_per_duration` | **0.0747** | Engineered ratio feature |
| 2 | `age` | **0.0672** | Protected demographic characteristic |
| 3 | `duration` | **0.0561** | Loan term |
| 4 | `amount` | **0.0542** | Loan amount |

> **Fairness concern:** `age` ranks 2nd in feature importance despite being the sensitive
> attribute used for the fairness constraint. This indicates potential proxy discrimination
> may persist — the model relies on a protected characteristic even after mitigation targets it.

---

## Visualisations

### Fairness Comparison — Before vs After Mitigation

![Fairness Comparison](outputs/bias_comparison.png)

*DP reduced by 34%, EO reduced by 44%. Accuracy improved by +3.5% —
fairness constraints did not degrade performance, they improved it.*

---

### Over-Approval Analysis — Prediction Distribution Shift

![Prediction Distribution](outputs/prediction_distribution.png)

*Overall approval rate increased from 69.0% to 81.5% (+12.5%) after mitigation.
This +12.5% shift is the empirical basis for the over-approval paradox: demographic
parity was achieved, but at the cost of approving significantly more applicants overall.*

---

### SHAP Feature Importance — What Drives Credit Decisions?

![SHAP Summary](outputs/shap_summary.png)

*`amount_per_duration` is the strongest global predictor (0.0747), followed closely
by `age` (0.0672). The presence of `age` in second position — despite being the
protected attribute — indicates the model still relies on demographic information
to make credit decisions.*

---

## The Over-Approval Paradox

This is the core research contribution of this project.

**The empirical finding:**
After enforcing demographic parity, the overall approval rate increased from
**69.0% to 81.5% — a shift of +12.5 percentage points**.

Two fundamentally different mechanisms could explain this:

1. **Genuine de-biasing:** The model learned to assess Young applicants on financial
   merit rather than age, correctly approving more creditworthy Young applicants
2. **Approval inflation:** The optimizer raised the overall approval rate, making
   parity trivially achievable by approving nearly everyone

**Why fairness metrics cannot distinguish these:**
Both mechanisms produce the same DP Difference reduction (34% in both cases).
Only auditing *who* is being newly approved — and whether those applicants are
genuinely creditworthy — can reveal which mechanism is operating.

**EU AI Act implication:** Art. 10 requires bias monitoring, but the current
guidance does not require post-mitigation distribution audits. A +12.5 percentage
point shift in approval rates is a material change that current compliance
frameworks would not flag. This is a regulatory gap.

---

## EU AI Act Alignment

| Article | Requirement | This Project's Evidence |
|---------|-------------|------------------------|
| **Art. 10** | Data governance — monitor protected characteristics | DP/EO metrics quantify age group disparity; SHAP confirms age ranks 2nd in importance (0.0672) |
| **Art. 13** | Transparency — interpretable explanations | SHAP global importance table + summary plot; per-class classification report |
| **Art. 14** | Human oversight — identify cases needing review | +12.5% approval shift flags systematic decisions needing human verification |
| **Art. 15** | Accuracy & robustness — stable performance | 5-fold CV: 0.673 ± 0.019 — low variance confirms stability across splits |

---

## Limitations

**Technical:**
- Only 4 features used (`age`, `amount`, `duration`, `amount_per_duration`) — original dataset has 20; feature selection may have removed fairness-relevant signals
- Per-group approval rates (Young vs Adult) show `nan` due to an index alignment bug between pandas boolean mask and numpy array — fix: use `y_pred[young_mask.values].mean()`
- Only one sensitive attribute analysed (age group) — intersectional analysis (age × gender) not yet included
- Exponentiated Gradient Reduction is one of several mitigation techniques — Reweighing and Calibrated EO not compared
- SHAP assumes feature independence — `amount_per_duration` is derived from `amount` and `duration`, creating dependency

**Methodological:**
- Over-approval analysis is quantitative (+12.5%) but lacks ground truth — we cannot determine whether newly approved applicants were genuinely creditworthy without follow-up repayment data
- EO difference still at 0.1875 post-mitigation — substantial residual bias remains
- Single dataset (1,000 instances) limits generalisability of the over-approval finding
- Post-mitigation ROC-AUC unavailable — Fairlearn mitigator does not expose `predict_proba`

**Scope:**
- This project detects bias and applies mitigation — it does not determine whether decisions violate any specific regulation (that requires legal analysis)
- The fairness–accuracy improvement (+3.5%) may not generalise to larger or more balanced datasets

---

## Related Work

- Agarwal et al. (2018) — *A Reductions Approach to Fair Classification* (Exponentiated Gradient)
- Barocas & Hardt (2019) — *Fairness and Machine Learning* (fairness metrics framework)
- Wachter et al. (2021) — *Why Fairness Cannot Be Automated* (limits of technical fairness)
- Selbst et al. (2019) — *Fairness and Abstraction in Sociotechnical Systems*

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Model training, cross-validation, metrics |
| `fairlearn` | Fairness metrics, Exponentiated Gradient Reduction |
| `shap` | Feature importance and explainability |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |

**Python version:** 3.8+

---

## Project Structure

```
AI_Fairness_Loan_Audit/
│
├── data/
│   └── german_credit.csv
├── notebooks/
│   └── 01_fairness_analysis.ipynb   ← Full analysis (start here)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── fairlearn_analysis.py
├── outputs/
│   ├── bias_comparison.png
│   ├── prediction_distribution.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── fairness_results_summary.csv
├── findings.md                      ← Standalone research findings document
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+

### Step 2 — Download dataset

### Step 3 — Run the full pipeline

```bash
python main.py
```

### Step 4 — Explore the notebook

Open `notebooks/01_fairness_analysis.ipynb` for step-by-step analysis
with interpretations, visualisations, and research framing.

---

## Portfolio Context

This project is **Part 1 of a 3-part Responsible AI portfolio**:

| Part | Focus | Repository |
|------|-------|-----------|
| **1** | **Fairness & Bias Mitigation** | **This repository** |
| 2 | Explainability (XAI) | [XAI_Credit_Risk](https://github.com/Saurabh-pilaniya07/XAI_Credit_Risk) |
| 3 | AI Governance & Policy | *(Coming soon)* |

**The unified argument:**
Responsible AI is not a single solution. It requires integrated evaluation across
technical methods (fairness metrics, XAI), governance frameworks (EU AI Act compliance),
and domain-specific context — simultaneously.

---

## Research Positioning

This work moves beyond asking *"Is the model accurate?"* to asking:

> *"Does the model treat all demographic groups equitably — and how can we tell?"*

**Technical contribution:** A reproducible fairness audit pipeline applying DP/EO
metrics, Exponentiated Gradient Reduction, and SHAP explainability to credit scoring —
with all results verified through 5-fold stratified cross-validation.

**Research contribution:** Empirical evidence that the over-approval paradox arises
from constraint-based fairness mitigation (+12.5% approval shift), and that `age`
remains a top-2 SHAP feature after mitigation — suggesting threshold manipulation
rather than genuine de-biasing. Both findings represent gaps not addressed by
current EU AI Act guidance.
