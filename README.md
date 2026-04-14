# Bias, Fairness & Explainability Audit in Loan Approval AI Systems

---

## Overview

This project investigates **algorithmic bias and fairness risks** in machine learning systems used for loan approval decisions. It moves beyond conventional performance metrics (e.g., accuracy) to evaluate **fairness, interpretability, and ethical implications** in high-stakes AI applications.

The work implements a **Responsible AI pipeline**, integrating:

* Baseline machine learning modeling
* Bias detection using fairness metrics
* Bias mitigation using Fairlearn
* Model explainability using SHAP
* Empirical analysis of fairness–accuracy trade-offs

---

## Objectives

* Detect bias in predictions across demographic groups
* Evaluate fairness using standard and advanced metrics
* Apply mitigation techniques to reduce disparity
* Analyze trade-offs between fairness and predictive performance
* Align technical findings with **AI governance frameworks (EU AI Act)**

---

## Dataset

**German Credit Dataset**
(Source: UCI Machine Learning Repository)

### Features include:

* Age
* Credit amount
* Loan duration
* Financial attributes

### Target Variable:

* `1` → Good credit
* `0` → Bad credit

### Sensitive Attribute:

* **Age Group** (Young vs Adult)

---

## Methodology

### 1. Model

* Random Forest Classifier
* Evaluated using:

  * Train/Test split
  * **5-fold Cross-Validation**

---

### 2. Fairness Evaluation

#### Basic Metrics:

* Demographic Parity (DP)
* Equal Opportunity (EO)

#### Advanced Metrics (Fairlearn):

* Demographic Parity Difference
* Equalized Odds Difference

---

### 3. Bias Mitigation

* Technique: **Exponentiated Gradient Reduction**
* Constraint: Demographic Parity
* Objective: Reduce disparity across sensitive groups

---

### 4. Explainability (SHAP)

* Global feature importance
* Local prediction explanations
* Understanding model decision structure

---

## Results

### Cross-Validation Performance

* Accuracy: **0.665 ± 0.018**

This indicates stable model performance and low variance across folds.

---

### Before Mitigation

* Accuracy: **0.670**
* Demographic Parity Difference: **0.0506**
* Equalized Odds Difference: **0.1367**

---

### After Mitigation

* Accuracy: **0.645**
* Demographic Parity Difference: **0.0164**
* Equalized Odds Difference: **0.1041**

---

### Comparison

| Metric        | Before | After  |
| ------------- | ------ | ------ |
| Accuracy      | 0.670  | 0.645  |
| DP Difference | 0.0506 | 0.0164 |
| EO Difference | 0.1367 | 0.1041 |

---

## Key Research Insights

### 1. Significant Bias Reduction

* DP Difference reduced by **~67%**
* EO Difference reduced by **~24%**

Bias mitigation effectively reduces demographic disparities.

---

### 2. Fairness–Accuracy Trade-off

* Accuracy decreased: **0.670 → 0.645**

Insight:

> Fairness improvements may come at the cost of predictive performance, highlighting inherent trade-offs in responsible AI systems.

---

### 3. Decision Behavior Shift

Post-mitigation predictions:

* **83% approvals**
* **17% rejections**

Interpretation:

> The model becomes more permissive, which may improve fairness but introduces potential risks of over-approval.

---

### 4. Model Stability

* Cross-validation variance is low (**±0.018**)

Insight:

> The model generalizes well and is not overfitting to a specific split.

---

### 5. Fairness is Multi-Dimensional

* DP improved significantly
* EO improved moderately

Insight:

> Different fairness metrics capture different aspects of bias; no single metric is sufficient.

---

## Limitations

* Bias mitigation may lead to overly permissive decision behavior
* Results depend on dataset characteristics and feature selection
* Fairness metrics do not fully capture societal or long-term impact
* SHAP assumes feature independence

---

## Ethical Implications

* Reducing bias alone is insufficient
* Decision quality and accountability must be considered
* Sensitive attributes (e.g., age) require careful handling

Responsible AI requires balancing:

* Fairness
* Accuracy
* Real-world consequences

---

## Policy & Governance Relevance

This work aligns with **high-risk AI system requirements** under the EU AI Act, particularly in:

* Credit scoring
* Financial decision-making

Key compliance dimensions demonstrated:

* Bias monitoring
* Transparency through explainability
* Accountability via measurable metrics

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Fairlearn
* SHAP
* Matplotlib, Seaborn

---

## Project Structure

```
ai-fairness-loan-audit/
│
├── data/
├── notebooks/
├── src/
├── outputs/
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Outputs

* Fairness metrics (before vs after mitigation)
* Cross-validation performance
* Bias comparison visualizations
* Prediction distribution analysis
* SHAP explainability plots

---

## Future Work

* Extend analysis to multiple sensitive attributes (gender, income)
* Integrate causal fairness approaches
* Evaluate on real-world financial datasets
* Incorporate regulatory compliance auditing pipelines

---

## Research Positioning

This project reflects a transition from:

> **Model-centric AI → Responsible AI systems**

It emphasizes:

* Ethical deployment
* Fairness-aware design
* Alignment between machine learning systems and regulatory frameworks

---

## Conclusion

This work demonstrates that **accuracy alone is insufficient** for evaluating AI systems in high-stakes domains.

By integrating fairness evaluation, mitigation, and explainability, it highlights the need for:

* Multi-dimensional evaluation
* Trade-off awareness
* Policy-aligned system design

**Responsible AI requires both technical rigor and ethical accountability.**

---