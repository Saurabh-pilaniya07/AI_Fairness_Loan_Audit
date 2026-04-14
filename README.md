# Bias, Fairness & Explainability Audit in Loan Approval AI Systems

---

## Overview

This project investigates **algorithmic bias and fairness risks** in machine learning models used for loan approval decisions. It moves beyond traditional performance metrics to evaluate **fairness, interpretability, and ethical implications** in high-stakes AI systems.

The project implements a complete **Responsible AI pipeline**, including:

* Baseline machine learning model
* Bias detection using fairness metrics
* Bias mitigation using Fairlearn
* Model explainability using SHAP
* Analysis of fairness–accuracy trade-offs

---

## Objectives

* Detect bias across demographic groups
* Evaluate fairness using multiple metrics
* Apply mitigation techniques to reduce bias
* Analyze fairness vs accuracy trade-offs
* Align results with **EU AI Act principles**

---

## Dataset

**German Credit Dataset** (https://github.com/selva86/datasets/blob/master/GermanCredit.csv)

### Features:

* Age
* Credit amount
* Loan duration
* Financial attributes

### Target:

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
  * **Stratified 5-fold Cross-Validation**

---

### 2. Fairness Evaluation

#### Metrics:

* Demographic Parity (DP)
* Equal Opportunity (EO)

#### Advanced (Fairlearn):

* Demographic Parity Difference
* Equalized Odds Difference

---

### 3. Bias Mitigation

* Method: **Exponentiated Gradient Reduction**
* Constraint: Demographic Parity
* Tuned for balanced fairness–accuracy trade-off

---

### 4. Explainability (SHAP)

* Global feature importance
* Local explanations
* Decision behavior analysis

---

## Results

### Cross-Validation Performance

* Accuracy: **0.643 ± 0.033**

 Indicates stable and reproducible model performance.

---

### Fairness Metrics: Before vs After Mitigation

| Metric        | Before | After  | Improvement    |
| ------------- | ------ | ------ | -------------- |
| Accuracy      | 0.640  | 0.645  | +0.8%          |
| DP Difference | 0.3040 | 0.1055 | ~65% reduction |
| EO Difference | 0.3367 | 0.1512 | ~55% reduction |

---

### Visualizations

**Fairness Comparison**
![Fairness Before vs After](outputs/bias_comparison.png)

---

**Prediction Distribution (Post-Mitigation)**
![Prediction Distribution](outputs/prediction_distribution.png)

---

**SHAP Feature Importance**
![SHAP Summary](outputs/shap_summary.png)

---

## Key Research Insights

### 1. Significant Bias Reduction

* DP reduced by ~65%
* EO reduced by ~55%

 Mitigation effectively reduces demographic disparities.

---

### 2. No Accuracy Trade-off

* Accuracy slightly improved (**+0.8%**)

 Strong result: fairness improved **without degrading performance**

---

### 3. Decision Behavior Shift

* **80% approvals vs 20% rejections**

 Model becomes slightly more permissive after mitigation.

---

### 4. Model Stability & Reproducibility

* Cross-validation variance: **±0.033**
* Fixed random seeds ensure consistent results

 Results are **reliable and reproducible**

---

### 5. Fairness is Multi-Dimensional

* DP improved significantly
* EO improved moderately

 No single fairness metric captures all bias aspects

---

## Limitations

* Residual bias still exists after mitigation
* Fairness improvements may affect decision behavior
* SHAP assumes feature independence
* Results depend on dataset characteristics

---

## Ethical Implications

* Fairness ≠ correctness
* Bias reduction alone is insufficient
* Sensitive features (e.g., age) require careful governance

 Responsible AI requires balancing:

* Fairness
* Accuracy
* Real-world impact

---

## Policy & Governance Relevance

This work aligns with **EU AI Act requirements** for high-risk systems such as:

* Credit scoring
* Financial decision-making

Key compliance aspects demonstrated:

* Bias monitoring
* Transparency
* Accountability

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Fairlearn
* SHAP
* Matplotlib

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
* Cross-validation results
* Bias comparison plots
* Prediction distribution analysis
* SHAP explainability visualizations

---

## Future Work

* Multi-attribute fairness (gender, income)
* Causal fairness methods
* Real-world financial datasets
* Regulatory compliance automation

---

## Research Positioning

This project demonstrates a shift from:

> **Model-centric AI → Responsible AI systems**

Focusing on:

* Ethical deployment
* Fairness-aware design
* Policy-aligned machine learning

---

## Conclusion

This work shows that **accuracy alone is insufficient** in high-stakes AI systems.

By integrating fairness, mitigation, and explainability, it highlights:

* The importance of multi-metric evaluation
* The complexity of fairness–performance trade-offs
* The need for policy-aligned AI development

 **Responsible AI requires both technical rigor and ethical accountability.**
