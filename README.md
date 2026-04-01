# Bias, Fairness & Explainability Audit in Loan Approval AI Systems

## Overview

This project investigates **algorithmic bias and fairness issues** in machine learning models used for loan approval decisions. It moves beyond traditional performance metrics (such as accuracy) to evaluate **fairness, ethical risks, and real-world implications** of AI systems.

The project implements a complete Responsible AI pipeline:

* Baseline machine learning model
* Bias detection using fairness metrics
* Bias mitigation using Fairlearn
* Model explainability using SHAP
* Analysis of trade-offs between fairness and performance

---

## Objectives

* Detect bias in ML predictions across demographic groups
* Evaluate fairness using standard and advanced metrics
* Apply mitigation techniques to reduce bias
* Analyze trade-offs between fairness and accuracy
* Connect findings with **AI governance and regulatory frameworks (EU AI Act)**

---

## Dataset

* **German Credit Dataset**
* Features include:

  * Age
  * Credit amount
  * Loan duration
  * Financial attributes

Target Variable:

* `1` → Good credit
* `0` → Bad credit

Sensitive Attribute:

* **Age Group** (Young vs Adult)

---

## Methodology

### 1. Baseline Model

* Random Forest Classifier
* Evaluated using accuracy

---

### 2. Fairness Metrics

#### Basic Metrics:

* **Demographic Parity (DP)**
* **Equal Opportunity (EO)**

#### Advanced Metrics (Fairlearn):

* Demographic Parity Difference
* Equalized Odds Difference

---

### 3. Bias Mitigation

* Technique: **Exponentiated Gradient Reduction**
* Constraint: Demographic Parity
* Goal: Reduce disparity across demographic groups

---

### 4. Explainability (SHAP)

* Global feature importance
* Local prediction explanations
* Understanding model decision patterns

---

## Results

### Before Mitigation:

* Accuracy: **0.655**
* Demographic Parity Difference: **0.0446**
* Equalized Odds Difference: **0.1571**

### After Mitigation:

* Accuracy: **0.665**
* Demographic Parity Difference: **0.0179**
* Equalized Odds Difference: **0.0435**

---

### Comparison

| Metric        | Before | After  |
| ------------- | ------ | ------ |
| Accuracy      | 0.655  | 0.665  |
| DP Difference | 0.0446 | 0.0179 |
| EO Difference | 0.1571 | 0.0435 |

---

## Key Research Insights

### 1. Significant Bias Reduction

Bias mitigation substantially reduced disparity:

* DP Difference: **0.0446 → 0.0179 (~60% reduction)**
* EO Difference: **0.1571 → 0.0435 (~72% reduction)**

---

### 2. Fairness Without Performance Loss

Contrary to common assumptions, fairness improvements were achieved **without reducing model accuracy**.

---

### 3. Fairness vs Decision Behavior Trade-off

After mitigation, the model showed:

* **86% approvals vs 14% rejections**

This indicates fairness improvements may arise from **more permissive decision-making**, raising concerns about over-approval.

---

### 4. Fairness is Multi-Dimensional

Different fairness metrics led to different conclusions, highlighting that:

> There is no single universal definition of fairness in AI systems.

---

## Limitations

* Bias mitigation may lead to overly lenient decisions
* Results depend on dataset distribution and feature selection
* Fairness metrics do not fully capture societal impact

---

## Ethical Implications

* Reducing bias alone is not sufficient
* Decision quality and accountability must also be considered
* Responsible AI requires balancing:

  * Fairness
  * Accuracy
  * Real-world consequences

---

## Policy & Governance Relevance

This project aligns with **EU AI Act requirements** for high-risk systems such as:

* Credit scoring
* Financial decision-making

Key compliance aspects:

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
* Bias comparison plots
* Prediction distribution analysis
* SHAP explainability visualizations

---

## Future Work

* Extend analysis to multiple sensitive attributes (gender, income)
* Apply causal fairness methods
* Evaluate real-world financial datasets
* Integrate regulatory compliance checks

---

## Research Positioning

This project demonstrates a transition from:

> **Model-centric AI → Responsible AI systems**

Focusing on:

* Ethical deployment
* Fairness-aware design
* Societal impact of machine learning
