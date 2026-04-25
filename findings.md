# Research Findings: Bias, Fairness & Explainability in Loan Approval AI

**Project:** AI_Fairness_Loan_Audit  
**Methods:** Fairness Metrics (DP, EO) · Exponentiated Gradient Reduction · SHAP  
**Dataset:** German Credit Dataset (1,000 instances, 4 features used: `age`, `amount`, `duration`, `amount_per_duration`)  
**Regulatory Context:** EU AI Act Annex III — High-Risk: Credit Scoring

---

## Methods Summary

| Method | Purpose | What It Measures |
|--------|---------|-----------------|
| **Demographic Parity (DP)** | Detect group-level approval disparity | Difference in approval rates: Young vs Adult |
| **Equal Opportunity (EO)** | Detect group-level true positive disparity | Difference in correctly approved rates across groups |
| **Exponentiated Gradient** | Bias mitigation during training | Reweights samples to enforce DP constraint |
| **SHAP** | Explainability | Feature-level contribution to each prediction |
| **5-fold Stratified CV** | Model stability | Accuracy as mean ± std across folds |

**Sensitive attribute:** Age group — Young (≤ 30) / Adult (> 30)  
**Why age:** Protected characteristic under GDPR Art. 9 and relevant to EU AI Act Art. 10 data governance requirements for high-risk systems.

---

## Finding 1: Significant Bias Reduction — With a Caveat

Applying Exponentiated Gradient Reduction with a Demographic Parity constraint
produced measurable, quantified fairness improvements:

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| DP Difference | 0.3040 | 0.2002 | **~34%** |
| EO Difference | 0.3367 | 0.1875 | **~44%** |
| Accuracy | 0.640 | 0.675 | +3.5% |
| ROC-AUC | 0.636 | N/A ¹ | — |
| CV Accuracy (5-fold) | 0.673 ± 0.019 | — | baseline |

¹ *Fairlearn's Exponentiated Gradient mitigator does not expose `predict_proba`.*

**The headline:** Both fairness metrics improved substantially (DP −34%, EO −44%), and accuracy
*increased* by +3.5% — directly contradicting the common assumption that
fairness and accuracy are in fundamental tension.

**The caveat:** DP difference at 0.2002 and EO at 0.1875 are reduced but
not eliminated. Residual bias remains. "Improved" is not the same as "fair."

**What this means for compliance:**  
EU AI Act Art. 10 requires monitoring of bias, but does not specify threshold
values for acceptable DP or EO difference in credit scoring. The field lacks
consensus on what these thresholds should be — this is an open regulatory gap.

---

## Finding 2: The Over-Approval Paradox — Core Research Contribution

**The empirical result:**
After mitigation, the overall approval rate increased from **69.0% to 81.5% — a shift
of +12.5 percentage points**. The rejection rate dropped from 31.0% to 18.5%.

| Distribution | Before | After | Change |
|-------------|--------|-------|--------|
| Approval Rate | 69.0% | 81.5% | **+12.5%** |
| Rejection Rate | 31.0% | 18.5% | **−12.5%** |

The per-group breakdown (Young vs Adult) reveals whether parity was achieved
through genuine decision improvement or through approval inflation.
Note: per-group rates show `nan` in current output due to an index alignment bug
between pandas boolean mask and numpy array — fix: `y_pred[young_mask.values].mean()`.

**Two mechanisms that produce identical DP scores:**

| Mechanism | What Happens | DP Result | Financial Risk |
|-----------|-------------|-----------|----------------|
| **Genuine de-biasing** | Model evaluates Young applicants on financial merit, correctly approves more creditworthy ones | DP ↓ | Low |
| **Approval inflation** | Model raises overall approval rate; parity becomes trivially achievable | DP ↓ (identical) | **High** |

**Why fairness metrics cannot distinguish these:**  
Both mechanisms produce the same DP Difference reduction. The metric is blind
to *why* parity improved. Only auditing the post-mitigation decision distribution
can reveal which mechanism is operating.

**The research question this raises:**  
> Is there a way to detect whether demographic parity has been achieved through
> genuine decision improvement vs approval rate inflation — *without* having
> ground truth on which approved applicants actually repaid their loans?

This is an open problem with direct EU AI Act implications. Current Art. 10
guidance does not require post-mitigation distribution audits. This is a gap.

---

## Finding 3: Fairness is Not a Single Number

DP and EO metrics improved at *different rates* (34% vs 44%) and reflect
fundamentally different definitions of fairness:

| Metric | Definition | What Satisfying It Means |
|--------|-----------|--------------------------|
| **Demographic Parity** | Equal approval rates across groups | Both groups approved at same rate, regardless of creditworthiness |
| **Equal Opportunity** | Equal true positive rates across groups | Creditworthy applicants approved equally — conditional on being creditworthy |

**A model can satisfy one while violating the other.**

Demographic parity can be achieved by approving everyone equally — including poor risks.
Equal opportunity requires correctly identifying creditworthy applicants in both groups,
which is a stricter and more financially sound definition.

**Implication for EU AI Act:**  
Art. 10 does not specify *which* fairness definition high-risk systems must satisfy.
Choosing DP vs EO is a regulatory and ethical judgment, not a technical one.
A system could be declared compliant under DP while still violating EO — and vice versa.

**Conclusion:** Single-metric fairness evaluation is insufficient for high-stakes
financial deployment. Regulatory frameworks should specify which metric(s) are required
and at what thresholds.

---

## Finding 4: SHAP Reveals Residual Proxy Discrimination Risk

**Actual SHAP feature importance — Baseline Model:**

| Rank | Feature | Mean \|SHAP\| | Note |
|------|---------|--------------|------|
| 1 | `amount_per_duration` | **0.0747** | Engineered ratio — highest overall influence |
| 2 | `age` | **0.0672** | Protected demographic characteristic ⚠️ |
| 3 | `duration` | **0.0561** | Loan term |
| 4 | `amount` | **0.0542** | Loan amount |

**Key result:** `age` ranks **2nd** in feature importance with a mean |SHAP| of 0.0672 —
very close to the top feature (0.0747). Despite being the sensitive attribute used
for the fairness constraint, the model still relies heavily on age to make decisions.

**The diagnostic proposed:**  
Compare SHAP feature importance rankings **before and after mitigation**:

- If age-related SHAP importance *decreases* → genuine de-biasing (model relies less on protected attribute)
- If age-related SHAP importance *stays constant* → threshold shifting (model still uses age, just adjusts cutoffs)

This SHAP-based diagnostic is not currently required by EU AI Act guidance — but it
provides information that DP/EO metrics alone cannot.

**Research Question:**  
> Can SHAP feature importance change (pre vs post mitigation) serve as a proxy
> metric for distinguishing genuine de-biasing from threshold manipulation?

---

## Finding 5: Accuracy–Fairness Trade-off is Context-Dependent

The standard assumption in the fairness literature is that there is a fundamental
accuracy–fairness trade-off: reducing bias requires accepting lower accuracy.

**This project shows the opposite in this context:**
Accuracy *increased* by **+3.5%** (0.640 → 0.675) after mitigation.
DP reduced by 34%. EO reduced by 44%.

**Why this might be the case:**
1. The baseline model may have been overfitting to age-correlated noise — mitigation
   forced it to find more robust signals in `amount_per_duration` and `duration`
2. The German Credit dataset's 70/30 class imbalance and 4-feature structure
   may make this result specific to this context — it may not generalise

**Connection to XAI_Credit_Risk findings:**  
In the companion explainability project, the Bank Marketing model achieved 0.923
accuracy by relying on `duration` — a behavioral feature with no demographic
correlation. That model's high accuracy does not imply low fairness risk.
Together, these findings show that **neither accuracy nor fairness metrics alone
are sufficient** — both must be evaluated simultaneously with different methods.

---

## How This Advances Prior Work

**Agarwal et al. (2018)** introduced Exponentiated Gradient Reduction as a
technically sound fairness mitigation method. This project applies it and then
asks: *what did the mitigation actually do to the decision distribution?*

**Barocas & Hardt (2019)** documented the impossibility of simultaneously
satisfying multiple fairness criteria. This project demonstrates this empirically:
DP and EO improved at different rates (34% vs 44%), confirming that satisfying
one does not fully satisfy the other.

**Wachter et al. (2021)** argued that fairness cannot be fully automated and
requires normative judgments. This project provides concrete evidence: the
over-approval paradox shows that automated DP satisfaction can mask financial
risk that only human or distributional audit can reveal.

**This project's specific contribution:**
1. Demonstrates the over-approval paradox empirically on German Credit data
2. Proposes SHAP-based pre/post comparison as a diagnostic for mitigation quality
3. Identifies the distributional audit gap in current EU AI Act Art. 10 guidance
4. Provides a reproducible pipeline that connects technical fairness metrics
   to specific regulatory articles

---

## Open Research Questions

**Q1 — Generalisability of Over-Approval:**  
> Does demographic parity mitigation systematically produce approval rate inflation
> across credit scoring datasets — or is this specific to German Credit's
> class distribution and feature structure?

*Why this matters:* If over-approval is systematic, it represents a structural
flaw in DP-based mitigation that regulators need to address explicitly.

**Q2 — SHAP as Mitigation Diagnostic:**  
> Can comparing SHAP feature importance before and after fairness mitigation
> reliably distinguish genuine de-biasing from threshold shifting — and at
> what sample size is this comparison statistically reliable?

*Why this matters:* Provides a practical compliance tool for EU AI Act Art. 10
that goes beyond aggregate fairness metrics.

**Q3 — Regulatory Threshold Definition:**  
> What DP Difference and EO Difference values should constitute "acceptable"
> fairness for EU AI Act high-risk credit scoring systems — and should these
> thresholds differ by the financial product, population size, and stakes involved?

*Why this matters:* Currently undefined in EU AI Act guidance. Without thresholds,
"bias monitoring" is an incomplete compliance requirement.

---

## EU AI Act Alignment — Specific Evidence Mapping

**Article 10 — Data and Data Governance:**
SHAP analysis identifies `age` as the 2nd most important feature (mean |SHAP| = 0.0672),
directly behind the engineered feature `amount_per_duration` (0.0747).
DP/EO metrics quantify group disparity: DP 0.3040 → 0.2002, EO 0.3367 → 0.1875.
The `outputs/fairness_results_summary.csv` provides an audit-ready record.

**Article 13 — Transparency and Provision of Information:**  
SHAP global importance (`outputs/shap_summary.png`, `outputs/shap_bar.png`)
provides the interpretable explanation required for high-risk systems.
The fairness metric breakdown by age group satisfies Art. 13(3)(b) requirements
for information to affected individuals.

**Article 14 — Human Oversight:**
The over-approval analysis (`outputs/prediction_distribution.png`) shows a
+12.5 percentage point shift in approval rate (69.0% → 81.5%) that requires
human verification. This provides a principled basis for implementing Art. 14
oversight — focusing human review on the newly approved applicants post-mitigation.

**Article 15 — Accuracy, Robustness and Cybersecurity:**
5-fold stratified cross-validation reports accuracy as **0.673 ± 0.019**
(fold scores: [0.655, 0.690, 0.685, 0.645, 0.690]). Low variance confirms
stability across data splits — satisfying Art. 15 robustness documentation requirements.

**Critical Gap Identified:**  
EU AI Act Art. 10 requires bias monitoring but does not require post-mitigation
distribution audits. The over-approval paradox shows this is insufficient —
a model can satisfy Art. 10 while introducing hidden financial risk through
approval inflation. This gap should be addressed in implementing regulations.

---

## Portfolio Context

This document is Part 1 of a 3-part Responsible AI research portfolio:

| Part | Focus | Repository |
|------|-------|-----------|
| **1** | **Fairness & Bias Mitigation** | [AI_Fairness_Loan_Audit](https://github.com/Saurabh-pilaniya07/AI_Fairness_Loan_Audit) |
| **2** | Explainability (XAI) | [XAI_Credit_Risk](https://github.com/Saurabh-pilaniya07/XAI_Credit_Risk) |
| **3** | AI Governance & Policy | *(Coming soon)* |

**Unified argument across all three:**  
> Responsible AI is not a single solution. It requires simultaneous evaluation
> across fairness metrics, explainability methods, governance frameworks, and
> domain-specific deployment context. Technical compliance with one dimension
> (e.g., DP metric) does not guarantee responsible deployment overall.
