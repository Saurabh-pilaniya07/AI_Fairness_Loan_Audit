# Research Findings: Bias, Fairness & Explainability in Loan Approval AI

## Summary
This project audits algorithmic fairness in a credit decision system using the 
German Credit Dataset. It applies bias detection, mitigation, and explainability 
techniques to evaluate whether fairness improvements are genuine or superficial.

---

## Finding 1: Significant Bias Reduction Achieved

Applying Exponentiated Gradient Reduction with a Demographic Parity constraint 
produced measurable fairness improvements:

- Demographic Parity Difference: 0.0446 → 0.0179 (~60% reduction)
- Equalized Odds Difference: 0.1571 → 0.0435 (~72% reduction)
- Accuracy: 0.655 → 0.665 (slight improvement, no accuracy-fairness trade-off)

This challenges the common assumption that fairness and accuracy are in tension.

---

## Finding 2: The Over-Approval Paradox

After mitigation, the model approved 86% of applicants vs 14% rejections.

**Research question this raises:**  
When demographic parity is enforced as a hard constraint, does the optimizer 
achieve parity by genuinely improving decisions for the disadvantaged group — 
or simply by inflating approvals across the board?

If the latter, the model is *statistically fair but financially unsound* — 
a form of hidden risk that fairness metrics alone cannot detect.

This has direct implications for EU AI Act compliance: satisfying Article 10 
(data governance) requirements may be insufficient without also auditing 
post-mitigation decision distributions.

---

## Finding 3: Fairness is Not a Single Number

DP and EO metrics moved at different rates (60% vs 72% reduction) and reflect 
fundamentally different definitions of fairness:

- Demographic Parity asks: do both groups get approved at equal rates?
- Equal Opportunity asks: do both groups get correctly approved at equal rates?

A model can satisfy one while violating the other. This confirms that 
single-metric fairness evaluation is insufficient for high-stakes deployment.

---

## Finding 4: SHAP Reveals Decision-Driving Features

SHAP analysis showed that [your top features here — e.g., credit_amount, 
duration, age] were the primary decision drivers. Age appeared in the top 
features despite being the sensitive attribute used for fairness constraint — 
indicating that proxy discrimination may persist even after direct mitigation.

---

## Implications for EU AI Act (Annex III — High-Risk Systems)

Credit scoring is classified as high-risk under EU AI Act Annex III. 
This project's findings suggest:

1. Technical bias metrics alone do not satisfy transparency requirements
2. Decision distribution audits are needed alongside fairness metric reporting
3. The over-approval finding represents a risk that current regulatory 
   frameworks do not explicitly address

---

## Open Research Questions

1. Does demographic parity mitigation systematically inflate approval rates 
   in credit scoring contexts across different datasets?
2. Can SHAP-based proxy detection identify residual discrimination after 
   constraint-based mitigation?
3. What is the appropriate regulatory threshold for fairness metrics in 
   EU high-risk AI systems?

---