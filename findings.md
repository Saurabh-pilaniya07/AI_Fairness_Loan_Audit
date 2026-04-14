# Key Research Findings

## Finding 1: Bias Mitigation Can Mask Risk
Enforcing demographic parity reduced DP difference by ~60% and EO 
difference by ~72%. However, this came with a significant shift in 
prediction distribution — 86% approvals post-mitigation vs a more 
balanced split before. This raises the question of whether the model 
is genuinely fairer or simply more permissive.

## Finding 2: Fairness Metrics Disagree
DP and EO metrics do not always move together. A model can satisfy 
demographic parity while violating equalized odds. This confirms that 
fairness is not a single measurable property.

## Finding 3: The Over-Approval Paradox
When demographic parity is enforced as a hard constraint, the 
optimization may increase approvals for the disadvantaged group 
rather than reduce approvals for the advantaged group. This creates 
a hidden financial risk disguised as fairness compliance.

## Implications for EU AI Act
Credit scoring is classified as high-risk under Annex III. These 
findings suggest that satisfying Article 10 (data governance) and 
Article 13 (transparency) requirements may be insufficient without 
also auditing decision distribution shifts post-mitigation.