# 6. End-to-End Case Study & Example Workflow

A sample walkthrough, applying the Playbook to a loan approval bias problem.

## 6.1. Audit and Diagnose

- Assess representation and outcomes across gender/race groups.
- Construct causal graph to map possible discrimination paths.

## 6.2. Select Interventions

- If data is skewed: Pre-process with reweighting/resampling.
- If model treats groups unequally: Apply in-processing constraints or adversarial debiasing.
- If model cannot be changed: Use post-processing thresholds/calibrations.

## 6.3. Validate Results

- Inspect fairness metrics before and after (e.g., parity gap, equalized odds gap, accuracy retention).
- Document all intervention and outcome steps.

## 6.4. Key Output Examples

- *"Gender approval gap from 5% → 1%."*
- *"Model accuracy drop: <2% after intervention."*

## 6.5. Tips

- Adapt steps for different domains (healthcare, hiring, insurance) and protected attributes.
- Document assumptions and monitor for new sources of bias after deployment.

---

## References & Further Reading

- [AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/)
- [Fairlearn](https://fairlearn.org/)
- [doWhy (Causal Modelling)](https://www.microsoft.com/en-us/research/project/dowhy/)
