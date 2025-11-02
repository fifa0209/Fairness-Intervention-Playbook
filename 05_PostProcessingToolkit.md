# 5. Post-Processing Fairness Toolkit

**Goal**: Improve fairness using *outputs* from existing or unmodifiable models.

## Steps

### 5.1. Group-Specific Thresholding

- Set different output score cut-offs for different groups to balance true/false positive rates.

### 5.2. Calibration

- Adjust predicted probabilities so that the interpretation is consistent across groups.

### 5.3. Prediction Transformation

- Map output scores to decisions using algorithms that ensure fairness criteria are met.

### 5.4. Rejection Option

- Let borderline cases go to manual review rather than automate potentially unfair decisions.

### 5.5. When to Use

- Ideal for production and legacy models where retraining is not possible or allowed.

---

**Resources**: AIF360 postprocessing modules, scikit-learn for calibration and thresholding
