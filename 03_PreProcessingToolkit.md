# 4. In-Processing Fairness Toolkit

**Goal**: Enforce fairness while *training* your ML model.

## Steps

### 4.1. Choose Fairness Objective

- **Demographic Parity**: Equal probability of positive outcome across groups.
- **Equalized Odds**: Equal true/false positive rates across groups.

### 4.2. Model Training Strategies

- **Constraint-based Optimization**: Add fairness constraints to loss function.
- **Adversarial Debiasing**: Train the predictor with an adversarial network trying to infer (and thus protect against) the protected attribute.
- **Fairness Regularizers**: Include penalties in the loss for unfair predictions.

### 4.3. Practical Tips

- Use frameworks like AIF360, fairlearn, or custom PyTorch/TF code.
- Tune for both accuracy *and* fairness; monitor trade-offs as you train.

### 4.4. When to Use

- When sensitive features impact model internals, and you have access/control over the model code.

---

**Resources**: AIF360, fairlearn, PyTorch, scikit-learn  
