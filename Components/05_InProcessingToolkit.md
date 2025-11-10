# 05: In-Processing Fairness Toolkit

## Purpose

Embed fairness directly into model training by modifying the learning algorithm itself. Address bias at its core by optimizing for both accuracy and fairness simultaneously.

**When to Use**: 
- Retraining model or building new model from scratch
- Want to address root causes at model level
- Need mathematical fairness guarantees

**Time**: 2-3 weeks  
**Expected Impact**: 60-85% gap reduction

---

## 5.1 Three Core Techniques

### Technique 1: Constrained Optimization

**Purpose**: Add explicit fairness constraints to the model's objective function

**When to Use**:
- Need mathematical guarantees about fairness
- Stakeholders require specific fairness criteria (e.g., equal opportunity)
- Model type supports constraint optimization (logistic regression, SVM, gradient boosting)

**How It Works**:
Standard machine learning optimizes: `Minimize Loss(predictions, labels)`

Constrained optimization adds fairness requirements:
```
Minimize: Loss(predictions, labels)
Subject to: Fairness_Constraint(predictions, protected_attributes) ≤ ε
```

**Example - Equal Opportunity Constraint**:
```
Minimize: Binary Cross-Entropy Loss
Subject to: |TPR_male - TPR_female| ≤ 0.05
```
This ensures true positive rates differ by at most 5% between groups.

**Four Common Fairness Constraints**:

1. **Demographic Parity** (Statistical Parity):
   - Constraint: P(ŷ=1|A=0) = P(ŷ=1|A=1)
   - Meaning: Equal approval/positive prediction rates across groups
   - Use when: Equal outcomes desired regardless of qualifications
   - Banking example: Marketing offers distributed equally

2. **Equal Opportunity**:
   - Constraint: P(ŷ=1|Y=1,A=0) = P(ŷ=1|Y=1,A=1)
   - Meaning: Among qualified individuals, equal approval rates
   - Use when: Want to ensure qualified people treated equally
   - Banking example: Among creditworthy borrowers, equal approval rates

3. **Equalized Odds**:
   - Constraint: Both TPR and FPR equal across groups
   - Meaning: Equal error rates for qualified and unqualified
   - Use when: Both false positives and false negatives matter
   - Banking example: Fraud detection with balanced error rates

4. **Predictive Parity** (Calibration):
   - Constraint: P(Y=1|ŷ=1,A=0) = P(Y=1|ŷ=1,A=1)
   - Meaning: Predicted risk equals actual risk for all groups
   - Use when: Predictions used for resource allocation
   - Banking example: Loan default predictions equally accurate

**Implementation Approach**:

**Step 1: Choose Fairness Constraint**
- Align with stakeholder values and regulatory requirements
- Banking lending: Equal Opportunity most common (ECOA alignment)
- Multiple objectives: Can combine constraints (e.g., EO + Calibration)

**Step 2: Formulate Constrained Problem**
Using Lagrangian method to convert hard constraints to soft penalties:
```
Loss_total = Loss_task + λ * Fairness_Violation

Where:
- Loss_task = Standard ML objective (e.g., BCE)
- Fairness_Violation = Degree constraint violated
- λ = Fairness strength parameter (tunable)
```

**Step 3: Optimize with Constraints**
- Use specialized optimizers (e.g., Fairlearn's ExponentiatedGradient)
- Or: Add penalty term to standard optimizer
- Iterate: Adjust λ if fairness or performance unsatisfactory

**Step 4: Validate Trade-off**
- Plot fairness vs accuracy (Pareto frontier)
- Select operating point with stakeholders
- Document chosen λ and rationale

**Trade-offs**:
- ✅ Mathematical fairness guarantees
- ✅ Addresses root cause in model
- ✅ Transparent (constraint explicit)
- ⚠️ May reduce accuracy (1-5% typical)
- ⚠️ Requires λ tuning (hyperparameter search)
- ⚠️ Not all constraints compatible (impossibility theorems)

**λ Parameter Guidance**:
```
λ = 0.0:  No fairness constraint (baseline)
λ = 0.1:  Gentle nudge (2-5% fairness improvement)
λ = 0.5:  Balanced (20-40% fairness improvement, ~2% accuracy loss)
λ = 1.0:  Strong fairness (50-70% improvement, ~4% accuracy loss)
λ = 2.0+: Maximum fairness (70-85% improvement, ~5%+ accuracy loss)
```

Start with λ=0.5 and adjust based on validation results.

---

### Technique 2: Adversarial Debiasing

**Purpose**: Train model to make accurate predictions while being unable to infer protected attributes

**When to Use**:
- Neural network models (deep learning)
- Complex, non-linear bias patterns
- Want to prevent model from encoding protected attributes in representations

**How It Works**:
Two-network architecture competing against each other:

1. **Predictor Network**: Tries to maximize task accuracy
2. **Adversary Network**: Tries to predict protected attributes from predictor's hidden layers

Training process:
- Predictor learns: "Make accurate predictions"
- Adversary learns: "Guess protected attributes from predictor's internals"
- Predictor also learns: "Fool the adversary" (make it impossible to guess)

Result: Predictor becomes accurate but cannot encode protected attribute information.

**Architecture Example - Loan Approval**:
```
Input Features (income, credit_score, employment_history)
        ↓
Predictor Network:
    Layer 1: 64 neurons (hidden representations)
    Layer 2: 32 neurons
    Layer 3: 1 neuron → Loan Approval Prediction (maximize accuracy)
        ↓
    [Hidden Layer 1 extracted]
        ↓
Adversary Network:
    Layer 1: 32 neurons
    Layer 2: 16 neurons  
    Layer 3: 1 neuron → Gender Prediction (minimize this)

Training Loss:
Loss = Loss_task(approval_pred, actual) - λ * Loss_adversary(gender_pred, actual_gender)
```

The negative sign on adversary loss means: Predictor tries to make adversary FAIL.

**Implementation Approach**:

**Step 1: Design Networks**
- Predictor: Standard architecture for your task
- Adversary: Smaller network (1/2 to 1/3 predictor size)
- Connection: Adversary receives intermediate representations from predictor

**Step 2: Training Strategy**
Alternating optimization (like GANs):
1. Update Adversary: Given current predictor, improve gender prediction
2. Update Predictor: Improve task accuracy AND fool adversary
3. Repeat until convergence

**Step 3: Gradient Reversal (Critical Technique)**
Use gradient reversal layer between predictor and adversary:
- Forward pass: Adversary receives predictor's representations normally
- Backward pass: Gradients REVERSED (negative sign)
- Effect: Predictor learns to maximize adversary's error

**Step 4: Progressive Training**
Start with low λ, gradually increase:
```
Epoch 1-10:   λ = 0.1  (predictor focuses on accuracy)
Epoch 11-20:  λ = 0.3  (begin adversarial training)
Epoch 21-30:  λ = 0.5  (balanced)
Epoch 31+:    λ = 1.0  (strong debiasing)
```
This prevents training instability.

**Trade-offs**:
- ✅ Powerful for neural networks
- ✅ Handles non-linear bias patterns
- ✅ Prevents encoding of protected attributes
- ⚠️ Training instability (requires careful tuning)
- ⚠️ Longer training time (adversarial iterations)
- ⚠️ Black box (hard to interpret what changed)

**Stability Tips**:
1. Use gradient clipping (prevent exploding gradients)
2. Start with low learning rates (0.0001)
3. Monitor both networks' losses (detect mode collapse)
4. Use progressive λ schedule (not constant)
5. Validate frequently (early stopping if unstable)

---

### Technique 3: Fairness Regularization

**Purpose**: Add fairness penalty term to loss function (softer than hard constraints)

**When to Use**:
- Want smooth fairness-accuracy trade-off
- Don't need strict guarantees (soft fairness goals)
- Tree-based models (random forests, XGBoost)

**How It Works**:
Instead of hard constraints, add penalty that increases with unfairness:
```
Loss_total = Loss_task + λ * Fairness_Penalty

Where Fairness_Penalty could be:
- Demographic parity violation: |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
- Equal opportunity violation: |TPR_A0 - TPR_A1|
- Calibration error: Σ |P(Y=1|ŷ=p,A=0) - P(Y=1|ŷ=p,A=1)|
```

**Key Difference from Constraints**:
- Constraints: "Must satisfy fairness threshold" (hard requirement)
- Regularization: "Prefer fairer solutions" (soft guidance)

**Implementation Approach**:

**Step 1: Choose Fairness Penalty**
Common penalties:

**A. Demographic Parity Penalty**:
```
Penalty = (P(ŷ=1|A=0) - P(ŷ=1|A=1))²
```
Penalizes prediction rate differences

**B. Equal Opportunity Penalty**:
```
Penalty = (TPR_A0 - TPR_A1)²
```
Penalizes true positive rate differences

**C. Mutual Information Penalty**:
```
Penalty = I(ŷ; A)  (mutual information between predictions and protected attr)
```
Penalizes statistical dependence

**Step 2: Integrate into Training**
For gradient-based models:
- Compute fairness penalty on each mini-batch
- Add λ * Penalty to loss
- Backpropagate through combined loss

For tree-based models:
- Modify split criterion to include fairness penalty
- Each split evaluated on: Information Gain - λ * Fairness_Violation
- Trees grow to balance accuracy and fairness

**Step 3: Tune λ via Grid Search**
```
λ_values = [0.1, 0.3, 0.5, 0.7, 1.0]

for λ in λ_values:
    model = train_with_regularization(data, λ)
    fairness = evaluate_fairness(model)
    accuracy = evaluate_accuracy(model)
    
    results[λ] = (fairness, accuracy)

# Plot Pareto frontier
plot_tradeoff(results)

# Select λ based on stakeholder preference
optimal_λ = stakeholder_selects_from_plot(results)
```

**Step 4: Validate Sensitivity**
- Test nearby λ values
- Ensure results stable (not hypersensitive to λ)
- Document λ selection rationale

**Trade-offs**:
- ✅ Smooth trade-off (tunable via λ)
- ✅ Flexible (works with most algorithms)
- ✅ Less training instability than adversarial
- ⚠️ No hard guarantees (may not meet strict threshold)
- ⚠️ Requires careful λ tuning
- ⚠️ Penalty design impacts effectiveness

---

## 5.2 Multi-Objective Optimization Framework

All three techniques are instances of multi-objective optimization: balancing accuracy and fairness.

### Pareto Optimality Concept

**Pareto Frontier**: Set of solutions where improving one objective requires sacrificing another.

```
Accuracy
   ↑
   │     ○ (λ=0.1)
   │    ○ (λ=0.3)
   │   ○ (λ=0.5)  ← Pareto frontier
   │  ○ (λ=0.7)
   │ ○ (λ=1.0)
   │
   └─────────────────→ Fairness
```

**Key Insights**:
1. No single "best" solution (depends on priorities)
2. Trade-off is unavoidable (fairness costs accuracy)
3. Stakeholder values determine optimal point
4. Operating point should be on Pareto frontier (not dominated)

### Generating the Pareto Frontier

**Approach 1: λ Sweep**
```
Train models with λ ∈ [0, 0.1, 0.2, ..., 2.0]
Plot accuracy vs fairness
Connect points to visualize trade-off
```

**Approach 2: Multi-Objective Optimization Algorithms**
- NSGA-II (Non-dominated Sorting Genetic Algorithm)
- MOEA/D (Multi-Objective Evolutionary Algorithm)
- Directly search for Pareto-optimal solutions

**Approach 3: ε-Constraint Method**
```
For each fairness threshold ε:
    Maximize accuracy
    Subject to: Fairness ≤ ε
    
This generates Pareto frontier by varying ε
```

### Stakeholder Selection Process

**Step 1: Present Trade-off Visually**
```
Model Performance vs Fairness Trade-off

AUC
0.82 │     ● (No fairness)
0.80 │    ●
0.78 │   ● ← Recommended (λ=0.5)
0.76 │  ●
0.74 │ ●
     └─────────────────────────
      0.15  0.10  0.05  0.02  0
           Gender Gap (lower is better)
```

**Step 2: Discuss Business Constraints**
- Minimum acceptable accuracy (e.g., AUC > 0.75)
- Maximum acceptable fairness violation (e.g., gap < 0.05)
- Regulatory requirements
- Risk tolerance

**Step 3: Document Decision**
```
Fairness Operating Point Selection

Chosen λ: 0.5
Model Performance:
  - AUC: 0.78 (down from 0.82 baseline)
  - Accuracy: 80% (down from 82%)
  
Fairness Metrics:
  - Gender gap: 0.03 (down from 0.18)
  - Equal opportunity diff: 0.02 (down from 0.22)
  
Rationale:
  - Meets regulatory threshold (<0.05)
  - Acceptable accuracy loss (2%)
  - Balanced approach per stakeholder discussion 2024-11-05
  
Approved by:
  - ML Engineering Lead: [Signature]
  - Compliance Officer: [Signature]
  - Product Owner: [Signature]
```

---

## 5.3 Selection Decision Tree

```
START: In-processing intervention needed

┌────────────────────────────────────────────┐
│ DECISION 1: What is your model type?      │
└────────────────┬───────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
    LINEAR            NEURAL           TREE-BASED
   (Logistic,          (Deep          (Random Forest,
    SVM, GLM)         Learning)        XGBoost, GBDT)
        │                 │                 │
        ↓                 ↓                 ↓
  CONSTRAINED      ADVERSARIAL        REGULARIZATION
  OPTIMIZATION      DEBIASING         (Penalty Term)
        │                 │                 │
        │                 │                 │
        └────────┬────────┴─────────────────┘
                 ↓
┌────────────────────────────────────────────┐
│ DECISION 2: What fairness definition?     │
└────────────────┬───────────────────────────┘
                 ↓
        ┌────────┴────────────┐
        │                     │
   EQUAL TREATMENT      EQUAL OUTCOMES
    (Process Fair)      (Results Fair)
        │                     │
        ↓                     ↓
  EQUAL OPPORTUNITY    DEMOGRAPHIC PARITY
  (Among qualified,     (Overall rates
   equal approval)       equal)
        │                     │
        │                     │
  ┌─────┴──────────────┬──────┘
  │                    │
  ↓                    ↓
EQUALIZED ODDS    PREDICTIVE PARITY
(Both error       (Predicted = Actual
 rates equal)      for all groups)

┌────────────────────────────────────────────┐
│ DECISION 3: How strict?                   │
└────────────────┬───────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
  SOFT FAIRNESS      HARD FAIRNESS
  (Preference)       (Requirement)
        │                 │
        ↓                 ↓
  REGULARIZATION    CONSTRAINT
  (Tune λ)          (Set threshold)

END: Technique selected, implement with chosen parameters
```

### Quick Reference Table

| Model Type | Fairness Goal | Strictness | Recommended Technique | λ Range |
|------------|---------------|------------|----------------------|---------|
| Logistic Regression | Equal Opportunity | Hard | Constrained Optimization | N/A (threshold: <0.05) |
| Neural Network | Demographic Parity | Soft | Adversarial Debiasing | 0.5-1.0 |
| Random Forest | Equal Opportunity | Soft | Regularization | 0.3-0.7 |
| XGBoost | Equalized Odds | Hard | Constrained Optimization | N/A (threshold: <0.05) |
| SVM | Calibration | Soft | Regularization | 0.5-1.0 |

---

## 5.4 Implementation Process

### Step 1: Baseline Model (2 days)

**Objective**: Establish performance without fairness constraints

**Process**:
1. Train model normally (no fairness considerations)
2. Evaluate accuracy metrics (AUC, F1, precision, recall)
3. Evaluate fairness metrics (gaps by protected attributes)
4. Document baseline performance

**Deliverable**:
```
Baseline Model Performance Report

Model: XGBoost Classifier
Task: Loan Approval Prediction

Performance Metrics:
  - AUC: 0.82
  - Accuracy: 82%
  - Precision: 0.78
  - Recall: 0.76

Fairness Metrics:
  - Gender approval gap: 18% (male: 76%, female: 58%)
  - Equal opportunity diff: 0.22
  - Demographic parity: 0.18

This baseline will be compared against fair models.
```

### Step 2: Select Technique (0.5 days)

**Use decision tree above** to select:
1. Technique type (constraints, adversarial, regularization)
2. Fairness definition (equal opportunity, demographic parity, etc.)
3. Strictness level (hard threshold or soft penalty)

**Document rationale**:
```
Technique Selection Rationale

Selected: Constrained Optimization with Equal Opportunity

Reasoning:
1. Model type: XGBoost (tree-based) → Supports constraints well
2. Fairness goal: Among creditworthy, equal approval rates
   - Aligns with ECOA (equal access to credit)
   - Stakeholder preference: Don't want to approve unqualified
3. Strictness: Hard constraint (regulatory requirement <0.05)

Alternative considered: Regularization
Rejected because: Need strict guarantee for compliance
```

### Step 3: Implement Fair Training (5-7 days)

**For Constrained Optimization**:
```
Day 1-2: Setup
  - Install fairness libraries (Fairlearn, AIF360)
  - Wrap model with fairness constraint wrapper
  - Configure constraint type and threshold

Day 3-5: Training
  - Train with constraint (may take 2-3x longer)
  - Monitor: Loss (task) and constraint violation
  - Adjust if not converging (relax threshold, change optimizer)

Day 6-7: Hyperparameter Tuning
  - Grid search over model hyperparameters + constraint strength
  - Find optimal combination
  - Validate on holdout set
```

**For Adversarial Debiasing**:
```
Day 1-2: Architecture Design
  - Design predictor network
  - Design adversary network (1/2 to 1/3 predictor size)
  - Implement gradient reversal layer

Day 3-5: Training
  - Implement alternating optimization
  - Use progressive λ schedule (0.1 → 1.0)
  - Monitor both networks' losses (detect instability)

Day 6-7: Stability Tuning
  - Adjust learning rates if unstable
  - Tune gradient clipping
  - Validate on holdout set
```

**For Fairness Regularization**:
```
Day 1-2: Penalty Implementation
  - Implement fairness penalty function
  - Integrate into training loop
  - Test penalty computation correctness

Day 3-5: λ Grid Search
  - Train models with λ ∈ [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
  - Record fairness and accuracy for each
  - Plot Pareto frontier

Day 6-7: Stakeholder Selection
  - Present trade-off options
  - Stakeholders select operating point
  - Retrain with optimal λ
  - Final validation
```

### Step 4: Validation (2 days)

**Compare to Baseline**:

| Metric | Baseline | Fair Model | Change | Threshold | Status |
|--------|----------|------------|--------|-----------|--------|
| Gender Gap | 18% | 3% | -83% | <5% | ✅ Pass |
| Equal Opportunity Diff | 0.22 | 0.04 | -82% | <0.05 | ✅ Pass |
| AUC | 0.82 | 0.78 | -4.9% | >0.75 | ✅ Pass |
| Accuracy | 82% | 80% | -2% | >75% | ✅ Pass |

**Intersectional Analysis**:
Check fairness for subgroups (gender × age, gender × race):
- Young women (25-35): Gap reduced 18% → 4% ✅
- Older women (45+): Gap reduced 14% → 2% ✅
- Maximum subgroup gap: 4% (within 8% threshold) ✅

**Statistical Significance**:
- Chi-square test: p < 0.001 (highly significant improvement)
- Bootstrap 95% CI: [0.025, 0.045] (excludes zero, robust)

**Performance Impact**:
- AUC loss: 4.9% (within 5% acceptable threshold)
- Accuracy loss: 2% (minimal business impact)
- Precision/Recall: Balanced (no disproportionate impact)

**Deliverable**: Fair model validation report with stakeholder sign-off

---

## 5.5 Best Practices

### DO's ✅

1. **Always generate Pareto frontier**
   - Don't pick λ arbitrarily
   - Show stakeholders the trade-off space
   - Document why specific operating point chosen

2. **Start with baseline**
   - Measure improvement relative to unconstrained model
   - Quantify fairness-accuracy trade-off precisely

3. **Validate on diverse test sets**
   - Temporal holdout (future data)
   - Geographic holdout (different regions)
   - Demographic stratified sampling

4. **Monitor training convergence**
   - Constrained optimization may not converge
   - Adversarial training can be unstable
   - Early stopping if diverging

5. **Document everything**
   - Which fairness definition chosen and why
   - How λ or threshold was selected
   - Stakeholder approval of trade-off
   - Model card with fairness properties

### DON'Ts ❌

1. **Don't skip baseline**
   - Need to quantify improvement
   - Can't evaluate trade-off without baseline

2. **Don't use one-size-fits-all λ**
   - Optimal λ varies by dataset, model, task
   - Always tune via grid search

3. **Don't ignore impossibility results**
   - Some fairness criteria are mutually exclusive
   - Can't have perfect demographic parity AND perfect calibration (when base rates differ)
   - Choose primary criterion, accept imperfection on others

4. **Don't over-constrain**
   - If accuracy drops >10%, model may be unusable
   - Consider post-processing as fallback

5. **Don't forget intersectionality**
   - Constraints should apply to all subgroups
   - Check gender × race, not just gender and race separately

---

## 5.6 Integration with Other Components

### Inputs from Pre-Processing

**Pre-Processing Provides**:
- Debiased training dataset (proxies removed, reweighted)
- Expected remaining bias after data fixes
- Feature transformation parameters

**In-Processing Uses This To**:
- Train on cleaner data (better starting point)
- Set fairness constraint strength based on remaining bias
- Avoid double-correction (data already partially debiased)

**Example**:
```
Pre-processing reduced gap: 18% → 10% (44% improvement)
Remaining for in-processing: 10% gap

Set in-processing target: <5% (half of remaining)
This guides λ selection: Start with λ=0.5, adjust as needed
```

### Outputs to Post-Processing

**In-Processing Provides**:
- Fair model with embedded constraints
- Training metrics (fairness achieved, accuracy trade-off)
- Model card documenting fairness properties

**Post-Processing Uses This To**:
- Fine-tune if gaps remain (e.g., 5% → 2%)
- Apply calibration for production deployment
- Threshold optimization for final adjustment

**Example**:
```
In-processing achieved: 18% → 5% gap (72% improvement)
Post-processing target: 5% → 2% gap (additional fine-tuning)

Post-processing applies:
- Threshold optimization (5% → 3%)
- Calibration (3% → 2%)

Total improvement: 18% → 2% (89% gap reduction)
```

### Outputs to Validation

**In-Processing Provides**:
- Fair model ready for testing
- Baseline comparison metrics
- Chosen fairness operating point (λ, threshold)

**Validation Uses This To**:
- Confirm fairness metrics meet thresholds
- Verify performance acceptable
- Check intersectional fairness
- Statistical significance testing

---

## 5.7 Common Pitfalls & Solutions

### Pitfall 1: Model Won't Converge

**Symptom**: Loss oscillates, doesn't decrease, or training fails

**Causes**:
- Constraint too strict (infeasible)
- Learning rate too high
- Adversarial training unstable

**Solutions**:
- Relax fairness threshold slightly (ε: 0.05 → 0.07)
- Reduce learning rate (0.001 → 0.0001)
- For adversarial: Use progressive λ schedule
- Try different optimizer (Adam → SGD or vice versa)

### Pitfall 2: Accuracy Degrades >10%

**Symptom**: Fair model unusable due to poor performance

**Causes**:
- Fairness and accuracy fundamentally incompatible for this task/data
- λ too high (over-constraining)
- Legitimate predictive features removed by mistake

**Solutions**:
- Lower λ (try λ/2)
- Revisit causal analysis: Is mediator truly non-legitimate?
- Consider post-processing instead (less accuracy impact)
- Collect better data (more representative)

### Pitfall 3: Fairness Achieved on Train, Fails on Test

**Symptom**: Validation shows worse fairness than training

**Causes**:
- Overfitting to training demographics
- Test set demographics different from training
- Constraint applied incorrectly

**Solutions**:
- Use larger fairness penalty (increase λ)
- Regularize model more (prevent overfitting)
- Validate on temporal holdout (future data, not random split)
- Check if test demographics match deployment population

### Pitfall 4: Intersectional Fairness Violated

**Symptom**: Overall fairness good, but specific subgroups still have large gaps

**Causes**:
- Constraint only applied to main effects (gender, race separately)
- Small subgroup sample sizes ignored during training
- Interactions not modeled

**Solutions**:
- Extend constraint to all subgroups: |TPR_{white_male} - TPR_{black_female}| < ε
- Resample small subgroups (ensure sufficient training data)
- Use hierarchical fairness: Main effects + interactions

---

## 5.8 Deliverables Checklist

After completing in-processing, you should have:

- [x] **Baseline Model**: Performance without fairness constraints
- [x] **Pareto Frontier**: Trade-off curve (fairness vs accuracy)
- [x] **Fair Model**: Trained with chosen fairness technique
- [x] **Hyperparameters**: Documented λ or threshold selection
- [x] **Validation Report**: Fairness and performance metrics
- [x] **Stakeholder Approval**: Sign-off on chosen operating point
- [x] **Model Card**: Fairness properties, limitations, usage guidelines
- [x] **Training Logs**: Convergence curves, stability checks
- [x] **Integration Guide**: How to deploy and use the fair model

---

## Summary

In-processing embeds fairness directly into model training through constrained optimization, adversarial debiasing, or fairness regularization. By optimizing for both accuracy and fairness simultaneously, in-processing addresses root causes at the model level.

**Key Takeaways**:
1. **Three techniques**: Constraints (hard guarantees), Adversarial (neural nets), Regularization (soft)
2. **Multi-objective**: Always involves fairness-accuracy trade-off (Pareto frontier)
3. **Stakeholder involvement**: Operating point selection requires values judgment
4. **Validation critical**: Ensure fairness on test set, check intersectionality
5. **Integration**: Builds on pre-processing, refined by post-processing

**Typical Impact**: 60-85% gap reduction when retraining allowed, 2-5% accuracy trade-off

