# 06: Post-Processing Fairness Toolkit

## Executive Summary

**Purpose**: Adjust model outputs to achieve fairness WITHOUT retraining the model. The fastest intervention option for deployed production systems.

**When to Use**: 
- Model already deployed in production, cannot retrain (regulatory approval, business constraints)
- Need quick fairness fix (1-2 weeks vs 6+ months for retraining)
- Fine-tuning after pre-processing or in-processing interventions
- Third-party models where you cannot access training process

**Key Benefits**:
- ✅ **Speed**: 1-2 weeks implementation (vs 3-6 months for retraining)
- ✅ **No retraining required**: Works with any deployed model
- ✅ **Significant impact**: 40-70% gap reduction (single technique), 70-90% (combined)
- ✅ **Reversible**: Can adjust or remove without affecting base model
- ✅ **Cost-effective**: Lower implementation cost than full model rebuild

**Expected Outcomes**:
- Fairness improvement: 40-70% gap reduction (single technique)
- Combined techniques: 70-90% total gap reduction
- Performance impact: Minimal (1-3% accuracy loss typical)
- Timeline: 1-2 weeks to production deployment

---

## Table of Contents

1. [Four Core Techniques](#1-four-core-techniques)
2. [Technique 1: Threshold Optimization](#2-threshold-optimization)
3. [Technique 2: Calibration](#3-calibration)
4. [Technique 3: Prediction Transformation](#4-prediction-transformation)
5. [Technique 4: Rejection Classification](#5-rejection-classification)
6. [Combining Techniques](#6-combining-techniques)
7. [Implementation Process](#7-implementation-process)
8. [Best Practices](#8-best-practices)
9. [Common Pitfalls & Solutions](#9-common-pitfalls-solutions)
10. [Integration with Other Components](#10-integration-with-other-components)

---

## 1. Four Core Techniques

### Overview

Post-processing operates on model outputs (predictions, probabilities, scores) rather than changing the model itself. This makes it uniquely suited for deployed systems where retraining is infeasible.

```
┌─────────────────────────────────────────────────────────┐
│           POST-PROCESSING INTERVENTION FLOW             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Model Input (Features)                                │
│         ↓                                               │
│  [DEPLOYED MODEL] ← Cannot change this                 │
│         ↓                                               │
│  Raw Predictions (Scores, Probabilities)               │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │  POST-PROCESSING LAYER              │               │
│  │  (This is what we modify)           │               │
│  │                                     │               │
│  │  1. Threshold Optimization          │               │
│  │  2. Calibration                     │               │
│  │  3. Prediction Transformation       │               │
│  │  4. Rejection Classification        │               │
│  └─────────────────────────────────────┘               │
│         ↓                                               │
│  Fair Predictions (Final Decisions)                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Technique Comparison

| Technique | Primary Use | Complexity | Implementation Time | Expected Impact | Requires Protected Attrs? |
|-----------|-------------|------------|---------------------|-----------------|---------------------------|
| **Threshold Optimization** | Binary decisions, different group score distributions | Low | 3-5 days | 40-60% | Yes (at inference) |
| **Calibration** | Probability scores mean different things for groups | Medium | 3-5 days | 20-30% additional | Yes (at inference) |
| **Prediction Transformation** | Complex fairness criteria, ranking systems | High | 5-7 days | Varies | Optional |
| **Rejection Classification** | High-stakes decisions, human review available | Medium | 3-5 days | 10-20% additional | Yes (at inference) |

### Selection Decision Tree

```
START: Post-processing needed for deployed model

┌────────────────────────────────────────────────────────┐
│ QUESTION 1: What is your primary fairness concern?    │
└────────────────┬───────────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
   EQUAL RATES      RELIABLE SCORES
   (Approval,       (Predictions mean
   Acceptance)       same thing)
        │                 │
        ↓                 ↓
   THRESHOLD         CALIBRATION
   OPTIMIZATION      (+ Threshold)
        │                 │
        └────────┬────────┘
                 ↓
┌────────────────────────────────────────────────────────┐
│ QUESTION 2: Do you have protected attributes at        │
│ inference time?                                        │
└────────────────┬───────────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        ↓                 ↓
   ALL TECHNIQUES    PREDICTION
   AVAILABLE         TRANSFORMATION
                     (doesn't require
                      explicit groups)
                     
┌────────────────────────────────────────────────────────┐
│ QUESTION 3: Do you have human review capacity?        │
└────────────────┬───────────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        ↓                 ↓
   ADD REJECTION      THRESHOLD +
   CLASSIFICATION     CALIBRATION
   (for borderline    ONLY
   cases)

END: Technique(s) selected
```

---

## 2. Threshold Optimization

### Concept

**Problem**: Using a single decision threshold for all demographic groups can perpetuate or amplify bias when groups have different score distributions.

**Solution**: Use group-specific thresholds that equalize error rates or approval rates across demographic groups.

### How It Works

**Standard Approach** (Single Threshold):
```
For all applicants:
  If model_score ≥ 0.5 → Approve
  If model_score < 0.5 → Deny

Problem: If female scores shifted lower due to training bias,
         more qualified women fall below threshold
```

**Threshold Optimization** (Group-Specific):
```
For male applicants:
  If model_score ≥ 0.52 → Approve  (slightly higher threshold)
  If model_score < 0.52 → Deny

For female applicants:
  If model_score ≥ 0.43 → Approve  (lower threshold compensates)
  If model_score < 0.43 → Deny

Result: Equalizes true positive rates (equal opportunity)
        or approval rates (demographic parity)
```

### Why This Works

**Score Distribution Analysis Example**:

```
Male Score Distribution (well-calibrated):
    ╭─────────╮
    │         │
    │         │
────┴─────────┴────────────────────> Score
  0.3      0.5       0.7
        ↑ Single threshold (0.5)

Female Score Distribution (shifted lower due to bias):
        ╭─────────╮
        │         │
        │         │
────────┴─────────┴────────────────> Score
      0.2      0.4       0.6
            ↑ Should use 0.43 instead

Observation: Female scores systematically 0.07-0.10 points lower
Cause: Training data bias, feature proxies, historical discrimination
Solution: Lower threshold for females to compensate
```

### Three Fairness Objectives

#### Objective 1: Equal Opportunity (Most Common in Banking)

**Definition**: Among qualified individuals, equal approval rates across groups

**Mathematical Formulation**:
```
Minimize: Overall error rate
Subject to: |TPR_male - TPR_female| < ε

Where:
  TPR = True Positive Rate = P(approve | qualified)
  ε = tolerance threshold (typically 0.05)
```

**When to Use**:
- Lending (ECOA compliance: among creditworthy, equal access)
- Hiring (among qualified candidates, equal interview rates)
- Healthcare (among medically appropriate, equal treatment access)

**Implementation**:
```python
from fairlearn.postprocessing import ThresholdOptimizer

# Initialize optimizer
optimizer = ThresholdOptimizer(
    estimator=trained_model,
    constraints="true_positive_rate_parity",  # Equal opportunity
    objective="accuracy_score"
)

# Fit on validation set to find optimal thresholds
optimizer.fit(
    X_validation, 
    y_validation, 
    sensitive_features=gender_validation
)

# Apply to test set
predictions_fair = optimizer.predict(
    X_test,
    sensitive_features=gender_test
)

# Get thresholds used
thresholds = optimizer.interpolated_thresholder_.interpolation_dict
print(f"Male threshold: {thresholds['male']}")
print(f"Female threshold: {thresholds['female']}")
```

**Example Results**:
```
Before (single threshold 0.50):
  Male: 76% approval, 85% TPR (among qualified)
  Female: 58% approval, 63% TPR (among qualified)
  Gap: 18% approval, 22% TPR ❌

After (optimized thresholds: male=0.52, female=0.43):
  Male: 72% approval, 83% TPR
  Female: 66% approval, 82% TPR
  Gap: 6% approval, 1% TPR ✅
```

#### Objective 2: Demographic Parity

**Definition**: Equal positive prediction rates across groups (regardless of qualification)

**Mathematical Formulation**:
```
Minimize: Overall error rate
Subject to: |P(approve|male) - P(approve|female)| < ε
```

**When to Use**:
- Marketing (equal distribution of promotional offers)
- Public resource allocation (equal access to programs)
- Situations where equal outcomes desired regardless of "merit"

**Implementation**:
```python
optimizer = ThresholdOptimizer(
    estimator=trained_model,
    constraints="demographic_parity",
    objective="accuracy_score"
)

# Rest identical to equal opportunity example
```

**Trade-off Warning**: May approve unqualified individuals to close gap. Use carefully.

#### Objective 3: Equalized Odds

**Definition**: Both true positive rate AND false positive rate equal across groups

**Mathematical Formulation**:
```
Minimize: Overall error rate
Subject to: |TPR_male - TPR_female| < ε AND
            |FPR_male - FPR_female| < ε
```

**When to Use**:
- High-stakes binary decisions (both error types matter)
- Criminal justice (false positives = innocent imprisoned, false negatives = criminals released)
- Medical diagnosis (false positives = unnecessary treatment, false negatives = missed disease)

**Implementation**:
```python
optimizer = ThresholdOptimizer(
    estimator=trained_model,
    constraints="equalized_odds",
    objective="accuracy_score"
)
```

**Note**: More stringent than equal opportunity, may reduce overall accuracy more.

### Step-by-Step Implementation

#### Step 1: Analyze Score Distributions (1 day)

**Goal**: Understand how scores differ by demographic group

**Process**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load validation data with scores
data = pd.read_csv('validation_with_scores.csv')

# Calculate statistics by group
male_scores = data[data['gender'] == 'male']['model_score']
female_scores = data[data['gender'] == 'female']['model_score']

print("Male scores:")
print(f"  Mean: {male_scores.mean():.3f}")
print(f"  Median: {male_scores.median():.3f}")
print(f"  Std: {male_scores.std():.3f}")

print("Female scores:")
print(f"  Mean: {female_scores.mean():.3f}")
print(f"  Median: {female_scores.median():.3f}")
print(f"  Std: {female_scores.std():.3f}")

# Visualize distributions
plt.figure(figsize=(10, 6))
plt.hist(male_scores, bins=50, alpha=0.5, label='Male', density=True)
plt.hist(female_scores, bins=50, alpha=0.5, label='Female', density=True)
plt.axvline(0.5, color='red', linestyle='--', label='Current threshold')
plt.xlabel('Model Score')
plt.ylabel('Density')
plt.legend()
plt.title('Score Distributions by Gender')
plt.savefig('score_distributions.png')
```

**Key Questions**:
1. Are distributions shifted? (Different means/medians)
2. Are distributions overlapping or separate?
3. Where is current threshold relative to each distribution?

**Example Output**:
```
Male scores:
  Mean: 0.580
  Median: 0.575
  Std: 0.180

Female scores:
  Mean: 0.510  ← 0.07 points lower
  Median: 0.490
  Std: 0.170

Interpretation: Female scores systematically lower
Action: Need lower threshold for females to compensate
```

#### Step 2: Grid Search for Optimal Thresholds (1-2 days)

**Goal**: Find threshold pair that maximizes accuracy while satisfying fairness constraint

**Process**:
```python
import numpy as np
from sklearn.metrics import accuracy_score

def calculate_tpr(y_true, y_pred):
    """True Positive Rate among actual positives"""
    positives = y_true == 1
    return y_pred[positives].mean()

def grid_search_thresholds(scores_male, scores_female, 
                           labels_male, labels_female,
                           fairness_tolerance=0.05):
    """
    Grid search over threshold pairs to find optimal.
    """
    # Define search space
    threshold_range = np.arange(0.30, 0.70, 0.01)
    
    best_accuracy = 0
    best_thresholds = None
    results = []
    
    for t_male in threshold_range:
        for t_female in threshold_range:
            # Apply thresholds
            pred_male = (scores_male >= t_male).astype(int)
            pred_female = (scores_female >= t_female).astype(int)
            
            # Calculate TPR (equal opportunity)
            tpr_male = calculate_tpr(labels_male, pred_male)
            tpr_female = calculate_tpr(labels_female, pred_female)
            tpr_diff = abs(tpr_male - tpr_female)
            
            # Calculate overall accuracy
            accuracy = accuracy_score(
                np.concatenate([labels_male, labels_female]),
                np.concatenate([pred_male, pred_female])
            )
            
            # Record result
            results.append({
                't_male': t_male,
                't_female': t_female,
                'tpr_male': tpr_male,
                'tpr_female': tpr_female,
                'tpr_diff': tpr_diff,
                'accuracy': accuracy,
                'fairness_met': tpr_diff < fairness_tolerance
            })
            
            # Update best if fairness met and accuracy higher
            if tpr_diff < fairness_tolerance and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = (t_male, t_female)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    return best_thresholds, best_accuracy, results_df

# Run grid search
best_t, best_acc, all_results = grid_search_thresholds(
    scores_male=validation_scores[gender == 'male'],
    scores_female=validation_scores[gender == 'female'],
    labels_male=validation_labels[gender == 'male'],
    labels_female=validation_labels[gender == 'female'],
    fairness_tolerance=0.05
)

print(f"Optimal thresholds found:")
print(f"  Male: {best_t[0]:.3f}")
print(f"  Female: {best_t[1]:.3f}")
print(f"  Accuracy: {best_acc:.3f}")

# Analyze feasible region
feasible = all_results[all_results['fairness_met']]
print(f"\nFeasible solutions: {len(feasible)} out of {len(all_results)}")
```

**Visualization**:
```python
# Plot Pareto frontier (fairness vs accuracy trade-off)
plt.figure(figsize=(10, 6))
plt.scatter(all_results['tpr_diff'], all_results['accuracy'], 
            c=all_results['fairness_met'], cmap='RdYlGn', alpha=0.6)
plt.axvline(0.05, color='red', linestyle='--', label='Fairness threshold')
plt.xlabel('TPR Difference (lower is better)')
plt.ylabel('Accuracy (higher is better)')
plt.title('Fairness-Accuracy Trade-off')
plt.colorbar(label='Fairness Met')
plt.legend()
plt.savefig('pareto_frontier.png')
```

#### Step 3: Validate on Test Set (1 day)

**Goal**: Confirm thresholds work on held-out data (not overfit to validation)

**Process**:
```python
# Apply optimal thresholds to test set
t_male, t_female = best_thresholds

pred_male_test = (test_scores[test_gender == 'male'] >= t_male).astype(int)
pred_female_test = (test_scores[test_gender == 'female'] >= t_female).astype(int)

# Calculate metrics
from sklearn.metrics import confusion_matrix, classification_report

def calculate_fairness_metrics(y_true, y_pred, group):
    """Calculate comprehensive metrics for a group"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'group': group,
        'approval_rate': y_pred.mean(),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    }

metrics_male = calculate_fairness_metrics(
    test_labels[test_gender == 'male'],
    pred_male_test,
    'male'
)

metrics_female = calculate_fairness_metrics(
    test_labels[test_gender == 'female'],
    pred_female_test,
    'female'
)

# Create comparison table
comparison = pd.DataFrame([metrics_male, metrics_female])
comparison['tpr_diff'] = abs(metrics_male['tpr'] - metrics_female['tpr'])
comparison['approval_gap'] = abs(metrics_male['approval_rate'] - 
                                   metrics_female['approval_rate'])

print("\nTest Set Performance:")
print(comparison.to_string(index=False))

# Check if thresholds met targets
assert comparison['tpr_diff'].iloc[0] < 0.05, "Equal opportunity not achieved"
assert comparison['approval_gap'].iloc[0] < 0.10, "Approval gap too large"

print("\n✅ Validation successful: Thresholds meet fairness criteria")
```

**Expected Output**:
```
Test Set Performance:
group  approval_rate  accuracy   tpr    fpr  precision    f1  tpr_diff  approval_gap
male         0.72      0.80    0.83   0.15      0.78    0.80     0.01         0.06
female       0.66      0.79    0.82   0.14      0.76    0.79

✅ Validation successful: Thresholds meet fairness criteria
```

#### Step 4: Deploy to Production (1-2 days)

**Goal**: Integrate threshold logic into production prediction service

**Implementation**:
```python
# production/models/prediction_service.py

class FairLoanDecisionService:
    """
    Production service applying group-specific thresholds.
    """
    
    def __init__(self, model, thresholds_config):
        self.model = model
        self.thresholds = thresholds_config
        
    def predict(self, features, protected_attributes):
        """
        Make fair prediction using group-specific thresholds.
        
        Args:
            features: dict of applicant features
            protected_attributes: dict with 'gender', 'age', etc.
            
        Returns:
            dict with decision, score, threshold_used
        """
        # Get model score
        score = self.model.predict_proba(features)[1]
        
        # Select appropriate threshold
        gender = protected_attributes.get('gender', 'unknown')
        threshold = self.thresholds.get(gender, self.thresholds['default'])
        
        # Make decision
        decision = 'APPROVED' if score >= threshold else 'DENIED'
        
        # Log for audit trail
        self._log_decision(features, protected_attributes, 
                          score, threshold, decision)
        
        return {
            'decision': decision,
            'score': float(score),
            'threshold_used': float(threshold),
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_decision(self, features, protected_attrs, 
                      score, threshold, decision):
        """Log decision for monitoring and audit"""
        log_entry = {
            'applicant_id': features.get('id'),
            'score': float(score),
            'threshold': float(threshold),
            'gender': protected_attrs.get('gender'),
            'decision': decision,
            'timestamp': datetime.now()
        }
        
        # Write to audit log (database or file)
        audit_logger.info(json.dumps(log_entry))

# Configuration
thresholds_config = {
    'male': 0.52,
    'female': 0.43,
    'non_binary': 0.48,
    'unknown': 0.50,  # Default for missing gender
    'default': 0.50
}

# Initialize service
decision_service = FairLoanDecisionService(
    model=trained_model,
    thresholds_config=thresholds_config
)

# Example usage
applicant_features = {
    'id': 'APP-47821',
    'income': 45000,
    'credit_score': 680,
    'employment_years': 3
}

protected_attrs = {
    'gender': 'female',
    'age': 28
}

result = decision_service.predict(applicant_features, protected_attrs)
print(result)
# Output: {'decision': 'APPROVED', 'score': 0.52, 'threshold_used': 0.43, ...}
```

**Testing Strategy**:
```python
# Integration tests
def test_threshold_application():
    """Test that correct thresholds are applied"""
    service = FairLoanDecisionService(model, thresholds_config)
    
    # Test male applicant
    result_male = service.predict(features, {'gender': 'male'})
    assert result_male['threshold_used'] == 0.52
    
    # Test female applicant
    result_female = service.predict(features, {'gender': 'female'})
    assert result_female['threshold_used'] == 0.43
    
    # Test edge case: score exactly at threshold
    features_edge = {...}  # Engineered to produce score = 0.43
    result_edge = service.predict(features_edge, {'gender': 'female'})
    assert result_edge['decision'] == 'APPROVED'  # >= threshold
    
    print("✅ All threshold tests passed")

# Canary deployment
def canary_test():
    """Route 10% of traffic to new service, monitor for issues"""
    for request in incoming_requests:
        if random.random() < 0.10:  # 10% canary
            result = new_decision_service.predict(request)
        else:
            result = old_decision_service.predict(request)
        
        # Log both for comparison
        compare_results(old_result, new_result)

# Run tests
test_threshold_application()
canary_test()  # Run for 24-48 hours before full rollout
```

### Performance Impact & Trade-offs

**Typical Results**:
```
Metric                  Before    After     Change
─────────────────────────────────────────────────────
Gender Approval Gap      18%      6%       -67% ✅
Equal Opportunity Diff   0.22     0.04     -82% ✅
Overall Approval Rate    68%      67%      -1pp ✅
Model AUC                0.78     0.78     0% ✅ (no change)
Accuracy                 82%      81%      -1% ✅
```

**Key Insights**:
- **AUC unchanged**: Same model, just different decision boundary
- **Accuracy slight decrease**: Trade-off for fairness (1-2% typical)
- **Approval rate stable**: Overall business impact minimal
- **Gap reduction significant**: 40-60% improvement common

**Trade-offs to Consider**:

1. **Perception of "Different Standards"**
   - Concern: "We're lowering the bar for women"
   - Reality: Correcting for systematic over-prediction of women's risk
   - Communication: "Same default rates across groups after adjustment"

2. **Requires Protected Attributes at Inference**
   - Need: Gender must be available when making predictions
   - Challenge: Privacy concerns, data availability
   - Solution: Ensure data pipeline includes protected attributes

3. **May Not Address Root Causes**
   - Limitation: Treats symptoms (biased scores) not disease (biased training)
   - Plan: Use pre/in-processing in next model version
   - Benefit: But provides immediate improvement while planning long-term fix

### When Threshold Optimization Works Best

**Ideal Scenarios**:
✅ Binary classification (approve/deny, hire/not hire)
✅ Well-calibrated model (scores have consistent meaning)
✅ Clear score distribution differences by group
✅ Protected attributes available at inference time
✅ Fairness definition aligns with error rate equalization

**Less Suitable When**:
❌ Multi-class classification (more than 2 outcomes)
❌ Regression (continuous predictions)
❌ Ranking systems (order matters more than binary decision)
❌ Protected attributes unavailable at inference
❌ Fairness requires beyond error rate parity

---

## 3. Calibration

### Concept

**Problem**: Model probability scores mean different things for different demographic groups, creating unfair risk assessments.

**Solution**: Apply group-specific calibration transformations to ensure scores have consistent meaning across all groups.

### The Calibration Problem

**Example - Loan Default Prediction**:

```
BEFORE Calibration (Miscalibrated):

Model says: "This male applicant has 70% default probability"
Reality: Among males with score 0.70, 85% actually default
Interpretation: Model UNDER-estimates risk for males

Model says: "This female applicant has 70% default probability"  
Reality: Among females with score 0.70, 55% actually default
Interpretation: Model OVER-estimates risk for females

Problem: Same score (0.70) means different things!
Impact: Women denied at higher rates due to inflated risk estimates
```

**AFTER Calibration (Well-Calibrated)**:

```
Model says: "70% default probability" (male)
Reality: 70% actually default ✓

Model says: "70% default probability" (female)
Reality: 70% actually default ✓

Result: Scores now have consistent, reliable meaning
Impact: Fair risk assessment across all groups
```

### Why Miscalibration Occurs

**Root Causes**:

1. **Training Data Bias**:
   - Historical discrimination in labels
   - Selection bias (underrepresented groups)
   - Label quality varies by group

2. **Feature Proxies**:
   - Features correlate differently with outcome by group
   - Example: Employment gaps predict default differently for men vs women

3. **Model Complexity**:
   - Complex models (neural networks, ensembles) often miscalibrated
   - Regularization affects calibration
   - Class imbalance impacts calibration

### Measuring Calibration

#### Expected Calibration Error (ECE)

**Definition**: Average difference between predicted probabilities and actual frequencies

**Calculation Process**:

```python
def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """
    Calculate Expected Calibration Error.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins (typically 10)
        
    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[:-1]) - 1
    
    ece = 0
    for i in range(n_bins):
        # Get samples in this bin
        in_bin = bin_indices == i
        
        if in_bin.sum() == 0:
            continue
        
        # Calculate average predicted probability in bin
        avg_predicted = y_pred_proba[in_bin].mean()
        
        # Calculate actual frequency in bin
        avg_actual = y_true[in_bin].mean()
        
        # Weight by proportion of samples in bin
        weight = in_bin.sum() / len(y_true)
        
        # Add to ECE
        ece += weight * abs(avg_predicted - avg_actual)
    
    return ece

# Calculate by group
ece_male = calculate_ece(
    y_true[gender == 'male'],
    y_pred_proba[gender == 'male']
)

ece_female = calculate_ece(
    y_true[gender == 'female'],
    y_pred_proba[gender == 'female']
)

print(f"Male ECE: {ece_male:.3f}")
print(f"Female ECE: {ece_female:.3f}")

# Interpretation:
# ECE < 0.05: Well calibrated ✅
# ECE 0.05-0.10: Moderate miscalibration ⚠️
# ECE > 0.10: Poor calibration ❌
```

**Example Output**:
```
Male ECE: 0.08 (moderate miscalibration)
Female ECE: 0.12 (poor calibration - worse than males)

Action: Calibration needed, prioritize females
```

#### Calibration Curve Visualization

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curves(y_true, y_pred_proba, gender, n_bins=10):
    """Visualize calibration for each group"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot calibration for each group
    for group in ['male', 'female']:
        mask = gender == group
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true[mask],
            y_pred_proba[mask],
            n_bins=n_bins
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', label=f'{group.capitalize()}')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Actual)')
    ax.set_title('Calibration Curves by Gender')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_curves_before.png')
    
plot_calibration_curves(y_true, y_pred_proba, gender)
```

**Interpretation of Calibration Curves**:

```
Perfect Calibration: Points fall on diagonal line
Above diagonal: Model UNDER-predicts (too optimistic)
Below diagonal: Model OVER-predicts (too pessimistic)

Example:
Point at (0.7, 0.85): Model says 70%, but 85% actually default
                     → Under-estimating risk

Point at (0.7, 0.55): Model says 70%, but only 55% default
                     → Over-estimating risk (denying qualified people)
```

### Two Calibration Methods

#### Method 1: Platt Scaling (Parametric)

**Concept**: Fit a logistic regression to transform model scores

**When to Use**:
- Most situations (default choice)
- Stable, interpretable
- Works well with moderate sample sizes (>100 per group)

**How It Works**:
```
For each group, fit:
  calibrated_score = sigmoid(a * raw_score + b)
  
Where a and b are fitted to minimize calibration error
```

**Implementation**:
```python
from sklearn.calibration import CalibratedClassifierCV

def apply_platt_scaling(model, X_val, y_val, gender_val):
    """
    Apply Platt scaling separately by group.
    
    Returns: Dictionary of calibrated models by group
    """
    calibrators = {}
    
    for group in ['male', 'female']:
        # Get samples for this group
        mask = gender_val == group
        X_group = X_val[mask]
        y_group = y_val[mask]
        
        # Fit Platt scaling
        calibrator = CalibratedClassifierCV(
            model, 
            method='sigmoid',  # Platt scaling
            cv='prefit'  # Model already trained
        )
        
        calibrator.fit(X_group, y_group)
        
        calibrators[group] = calibrator
        
        print(f"{group.capitalize()} calibrator fitted on {len(y_group)} samples")
    
    return calibrators

# Fit calibrators
calibrators = apply_platt_scaling(
    model=trained_model,
    X_val=X_validation,
    y_val=y_validation,
    gender_val=gender_validation
)

# Apply calibration
def calibrated_predict(X, gender, calibrators):
    """Apply appropriate calibrator based on gender"""
    predictions = np.zeros(len(X))
    
    for group in ['male', 'female']:
        mask = gender == group
        if mask.sum() > 0:
            predictions[mask] = calibrators[group].predict_proba(X[mask])[:, 1]
    
    return predictions

# Get calibrated predictions
calibrated_scores = calibrated_predict(X_test, gender_test, calibrators)
```

**Validation**:
```python
# Calculate ECE before and after
ece_before_male = calculate_ece(y_test[gender_test == 'male'], 
                                 raw_scores[gender_test == 'male'])
ece_after_male = calculate_ece(y_test[gender_test == 'male'],
                                calibrated_scores[gender_test == 'male'])

ece_before_female = calculate_ece(y_test[gender_test == 'female'],
                                   raw_scores[gender_test == 'female'])
ece_after_female = calculate_ece(y_test[gender_test == 'female'],
                                  calibrated_scores[gender_test == 'female'])

print("Calibration Improvement:")
print(f"Male ECE: {ece_before_male:.3f} → {ece_after_male:.3f} "
      f"({(1 - ece_after_male/ece_before_male)*100:.1f}% improvement)")
print(f"Female ECE: {ece_before_female:.3f} → {ece_after_female:.3f} "
      f"({(1 - ece_after_female/ece_before_female)*100:.1f}% improvement)")
```

**Expected Results**:
```
Calibration Improvement:
Male ECE: 0.080 → 0.030 (62.5% improvement) ✅
Female ECE: 0.120 → 0.030 (75.0% improvement) ✅

Both groups now equally well-calibrated (ECE ≈ 0.03)
```

#### Method 2: Isotonic Regression (Non-Parametric)

**Concept**: Fit a piecewise-constant function that preserves score ordering

**When to Use**:
- Complex miscalibration patterns (non-monotonic)
- Large sample sizes (>1000 per group)
- Want maximum flexibility

**Advantages**:
- No assumptions about calibration function shape
- Can handle any calibration pattern
- Guaranteed to improve calibration

**Disadvantages**:
- Requires more data (prone to overfitting)
- Less interpretable
- Can overfit to noise

**Implementation**:
```python
from sklearn.isotonic import IsotonicRegression

def apply_isotonic_calibration(model, X_val, y_val, gender_val):
    """
    Apply isotonic regression calibration.
    """
    calibrators = {}
    
    for group in ['male', 'female']:
        mask = gender_val == group
        
        # Get raw scores
        raw_scores = model.predict_proba(X_val[mask])[:, 1]
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(raw_scores, y_val[mask])
        
        calibrators[group] = iso_reg
    
    return calibrators

# Usage similar to Platt scaling
calibrators_iso = apply_isotonic_calibration(
    trained_model, X_validation, y_validation, gender_validation
)
```

**When to Choose Which**:
```
Sample Size per Group:
  < 500:    Platt Scaling (more stable)
  500-1000: Platt Scaling (default)
  > 1000:   Either (Isotonic if complex patterns)

Calibration Pattern:
  Monotonic:     Platt Scaling
  Non-monotonic: Isotonic Regression
  
Interpretability:
  Important:     Platt Scaling (a and b parameters interpretable)
  Not important: Isotonic Regression
```

### Step-by-Step Implementation

#### Step 1: Assess Current Calibration (1 day)

```python
def comprehensive_calibration_assessment(y_true, y_pred_proba, gender):
    """
    Complete calibration analysis.
    
    Returns: Dictionary with metrics and visualizations
    """
    results = {}
    
    # Calculate ECE by group
    for group in ['male', 'female']:
        mask = gender == group
        ece = calculate_ece(y_true[mask], y_pred_proba[mask])
        results[f'ece_{group}'] = ece
    
    # Bin-wise analysis
    for group in ['male', 'female']:
        mask = gender == group
        
        bins_analysis = []
        for i in range(10):
            lower = i / 10
            upper = (i + 1) / 10
            
            in_bin = (y_pred_proba[mask] >= lower) & (y_pred_proba[mask] < upper)
            
            if in_bin.sum() > 0:
                avg_predicted = y_pred_proba[mask][in_bin].mean()
                avg_actual = y_true[mask][in_bin].mean()
                n_samples = in_bin.sum()
                error = abs(avg_predicted - avg_actual)
                
                bins_analysis.append({
                    'bin': f'{lower:.1f}-{upper:.1f}',
                    'avg_predicted': avg_predicted,
                    'avg_actual': avg_actual,
                    'error': error,
                    'n_samples': n_samples
                })
        
        results[f'bins_{group}'] = pd.DataFrame(bins_analysis)
    
    # Plot calibration curves
    plot_calibration_curves(y_true, y_pred_proba, gender)
    
    return results

# Run assessment
assessment = comprehensive_calibration_assessment(
    y_validation, validation_scores, gender_validation
)

# Print detailed results
print("\nMale Calibration Analysis:")
print(assessment['bins_male'].to_string(index=False))
print(f"\nMale ECE: {assessment['ece_male']:.4f}")

print("\n" + "="*60)
print("\nFemale Calibration Analysis:")
print(assessment['bins_female'].to_string(index=False))
print(f"\nFemale ECE: {assessment['ece_female']:.4f}")
```

**Example Output**:
```
Male Calibration Analysis:
bin      avg_predicted  avg_actual  error  n_samples
0.3-0.4      0.35          0.42     0.07      287
0.4-0.5      0.45          0.48     0.03      412
0.5-0.6      0.55          0.58     0.03      523
0.6-0.7      0.65          0.72     0.07      418  ← Miscalibrated
0.7-0.8      0.75          0.81     0.06      324
0.8-0.9      0.85          0.87     0.02      156

Male ECE: 0.0800 (needs improvement)

==============================================================

Female Calibration Analysis:
bin      avg_predicted  avg_actual  error  n_samples
0.3-0.4      0.35          0.29     0.06      412  ← Over-predicts
0.4-0.5      0.45          0.38     0.07      526
0.5-0.6      0.55          0.49     0.06      487
0.6-0.7      0.65          0.58     0.07      382  ← Systematic bias
0.7-0.8      0.75          0.69     0.06      298
0.8-0.9      0.85          0.82     0.03      124

Female ECE: 0.1200 (poor calibration - priority fix)

Observation: Model systematically over-predicts risk for women
Impact: Qualified women denied due to inflated risk estimates
```

#### Step 2: Fit Calibration Models (1-2 days)

```python
# Choose method based on sample size and pattern
def select_calibration_method(n_samples):
    """Choose calibration method based on data availability"""
    if n_samples < 500:
        return 'sigmoid'  # Platt scaling
    else:
        return 'isotonic'  # Can use either, isotonic more flexible

# Fit calibration for each group
from sklearn.calibration import CalibratedClassifierCV

calibrated_models = {}

for group in ['male', 'female']:
    mask = gender_validation == group
    n_samples = mask.sum()
    
    # Select method
    method = select_calibration_method(n_samples)
    print(f"\nCalibrating {group} (n={n_samples}) using {method}")
    
    # Fit calibration
    calibrated_model = CalibratedClassifierCV(
        base_estimator=trained_model,
        method=method,
        cv='prefit'  # Model already trained
    )
    
    calibrated_model.fit(
        X_validation[mask],
        y_validation[mask]
    )
    
    calibrated_models[group] = calibrated_model
    
    # Validate improvement
    raw_scores = trained_model.predict_proba(X_validation[mask])[:, 1]
    calibrated_scores = calibrated_model.predict_proba(X_validation[mask])[:, 1]
    
    ece_before = calculate_ece(y_validation[mask], raw_scores)
    ece_after = calculate_ece(y_validation[mask], calibrated_scores)
    
    improvement = (1 - ece_after / ece_before) * 100
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f} ({improvement:.1f}% improvement)")

# Save calibrated models
import joblib
for group, model in calibrated_models.items():
    joblib.dump(model, f'models/calibrator_{group}.pkl')
    print(f"Saved calibrator for {group}")
```

#### Step 3: Combine with Threshold Optimization (1 day)

**Rationale**: Calibration + Thresholds work synergistically

```python
class FairPredictionPipeline:
    """
    Complete post-processing pipeline:
    1. Calibration (fix score meanings)
    2. Threshold optimization (equalize error rates)
    """
    
    def __init__(self, base_model, calibrators, thresholds):
        self.base_model = base_model
        self.calibrators = calibrators
        self.thresholds = thresholds
    
    def predict_proba(self, X, gender):
        """Get calibrated probability scores"""
        scores = np.zeros(len(X))
        
        for group in self.calibrators.keys():
            mask = gender == group
            if mask.sum() > 0:
                # Apply group-specific calibration
                scores[mask] = self.calibrators[group].predict_proba(X[mask])[:, 1]
        
        return scores
    
    def predict(self, X, gender):
        """Get binary predictions using group-specific thresholds"""
        # Step 1: Get calibrated scores
        calibrated_scores = self.predict_proba(X, gender)
        
        # Step 2: Apply group-specific thresholds
        predictions = np.zeros(len(X), dtype=int)
        
        for group in self.thresholds.keys():
            mask = gender == group
            if mask.sum() > 0:
                threshold = self.thresholds[group]
                predictions[mask] = (calibrated_scores[mask] >= threshold).astype(int)
        
        return predictions, calibrated_scores
    
    def evaluate(self, X, y_true, gender):
        """Comprehensive evaluation"""
        predictions, scores = self.predict(X, gender)
        
        results = {}
        for group in ['male', 'female']:
            mask = gender == group
            
            # Calibration quality
            ece = calculate_ece(y_true[mask], scores[mask])
            
            # Fairness metrics
            tpr = ((predictions[mask] == 1) & (y_true[mask] == 1)).sum() / (y_true[mask] == 1).sum()
            fpr = ((predictions[mask] == 1) & (y_true[mask] == 0)).sum() / (y_true[mask] == 0).sum()
            approval_rate = predictions[mask].mean()
            
            # Accuracy
            accuracy = (predictions[mask] == y_true[mask]).mean()
            
            results[group] = {
                'ece': ece,
                'tpr': tpr,
                'fpr': fpr,
                'approval_rate': approval_rate,
                'accuracy': accuracy
            }
        
        return pd.DataFrame(results).T

# Initialize pipeline
pipeline = FairPredictionPipeline(
    base_model=trained_model,
    calibrators=calibrated_models,
    thresholds={'male': 0.52, 'female': 0.43}
)

# Evaluate on test set
results = pipeline.evaluate(X_test, y_test, gender_test)
print("\nCombined Pipeline Performance:")
print(results.to_string())

# Calculate gaps
results['tpr_gap'] = abs(results.loc['male', 'tpr'] - results.loc['female', 'tpr'])
results['approval_gap'] = abs(results.loc['male', 'approval_rate'] - 
                               results.loc['female', 'approval_rate'])

print(f"\nTPR Gap: {results.loc['male', 'tpr_gap']:.4f} (Target: <0.05)")
print(f"Approval Gap: {results.loc['male', 'approval_gap']:.4f} (Target: <0.10)")
```

**Expected Combined Results**:
```
Combined Pipeline Performance (Calibration + Thresholds):

              ece    tpr    fpr  approval_rate  accuracy
male        0.030  0.840  0.150          0.700     0.800
female      0.030  0.835  0.140          0.675     0.795

TPR Gap: 0.0050 (Target: <0.05) ✅
Approval Gap: 0.0250 (Target: <0.10) ✅

Improvement Summary:
  Baseline gap: 18%
  After thresholds only: 6%
  After calibration + thresholds: 2.5% ✅
  
  Total improvement: 86% gap reduction
```

#### Step 4: Deploy to Production (1-2 days)

```python
# production/fair_prediction_service.py

class ProductionFairPredictor:
    """
    Production-ready service with calibration + thresholds.
    """
    
    def __init__(self, model_path, calibrators_path, thresholds_path):
        # Load models
        self.base_model = joblib.load(model_path)
        self.calibrators = {
            'male': joblib.load(f'{calibrators_path}/calibrator_male.pkl'),
            'female': joblib.load(f'{calibrators_path}/calibrator_female.pkl')
        }
        
        # Load thresholds
        with open(thresholds_path, 'r') as f:
            self.thresholds = json.load(f)
    
    def predict(self, features, protected_attributes):
        """
        Make fair prediction with full pipeline.
        
        Pipeline:
        1. Base model prediction
        2. Group-specific calibration
        3. Group-specific threshold
        4. Logging and monitoring
        """
        gender = protected_attributes.get('gender', 'unknown')
        
        # Step 1: Base model score
        raw_score = self.base_model.predict_proba([features])[0][1]
        
        # Step 2: Calibration
        if gender in self.calibrators:
            calibrated_score = self.calibrators[gender].predict_proba([features])[0][1]
        else:
            calibrated_score = raw_score  # Fallback for unknown gender
        
        # Step 3: Threshold application
        threshold = self.thresholds.get(gender, self.thresholds.get('default', 0.50))
        decision = 'APPROVED' if calibrated_score >= threshold else 'DENIED'
        
        # Step 4: Prepare response
        response = {
            'decision': decision,
            'calibrated_score': float(calibrated_score),
            'raw_score': float(raw_score),
            'threshold_used': float(threshold),
            'calibration_adjustment': float(calibrated_score - raw_score),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log for monitoring
        self._log_prediction(features, protected_attributes, response)
        
        return response
    
    def _log_prediction(self, features, protected_attrs, response):
        """Log prediction for audit and monitoring"""
        log_entry = {
            'applicant_id': features.get('id'),
            'gender': protected_attrs.get('gender'),
            'age': protected_attrs.get('age'),
            'raw_score': response['raw_score'],
            'calibrated_score': response['calibrated_score'],
            'threshold': response['threshold_used'],
            'decision': response['decision'],
            'timestamp': response['timestamp']
        }
        
        # Write to monitoring system
        monitoring_logger.info(json.dumps(log_entry))

# Initialize service
predictor = ProductionFairPredictor(
    model_path='models/base_model.pkl',
    calibrators_path='models/calibrators',
    thresholds_path='config/thresholds.json'
)

# Example prediction
result = predictor.predict(
    features={'income': 45000, 'credit_score': 680, ...},
    protected_attributes={'gender': 'female', 'age': 28}
)

print(result)
# Output:
# {
#   'decision': 'APPROVED',
#   'calibrated_score': 0.52,
#   'raw_score': 0.48,  # Model over-predicted risk
#   'threshold_used': 0.43,
#   'calibration_adjustment': 0.04,  # Calibration reduced score
#   'timestamp': '2024-11-06T10:30:00'
# }
```

### Performance Impact

**Typical Calibration Results**:
```
Metric                          Before    After     Improvement
────────────────────────────────────────────────────────────────
Male ECE                        0.080     0.030     62.5% ↓
Female ECE                      0.120     0.030     75.0% ↓
Score Consistency (M vs F)      ❌        ✅        Achieved

When combined with thresholds:
Gender Approval Gap             6%        2.5%      Additional 58% ↓
Equal Opportunity Diff          0.06      0.01      Additional 83% ↓
```

**Why Calibration Adds Value Beyond Thresholds**:

1. **Improved Reliability**: Scores now have consistent meaning
2. **Better Risk Assessment**: Default predictions more accurate
3. **Stakeholder Trust**: Predictions interpretable across all groups
4. **Regulatory Compliance**: Demonstrates fairness in risk estimation

### When Calibration Is Critical

**High Priority**:
✅ Risk-based pricing (interest rates based on scores)
✅ Probability scores used for resource allocation
✅ Regulatory scrutiny of risk assessment methodology
✅ Scores presented to humans for decision-making

**Lower Priority** (but still beneficial):
- Binary decisions only (approve/deny)
- Thresholds already achieve fairness
- Scores not exposed to end users

---

*This is part 1 of the Post-Processing Toolkit. Would you like me to continue with the remaining sections (Prediction Transformation, Rejection Classification, Combining Techniques, Implementation Process, Best Practices, etc.)?*

---

## 4. Prediction Transformation

### Concept

**Purpose**: Learn optimal non-linear transformations of model scores to satisfy complex fairness criteria while minimizing prediction distortion.

**When to Use**:
- Complex fairness requirements (multiple criteria simultaneously)
- Ranking/recommendation systems (not binary decisions)
- Protected attributes unavailable at inference time (proxy-based approach)
- Need to preserve relative ordering within groups

### How Prediction Transformation Differs

**Threshold Optimization**: Changes decision boundary only
**Calibration**: Fixes probability interpretation
**Prediction Transformation**: Learns custom score transformation

```
Raw Scores:        [0.45, 0.52, 0.68, 0.73]
                          ↓
Transformation:    f(score, protected_attr)
                          ↓
Fair Scores:       [0.48, 0.54, 0.67, 0.72]
```

### Three Transformation Types

#### Type 1: Linear Transformation by Group

**Simplest approach**: Apply linear scaling per group

```python
def linear_group_transformation(scores, gender, params):
    """
    Transform: fair_score = α * raw_score + β
    
    Different α and β for each group
    """
    transformed = np.zeros_like(scores)
    
    for group in ['male', 'female']:
        mask = gender == group
        alpha = params[group]['alpha']
        beta = params[group]['beta']
        
        transformed[mask] = alpha * scores[mask] + beta
    
    # Clip to [0, 1] range
    transformed = np.clip(transformed, 0, 1)
    
    return transformed

# Learn parameters on validation set
def learn_linear_params(scores_val, labels_val, gender_val, fairness_metric='equal_opportunity'):
    """
    Grid search to find optimal linear transformation parameters.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        # Unpack parameters
        alpha_m, beta_m, alpha_f, beta_f = params
        
        param_dict = {
            'male': {'alpha': alpha_m, 'beta': beta_m},
            'female': {'alpha': alpha_f, 'beta': beta_f}
        }
        
        # Transform scores
        transformed = linear_group_transformation(scores_val, gender_val, param_dict)
        
        # Calculate fairness violation
        tpr_male = calculate_tpr(labels_val[gender_val == 'male'], 
                                 transformed[gender_val == 'male'] >= 0.5)
        tpr_female = calculate_tpr(labels_val[gender_val == 'female'],
                                   transformed[gender_val == 'female'] >= 0.5)
        fairness_violation = abs(tpr_male - tpr_female)
        
        # Calculate distortion (how much we changed scores)
        distortion = np.mean((transformed - scores_val) ** 2)
        
        # Combined objective: balance fairness and distortion
        return fairness_violation + 0.1 * distortion  # λ=0.1 weight for distortion
    
    # Initial guess
    x0 = [1.0, 0.0, 1.0, 0.0]  # No transformation initially
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead')
    
    # Return learned parameters
    alpha_m, beta_m, alpha_f, beta_f = result.x
    return {
        'male': {'alpha': alpha_m, 'beta': beta_m},
        'female': {'alpha': alpha_f, 'beta': beta_f}
    }

# Usage
params = learn_linear_params(scores_validation, labels_validation, gender_validation)
fair_scores = linear_group_transformation(scores_test, gender_test, params)
```

#### Type 2: Learned Non-Linear Transformation

**More flexible**: Use small neural network to learn transformation

```python
import torch
import torch.nn as nn

class FairScoreTransformer(nn.Module):
    """
    Neural network that learns fair score transformation.
    
    Input: (raw_score, protected_attributes_encoded)
    Output: fair_score
    """
    
    def __init__(self, n_protected_features=2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1 + n_protected_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, scores, protected_features):
        """
        Args:
            scores: raw model scores (N,)
            protected_features: one-hot encoded protected attrs (N, n_protected_features)
        """
        # Concatenate inputs
        x = torch.cat([scores.unsqueeze(1), protected_features], dim=1)
        
        # Transform
        fair_scores = self.network(x).squeeze()
        
        return fair_scores

def train_transformer(model, train_loader, val_loader, epochs=50):
    """
    Train transformer to minimize: distortion + fairness_violation
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            scores, protected, labels = batch
            
            # Forward pass
            fair_scores = model(scores, protected)
            
            # Distortion loss (MSE from original scores)
            distortion = torch.mean((fair_scores - scores) ** 2)
            
            # Fairness loss (equal opportunity violation)
            # Calculate TPR for each group
            male_mask = protected[:, 0] == 1
            female_mask = protected[:, 1] == 1
            
            tpr_male = ((fair_scores[male_mask] >= 0.5) & (labels[male_mask] == 1)).float().mean()
            tpr_female = ((fair_scores[female_mask] >= 0.5) & (labels[female_mask] == 1)).float().mean()
            
            fairness_loss = torch.abs(tpr_male - tpr_female)
            
            # Combined loss
            loss = 0.1 * distortion + fairness_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            val_fairness, val_distortion = evaluate_transformer(model, val_loader)
            print(f"Epoch {epoch+1}: Fairness={val_fairness:.4f}, Distortion={val_distortion:.4f}")
    
    return model

# Usage
transformer = FairScoreTransformer(n_protected_features=2)
trained_transformer = train_transformer(transformer, train_loader, val_loader)

# Apply to new data
with torch.no_grad():
    fair_scores = trained_transformer(test_scores_tensor, test_protected_tensor)
```

#### Type 3: Ranking-Aware Transformation

**For recommendation systems**: Preserve ranking order while achieving fairness

```python
def ranking_fair_transformation(scores, gender, protected_group='female', alpha=0.5):
    """
    Transform scores to promote protected group in rankings while preserving
    relative order within groups.
    
    Args:
        scores: Model scores
        gender: Protected attribute
        protected_group: Which group to promote
        alpha: Strength of promotion (0 = no change, 1 = maximum)
    """
    transformed = scores.copy()
    
    # Boost scores for protected group
    mask = gender == protected_group
    
    # Calculate boost: move scores up proportionally
    # Higher scores get smaller boost (already competitive)
    # Lower scores get larger boost (need help)
    boost = alpha * (1 - scores[mask]) * 0.1  # Max 10% boost
    
    transformed[mask] = scores[mask] + boost
    
    # Ensure transformed scores preserve within-group ranking
    for group in ['male', 'female']:
        mask = gender == group
        
        # Get ranking of original scores
        original_rank = scores[mask].argsort().argsort()
        
        # Sort transformed scores to match original ranking
        transformed[mask] = np.sort(transformed[mask])[original_rank]
    
    return np.clip(transformed, 0, 1)

# Usage for top-k recommendations
fair_scores = ranking_fair_transformation(recommendation_scores, user_gender)

# Select top-k
top_k_indices = np.argsort(fair_scores)[-k:]
```

### When to Use Prediction Transformation

**Ideal Scenarios**:
✅ Ranking/recommendation systems (relative order matters)
✅ Multiple conflicting fairness objectives
✅ Protected attributes unavailable at inference (learn proxy-free transformation)
✅ Need smooth, interpretable fairness-accuracy trade-off

**Less Suitable**:
❌ Simple binary classification (thresholds simpler and more interpretable)
❌ Well-calibrated scores (calibration more direct)
❌ Small datasets (risk overfitting)

---

## 5. Rejection Classification

### Concept

**Purpose**: Route uncertain or high-fairness-risk predictions to human review instead of automated decisions, creating a hybrid human-AI system.

**Key Insight**: Fairness errors often concentrate in borderline cases where the model is most uncertain.

### Why Rejection Works

**Fairness Risk Distribution**:

```
Score Range    Model Uncertainty    Fairness Gap    Action
────────────────────────────────────────────────────────────
0.8-1.0        Low (confident)      1-2%           ✅ Automate
0.6-0.8        Medium               3-5%           ✅ Automate
0.4-0.6        High (uncertain)     10-15%         ⚠️ Review
0.2-0.4        Medium               4-6%           ✅ Automate
0.0-0.2        Low (confident)      1-2%           ✅ Automate

Observation: Borderline cases (0.4-0.6) have 3-5x higher fairness risk
Solution: Route these to human reviewers
```

### Two Rejection Strategies

#### Strategy 1: Confidence-Based Rejection

**Reject low-confidence predictions regardless of demographics**

```python
def confidence_based_rejection(scores, confidence_threshold=0.7):
    """
    Route low-confidence predictions to human review.
    
    Args:
        scores: Model probability scores
        confidence_threshold: Min confidence for automation
        
    Returns:
        mask: True = automate, False = route to human
    """
    # Confidence = distance from decision boundary (0.5)
    confidence = np.abs(scores - 0.5)
    
    # Automate if confident
    automate_mask = confidence >= (confidence_threshold - 0.5)
    
    return automate_mask

# Usage
automate_mask = confidence_based_rejection(model_scores, confidence_threshold=0.7)

automated_decisions = make_automated_decision(model_scores[automate_mask])
review_queue = create_review_queue(model_scores[~automate_mask])

print(f"Automated: {automate_mask.mean():.1%}")
print(f"Human review: {(~automate_mask).mean():.1%}")
```

**Typical Results**:
```
Confidence Threshold    Coverage    Fairness Gap    Cost
0.9 (very conservative) 60%         0.5%            High review cost
0.7 (balanced)          85%         2%              Moderate
0.5 (aggressive)        95%         4%              Low review cost
```

#### Strategy 2: Fairness-Aware Rejection

**Prioritize routing cases that contribute most to fairness gaps**

```python
class FairnessAwareRejection:
    """
    Route predictions to human review based on fairness risk.
    """
    
    def __init__(self, coverage_target=0.85, fairness_threshold=0.05):
        """
        Args:
            coverage_target: Target automation rate (0.85 = 85% automated)
            fairness_threshold: Max acceptable fairness gap
        """
        self.coverage_target = coverage_target
        self.fairness_threshold = fairness_threshold
        self.rejection_params = None
    
    def fit(self, scores, gender, labels):
        """
        Learn which cases to reject based on fairness contribution.
        """
        # Calculate fairness gap by score range
        score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        
        fairness_risk = {}
        for lower, upper in score_ranges:
            in_range = (scores >= lower) & (scores < upper)
            
            if in_range.sum() < 10:
                continue
            
            # Calculate gap in this range
            male_mask = (gender == 'male') & in_range
            female_mask = (gender == 'female') & in_range
            
            if male_mask.sum() > 0 and female_mask.sum() > 0:
                tpr_male = ((scores[male_mask] >= 0.5) & (labels[male_mask] == 1)).mean()
                tpr_female = ((scores[female_mask] >= 0.5) & (labels[female_mask] == 1)).mean()
                
                gap = abs(tpr_male - tpr_female)
                fairness_risk[(lower, upper)] = gap
        
        # Identify rejection region: highest fairness risk + achieve coverage target
        rejection_region = max(fairness_risk, key=fairness_risk.get)
        
        self.rejection_params = {
            'score_range': rejection_region,
            'fairness_risk_by_range': fairness_risk
        }
        
        return self
    
    def should_reject(self, score, gender):
        """
        Determine if prediction should be routed to human review.
        """
        lower, upper = self.rejection_params['score_range']
        
        # Base rejection: in high-risk score range
        in_risk_range = (score >= lower) & (score < upper)
        
        # Additional factors:
        # 1. Prioritize disadvantaged group
        is_disadvantaged = gender == 'female'  # Based on overall analysis
        
        # Reject if: in risk range AND disadvantaged
        # (This ensures human review for cases most likely to be unfair)
        reject = in_risk_range & is_disadvantaged
        
        return reject
    
    def apply(self, scores, gender):
        """
        Apply rejection logic to batch of predictions.
        
        Returns:
            automate_mask: Boolean array (True = automate, False = review)
        """
        automate_mask = np.ones(len(scores), dtype=bool)
        
        for i, (score, g) in enumerate(zip(scores, gender)):
            if self.should_reject(score, g):
                automate_mask[i] = False
        
        return automate_mask

# Usage
rejection_system = FairnessAwareRejection(coverage_target=0.85, fairness_threshold=0.05)
rejection_system.fit(val_scores, val_gender, val_labels)

# Apply to test set
automate_mask = rejection_system.apply(test_scores, test_gender)

print(f"Automated: {automate_mask.mean():.1%}")
print(f"Routed to review: {(~automate_mask).mean():.1%}")
```

### Human Review Interface Design

**Critical**: How cases are presented to human reviewers affects fairness

```python
def create_review_interface_data(applicant_features, model_prediction, protected_attrs):
    """
    Prepare data for human reviewer.
    
    Best practices:
    1. Show model prediction as SUGGESTION, not default
    2. Provide similar cases for reference
    3. Hide protected attributes (blind review)
    4. Show rationale/explanation
    """
    
    return {
        'applicant_id': applicant_features['id'],
        
        # Applicant information (WITHOUT protected attributes)
        'income': applicant_features['income'],
        'credit_score': applicant_features['credit_score'],
        'employment_years': applicant_features['employment_years'],
        'loan_amount': applicant_features['loan_amount'],
        
        # Model suggestion (NOT highlighted as recommendation)
        'model_suggestion': {
            'decision': 'APPROVE' if model_prediction >= 0.5 else 'DENY',
            'confidence': abs(model_prediction - 0.5) * 2,  # 0-1 scale
            'explanation': generate_explanation(applicant_features, model_prediction)
        },
        
        # Similar cases for reference
        'similar_cases': find_similar_cases(applicant_features, k=5),
        
        # Reason routed for review
        'review_reason': 'Borderline case with fairness sensitivity',
        
        # Reviewer decision form
        'decision_options': ['APPROVE', 'DENY', 'REQUEST_MORE_INFO'],
        'rationale_required': True
    }
```

**HTML Interface Example**:
```html
<div class="review-interface">
    <h2>Loan Application Review</h2>
    <p class="review-reason">⚠️ Routed for review: Borderline case</p>
    
    <!-- Applicant Details (Protected attributes hidden) -->
    <section class="applicant-info">
        <h3>Applicant Information</h3>
        <table>
            <tr><td>Income:</td><td>$45,000/year</td></tr>
            <tr><td>Credit Score:</td><td>680</td></tr>
            <tr><td>Employment:</td><td>3 years current job</td></tr>
            <tr><td>Loan Amount:</td><td>$15,000</td></tr>
        </table>
    </section>
    
    <!-- Model Suggestion (not prominent) -->
    <section class="model-suggestion">
        <h3>Model Analysis</h3>
        <p>Model suggests: <span class="suggestion">Borderline Approve</span></p>
        <p>Confidence: Low (55%)</p>
        <details>
            <summary>View Explanation</summary>
            <p>Primary factors: Credit score adequate, income slightly low for requested amount...</p>
        </details>
    </section>
    
    <!-- Similar Cases (for consistency) -->
    <section class="similar-cases">
        <h3>Similar Past Cases</h3>
        <table>
            <tr>
                <th>Income</th><th>Credit</th><th>Employment</th><th>Decision</th><th>Outcome</th>
            </tr>
            <tr>
                <td>$46K</td><td>682</td><td>2.5 yrs</td><td>Approved</td><td>No default</td>
            </tr>
            <tr>
                <td>$44K</td><td>678</td><td>4 yrs</td><td>Approved</td><td>No default</td>
            </tr>
            <tr>
                <td>$45K</td><td>675</td><td>2 yrs</td><td>Denied</td><td>N/A</td>
            </tr>
        </table>
    </section>
    
    <!-- Decision Form -->
    <section class="decision-form">
        <h3>Your Decision</h3>
        <form>
            <input type="radio" name="decision" value="approve"> Approve<br>
            <input type="radio" name="decision" value="deny"> Deny<br>
            <input type="radio" name="decision" value="more-info"> Request More Information<br>
            
            <label>Rationale (required):</label>
            <textarea name="rationale" required></textarea>
            
            <button type="submit">Submit Decision</button>
        </form>
    </section>
</div>
```

### Measuring Rejection Impact

```python
def analyze_rejection_impact(automated_preds, reviewed_preds, labels, gender):
    """
    Compare automated vs human-reviewed decisions.
    """
    results = {
        'automated': {},
        'reviewed': {},
        'combined': {}
    }
    
    # Automated decisions fairness
    for group in ['male', 'female']:
        mask_auto = (gender == group) & automated_preds['is_automated']
        
        if mask_auto.sum() > 0:
            tpr_auto = ((automated_preds['decision'][mask_auto] == 1) & 
                       (labels[mask_auto] == 1)).sum() / (labels[mask_auto] == 1).sum()
            
            results['automated'][f'tpr_{group}'] = tpr_auto
    
    # Human-reviewed decisions fairness
    for group in ['male', 'female']:
        mask_review = (gender == group) & reviewed_preds['is_reviewed']
        
        if mask_review.sum() > 0:
            tpr_review = ((reviewed_preds['decision'][mask_review] == 1) & 
                         (labels[mask_review] == 1)).sum() / (labels[mask_review] == 1).sum()
            
            results['reviewed'][f'tpr_{group}'] = tpr_review
    
    # Combined (overall system fairness)
    all_decisions = np.where(automated_preds['is_automated'],
                            automated_preds['decision'],
                            reviewed_preds['decision'])
    
    for group in ['male', 'female']:
        mask = gender == group
        tpr_combined = ((all_decisions[mask] == 1) & (labels[mask] == 1)).sum() / (labels[mask] == 1).sum()
        results['combined'][f'tpr_{group}'] = tpr_combined
    
    # Calculate gaps
    results['automated']['gap'] = abs(results['automated']['tpr_male'] - 
                                     results['automated']['tpr_female'])
    results['reviewed']['gap'] = abs(results['reviewed']['tpr_male'] - 
                                    results['reviewed']['tpr_female'])
    results['combined']['gap'] = abs(results['combined']['tpr_male'] - 
                                    results['combined']['tpr_female'])
    
    return results

# Usage
impact = analyze_rejection_impact(auto_decisions, human_decisions, test_labels, test_gender)

print("Rejection Classification Impact Analysis:")
print(f"Automated decisions gap: {impact['automated']['gap']:.3f}")
print(f"Human-reviewed gap: {impact['reviewed']['gap']:.3f}")
print(f"Combined system gap: {impact['combined']['gap']:.3f}")
```

**Typical Results**:
```
Without Rejection (100% automated):
  Gender gap: 3%

With Rejection (85% automated, 15% reviewed):
  Automated portion gap: 2% (easier cases, fairer)
  Human-reviewed gap: 0.5% (humans fairer on borderline)
  Combined system gap: 0.5% * 0.85 + 0.5% * 0.15 = 1.7%
  
Improvement: 3% → 1.7% (43% additional reduction)
Cost: 15% of cases require human review (~$20/case)
```

###
