# 04: Pre-Processing Fairness Toolkit

## Purpose

Fix biased DATA before training models. Address representation disparities, proxy features, and selection bias at the source.

**When to Use**: 
- Causal analysis identifies proxy or selection bias
- Retraining model or building new model
- Want to address root causes in data

**Time**: 1-2 weeks  
**Expected Impact**: 50-80% gap reduction for proxy/selection bias

---

## 4.1 Three Core Techniques

### Technique 1: Reweighting & Resampling

**Purpose**: Address representation disparities and selection bias

**When to Use**:
- Training data underrepresents certain groups (e.g., 80% male, 20% female)
- Historical data reflects past discrimination
- Need to balance influence without changing data structure

**How It Works**:
- **Reweighting**: Assign higher weights to underrepresented samples
- **Resampling**: Oversample minority class or undersample majority class

**Example**:
```
Before: 8000 male samples, 2000 female samples
After (Reweighting): Same samples, female weight = 4.0
After (Resampling): 8000 male, 8000 female (synthetic added)
```

**Implementation Approach**:
1. Calculate demographic distribution in training data
2. Compare to deployment population (target distribution)
3. Compute weights: `weight = target_proportion / current_proportion`
4. Apply weights in model training

**Trade-offs**:
- ✅ Preserves all original data
- ✅ Simple to implement and understand
- ⚠️ May amplify noise if minority data is low quality
- ⚠️ Doesn't remove proxy features

**Selection Guidance**:
- Small imbalance (60/40): Reweighting
- Medium imbalance (70/30): Resampling  
- Severe imbalance (90/10): Synthetic generation
- Historical bias: Time-weighted (recent data higher weight)

---

### Technique 2: Feature Transformation

**Purpose**: Remove proxy discrimination while preserving legitimate predictive information

**When to Use**:
- Causal analysis identifies proxy features (correlate with protected attrs, no causal link)
- Examples: Zip code → race, First name → gender, Part-time work → gender

**How It Works**:
Three transformation methods:

**A. Disparate Impact Remover**:
- Linear transformation reduces correlation with protected attribute
- Preserves rank ordering of feature values
- Best for: Single proxy feature, want interpretability

**B. Optimal Transport**:
- Sophisticated distribution matching
- Transforms feature distribution to remove demographic disparities
- Best for: Multiple proxy features, complex relationships

**C. Fair Representations**:
- Learn new feature space where protected attributes are obscured
- Uses autoencoders or adversarial networks
- Best for: High-dimensional data, willing to trade interpretability

**Implementation Approach**:
1. Identify proxy features (correlation >0.5, no domain-confirmed causal link)
2. Choose transformation method based on complexity
3. Apply transformation to training data
4. Validate: Check correlation reduced, predictive power maintained
5. Document transformation parameters (apply to test/production data)

**Trade-offs**:
- ✅ Addresses root cause (removes proxy effect)
- ✅ Can preserve most predictive information
- ⚠️ May reduce model performance (2-5% typical)
- ⚠️ Fair representations lose interpretability
- ⚠️ Need to apply same transformation at inference time

**Example - Zip Code Proxy**:
```
Problem: Zip code correlates with race (r=0.72) due to residential segregation
Analysis: Zip code contains economic signal + proxy for race
Solution: 
  - Extract legitimate economic features (median income, unemployment rate)
  - Remove raw zip code
  - Or: Transform zip code to remove race correlation
Result: Economic predictiveness maintained, race proxy removed
```

---

### Technique 3: Synthetic Data Generation

**Purpose**: Augment severely imbalanced datasets

**When to Use**:
- Extreme imbalance (<10% minority group)
- Insufficient samples for minority group to learn patterns
- Cannot collect more real data

**How It Works**:
Two generation methods:

**A. SMOTE (Synthetic Minority Over-sampling Technique)**:
- Creates synthetic samples by interpolating between existing minority samples
- Best for: Tabular data, simple and fast

**B. Conditional GANs**:
- Generative model learns minority group distribution
- Generates realistic synthetic samples
- Best for: Complex data, need high-quality synthetics

**Implementation Approach**:
1. Train generator on minority group data
2. Generate synthetic samples to balance dataset
3. Validate quality: Check synthetic samples are realistic
4. Combine with real data for training
5. Document that synthetics used (transparency)

**Trade-offs**:
- ✅ Solves severe imbalance problem
- ✅ Can improve minority group model performance
- ⚠️ Quality concerns (synthetic ≠ real)
- ⚠️ Ethical questions about fake data
- ⚠️ May not capture full minority group diversity

**Use Sparingly**: Only when real data collection infeasible

---

## 4.2 Selection Decision Tree

```
START: Causal analysis identifies data-level issues

What bias pattern detected?

├─ REPRESENTATION DISPARITY
│  │ (Underrepresented groups in training data)
│  │
│  ├─ How severe?
│  │  ├─ Small (60/40) → Reweighting
│  │  ├─ Medium (70/30) → Resampling  
│  │  └─ Severe (90/10) → Synthetic Generation
│  │
│  └─ Output: Balanced training dataset

├─ PROXY DISCRIMINATION  
│  │ (Features correlate but no causal link)
│  │
│  ├─ How many proxy features?
│  │  ├─ Single → Disparate Impact Remover
│  │  ├─ Multiple → Optimal Transport
│  │  └─ High-dimensional → Fair Representations
│  │
│  └─ Output: Transformed features (proxy removed)

├─ SELECTION BIAS
│  │ (Historical data reflects past discrimination)
│  │
│  ├─ Can collect new data?
│  │  ├─ Yes → Collect representative data + Time-weight
│  │  └─ No → Reweight by time (recent higher)
│  │
│  └─ Output: Reweighted dataset

└─ LABEL BIAS
   │ (Training labels contain discrimination)
   │
   ├─ Apply label correction techniques
   │  └─ Prejudice Remover or Massaging
   │
   └─ Output: Corrected labels

END: Debiased training data ready for in-processing
```

---

## 4.3 Implementation Process

### Step 1: Data Auditing (1-2 days)

**Objectives**:
- Understand current data quality
- Quantify representation disparities
- Identify proxy features
- Assess label quality

**Process**:
1. **Representation Analysis**:
   - Calculate demographic distributions
   - Compare to deployment population
   - Check intersectional representation (gender × race)
   - Flag groups with <100 samples

2. **Correlation Analysis**:
   - Compute correlation between each feature and protected attributes
   - Flag correlations >0.5 as potential proxies
   - Consult domain experts: Causal link or proxy?

3. **Temporal Analysis**:
   - Plot demographic distributions over time
   - Identify periods of discrimination in historical data
   - Determine appropriate time windows

4. **Label Quality**:
   - Check for systematic labeling differences by group
   - Interview annotators about potential biases
   - Validate labels on diverse sample

**Deliverable**: Data quality report with recommendations

### Step 2: Technique Selection (0.5 days)

**Use the decision tree above**, considering:
- Primary bias pattern from causal analysis
- Data availability constraints
- Model retraining timeline
- Performance requirements

**Deliverable**: Selected technique with rationale documented

### Step 3: Implementation (3-5 days)

**For Reweighting**:
```
1. Calculate target weights:
   weight[group] = (target_proportion / current_proportion)
   
2. Validate weights sum correctly:
   Check: sum(weights) ≈ N (total samples)
   
3. Apply in training:
   model.fit(X, y, sample_weight=weights)
   
4. Document:
   - Weight calculation method
   - Target distribution used
   - Resulting effective sample sizes
```

**For Feature Transformation**:
```
1. Fit transformation on training data:
   transformer.fit(X_train, protected_attrs_train)
   
2. Transform all datasets:
   X_train_fair = transformer.transform(X_train)
   X_test_fair = transformer.transform(X_test)
   
3. Validate transformation:
   - Check correlation reduction
   - Verify predictive power maintained
   
4. Save transformation parameters:
   - Must apply same transform at inference
   - Document in model card
```

**For Synthetic Generation**:
```
1. Train generator on minority group:
   generator.fit(X_minority, y_minority)
   
2. Generate synthetics:
   X_synthetic = generator.generate(n_samples=5000)
   
3. Quality check:
   - Visual inspection
   - Statistical similarity tests
   - Domain expert review
   
4. Combine with real data:
   X_combined = pd.concat([X_real, X_synthetic])
   Label as synthetic for transparency
```

### Step 4: Validation (1-2 days)

**Validate Pre-Processing Effectiveness**:

1. **Fairness Improvement**:
   - Train baseline model on original data
   - Train model on preprocessed data
   - Compare demographic gaps
   - Target: >30% reduction from pre-processing alone

2. **Performance Check**:
   - Compare AUC on test set
   - Acceptable loss: <3% for pre-processing
   - If loss >5%, reconsider technique

3. **Intersectional Analysis**:
   - Check all demographic combinations
   - Ensure no subgroup harmed by preprocessing

4. **Documentation**:
   - What changed in data
   - Why this technique was chosen
   - Validation results
   - Parameters for reproduction

**Deliverable**: Pre-processing validation report

---

## 4.4 Best Practices

### DO's ✅

1. **Always start with causal analysis**
   - Don't remove features without understanding why they're proxies
   - Domain expertise critical for proxy identification

2. **Validate on held-out data**
   - Pre-processing can overfit to training distribution
   - Test on temporal holdout (future data)

3. **Document everything**
   - Preprocessing must be reproducible
   - Parameters saved for production inference
   - Rationale documented for auditors

4. **Consider intersectionality**
   - Reweight for all protected attribute combinations
   - Don't just balance gender, balance gender × race × age

5. **Preserve legitimate predictors**
   - Goal is remove bias, not all signal
   - Feature transformation should maintain predictive value

### DON'Ts ❌

1. **Don't remove features blindly**
   - "Zip code correlates with race, remove it" → May lose legitimate economic signal
   - Instead: Extract legitimate signal, transform to remove proxy effect

2. **Don't over-rely on synthetic data**
   - Synthetic ≠ real, quality concerns
   - Use only when no alternative
   - Document transparently

3. **Don't forget to apply at inference**
   - Pre-processing transformations must apply to production data
   - Test: Does inference pipeline apply same transformations?

4. **Don't neglect performance**
   - Pre-processing shouldn't destroy model utility
   - If AUC drops >5%, reconsider approach

5. **Don't treat as one-time**
   - Data distributions change over time
   - Revisit pre-processing quarterly

---

## 4.5 Integration with Other Components

### Inputs from Causal Analysis

**Causal Analysis Provides**:
- Discrimination mechanisms identified (direct, proxy, mediator, selection)
- Quantified contribution of each pathway
- List of proxy features to address
- Selection bias severity estimate

**Pre-Processing Uses This To**:
- Select appropriate technique (decision tree)
- Prioritize which features to transform first
- Set reweighting parameters based on selection bias severity
- Document causal rationale for changes

### Outputs to In-Processing

**Pre-Processing Provides**:
- Debiased training dataset
- Transformation parameters (for inference)
- Validation report (fairness improvement from data fixes)
- Remaining bias to address (in-processing target)

**In-Processing Uses This To**:
- Train on cleaner data (better starting point)
- Set fairness constraint strength (λ) based on remaining bias
- Avoid double-correction (don't overcompensate)

### Outputs to Validation

**Pre-Processing Provides**:
- Baseline: Original data performance
- Preprocessed: Debiased data performance
- Delta: Fairness improvement from data fixes alone

**Validation Uses This To**:
- Attribute improvement correctly (data vs model vs post)
- Validate that pre-processing didn't introduce new biases
- Measure cumulative effect (pre + in + post)

---

## 4.6 Common Pitfalls & Solutions

### Pitfall 1: Removing Legitimate Predictors

**Symptom**: Accuracy drops >5% after pre-processing

**Cause**: Removed feature that had both proxy effect AND legitimate predictive value

**Solution**:
- Don't remove entirely, transform instead
- Extract legitimate components (e.g., economic data from zip code)
- Use Optimal Transport to preserve information while removing bias

### Pitfall 2: Insufficient Minority Data

**Symptom**: Synthetic generation produces low-quality samples

**Cause**: Too few real minority samples (<100) to learn distribution

**Solution**:
- Collect more real data if possible
- Use simpler generation (SMOTE) rather than complex (GANs)
- Document quality limitations transparently
- Consider hybrid: Real data + conservative synthetic augmentation

### Pitfall 3: Overfitting to Training Distribution

**Symptom**: Pre-processing works on training data, fails on test data

**Cause**: Transformation parameters fit too closely to training set

**Solution**:
- Validate on temporal holdout (future data)
- Use simpler transformations (less parameters to overfit)
- Cross-validation during parameter selection

### Pitfall 4: Forgetting Inference Pipeline

**Symptom**: Production model fails, pre-processing not applied

**Cause**: Transformation parameters not deployed to inference

**Solution**:
- Save transformation as part of model pipeline
- Test inference pipeline applies transformations
- Document in model card: "Pre-processing required"

---

## 4.7 Deliverables Checklist

After completing pre-processing, you should have:

- [x] **Data Quality Report**: Current state analysis
- [x] **Technique Selection Document**: Rationale for chosen approach
- [x] **Debiased Dataset**: Training data with bias mitigated
- [x] **Transformation Parameters**: Saved for production use
- [x] **Validation Report**: Before/after fairness metrics
- [x] **Documentation**: Changes made, why, and how to reproduce
- [x] **Integration Guide**: How to apply in production inference

---

## Summary

Pre-Processing fixes bias at its source—the data. By addressing representation disparities, removing proxy effects, and correcting selection bias before training, you create a solid foundation for fair ML systems.

**Key Takeaways**:
1. Three techniques: Reweighting, Transformation, Generation
2. Choose based on bias pattern (decision tree)
3. Validate effectiveness (>30% improvement target)
4. Document for reproducibility and audit
5. Integrate with inference pipeline (critical)

**Typical Impact**: 50-80% gap reduction for proxy/selection bias

