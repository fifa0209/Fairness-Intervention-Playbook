# 3. Pre-Processing Fairness Toolkit

**Goal**: Address unfairness by transforming the data *before* modeling.

## Steps

### 3.1. Audit and Diagnose Data

- Analyze group representation using statistical summary tables and visualizations.
- Check for correlation between features and protected attributes.

### 3.2. Transformation Techniques

- **Reweighting**: Assign higher weights to underrepresented groups.
- **Resampling**: Balance groups using common sampling techniques.
- **Feature Transformation**: Remove or disguise proxy variables, e.g., replace zip codes with less granular regions.

### 3.3. Implementation Guidance

- Use `pandas` for analysis, data manipulation, and plotting.
- Reweight groups:
  - Calculate instance weights = 1 / (group size); apply during model fitting.

### 3.4. When to Use

- When dataset is imbalanced or has biased labeling.
- When you can control data processing or model input pipeline.

### 3.5. Example

> *If females make up 70% of loan denials but only 30% of applicants, increase their representation in training or assign higher instance weights.*

---

**Resources**: Scikit-learn (pip install scikit-learn), AIF360 for fairness pre-processing
