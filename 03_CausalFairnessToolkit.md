# 03: Causal Fairness Toolkit

## Purpose and Philosophy

The Causal Fairness Toolkit helps you understand **WHY** bias exists in your ML system before attempting to fix it. Rather than simply observing that different demographic groups receive different outcomes, causal analysis reveals the underlying mechanisms that create these disparities.

**Core Principle**: Treating symptoms without understanding root causes leads to ineffective interventions. Causal analysis is the diagnostic phase that ensures your treatment addresses the actual disease, not just its visible symptoms.

**Time Investment**: 3-5 days  
**ROI**: Saves 2-3 weeks of trial-and-error, 3x higher success rate

---

## 3.1 Why Causal Analysis Matters

### The Problem with Correlation-Only Approaches

Traditional fairness audits identify disparities but don't explain them:
- "Women receive loan approvals 18% less often than men" ← *What we observe*
- **But why?** ← *What we need to know*

Without causal understanding, teams make costly mistakes:

**Example 1: Removing the Wrong Feature**
- Observation: Zip code correlates with race
- Wrong fix: Remove zip code from model
- Result: Model performance drops 15%, bias barely improves
- Root cause missed: Zip code proxies for race due to historical redlining, but it also contains legitimate economic information

**Example 2: Applying the Wrong Intervention**
- Observation: Gender approval gap exists
- Wrong fix: Apply demographic parity constraint
- Result: Qualified women still underrepresented, unqualified men over-approved
- Root cause missed: Historical employment discrimination created income gaps; equal approval rates ignore qualification differences

### What Causal Analysis Provides

1. **Mechanism Identification**: Direct vs. Proxy vs. Mediator vs. Selection bias
2. **Intervention Targeting**: Which stage of the pipeline to modify
3. **Trade-off Understanding**: Which variables are legitimately predictive vs. discriminatory
4. **Counterfactual Reasoning**: Would an individual's outcome change if only their protected attribute differed?

---

## The Four Types of Discrimination Mechanisms

### 1. Direct Discrimination

**Definition**: Protected attributes directly influence the decision.

**Causal Structure**:
```
Protected Attribute (Gender) → Decision (Loan Approval)
```

**Real-World Example**:
- A loan model explicitly uses "gender" as an input feature
- Different credit score thresholds are hardcoded by gender
- Manual decision rules reference protected attributes

**Identifying Signs**:
- Protected attribute appears in the feature set
- Decision logic explicitly references demographic categories
- Historical documentation shows different standards by group

**Intervention Priority**: **CRITICAL** - This is illegal in most contexts and must be removed immediately.

**Appropriate Interventions**:
- Remove protected attribute from features
- Add fairness constraints during model training
- Implement blind review processes

---

### 2. Proxy Discrimination

**Definition**: Seemingly neutral features correlate with protected attributes due to unmeasured common causes, creating indirect discrimination without legitimate predictive value.

**Causal Structure**:
```
        Unmeasured Confounder
               ↙     ↘
Protected Attribute    Proxy Feature
                            ↓
                        Decision
```

**Real-World Example**:
- **Zip code as race proxy**: Historical redlining caused racial residential segregation. Zip code correlates with race but encodes discrimination, not creditworthiness.
- **First name as gender proxy**: "Jennifer" vs. "James" predicts gender. Using names in hiring discriminates without justification.
- **Part-time employment as gender proxy**: Women disproportionately work part-time due to caregiving expectations. Penalizing part-time status discriminates against women.

**Identifying Signs**:
- High correlation (>0.5) between feature and protected attribute
- Feature's predictive power vanishes when protected attribute controlled
- Domain expertise confirms no causal link to outcome
- Historical analysis shows feature emerged from discriminatory practices

**Intervention Priority**: **HIGH** - These features perpetuate historical discrimination without providing legitimate predictive signal.

**Appropriate Interventions**:
- Pre-processing: Transform or remove proxy features
- In-processing: Adversarial debiasing (prevent model from encoding protected attributes)
- Documentation: Explain why feature was removed to stakeholders

**Critical Distinction**: Not all correlations are proxies. A feature is only a problematic proxy if:
1. It correlates with protected attributes AND
2. Its predictive power comes FROM that correlation, not from legitimate causal pathways

---

### 3. Mediator Discrimination

**Definition**: Protected attributes influence legitimate intermediate variables, which then affect outcomes. This creates a genuine dilemma because the mediator has both discriminatory origins AND predictive validity.

**Causal Structure**:
```
Protected Attribute → Mediator Variable → Decision
    (Gender)        (Income/Education)   (Approval)
```

**Real-World Example**:
- **Gender → Income Gap → Loan Approval**
  - Historical wage discrimination means women earn less
  - Income legitimately predicts loan repayment ability
  - But lower income stems from discrimination, not capability

- **Race → Educational Access → Hiring**
  - Systemic inequality limits educational opportunities for minorities
  - Education credentials predict job performance
  - But credential gaps reflect opportunity gaps, not ability gaps

**Identifying Signs**:
- Protected attribute causally influences the mediator
- Mediator has legitimate predictive relationship with outcome
- Domain expertise confirms mediator reflects historical inequality
- Removing mediator significantly harms model performance

**Intervention Priority**: **COMPLEX** - Requires balancing fairness and legitimate prediction.

**The Mediator Dilemma**:
```
Should we use income to predict loan repayment?

YES because:                    NO because:
- Income truly predicts ability - Income gaps reflect discrimination
- Ignoring it risks defaults    - Using it perpetuates inequality
- Lenders need accurate risk    - Qualified women denied unfairly
```

**Appropriate Interventions**:
- **Partial adjustment**: Use mediator but add fairness constraints
- **Multi-objective optimization**: Explicitly balance accuracy and fairness
- **Augmentation**: Supplement mediator with non-discriminatory proxies for capability
- **Long-term**: Address root causes (pay equity, educational access)

**Framework for Decision**:
1. **Is the mediator's causal link to outcome legitimate?**
   - Income → Repayment ability: YES (stronger economy → better repayment)
   - Education → Job performance: PARTIAL (credential ≠ capability)

2. **Can we measure the underlying construct directly?**
   - Instead of education credentials: Work samples, skills tests
   - Instead of employment gaps: Caregiving contributions, volunteer work

3. **What are the consequences of each choice?**
   - Use mediator: Accurate predictions, perpetuates inequality
   - Remove mediator: Fairer outcomes, increased risk/cost
   - Constrained use: Balance both concerns

---

### 4. Selection Bias

**Definition**: Training data reflects historical discrimination, causing models to learn and perpetuate past biases even when no problematic features are used.

**Causal Structure**:
```
Historical Discrimination → Data Collection Process → Training Data
                                                            ↓
                                                     Biased Model
```

**Real-World Example**:
- **Hiring data from 1990-2010**: When few women were hired, model learns women are "risky hires"
- **Loan data pre-Fair Housing Act**: Minorities systematically denied, model learns minorities are "high risk"
- **Criminal justice data**: Over-policing of minority communities creates biased arrest records

**Identifying Signs**:
- Training data demographics don't match current population
- Historical policies explicitly discriminated
- Underrepresented groups have systematically different outcomes in training data
- Model performs worse on minority groups (didn't learn their patterns well)

**Intervention Priority**: **FOUNDATIONAL** - If training data is biased, all downstream fixes are bandaids.

**Appropriate Interventions**:
- Pre-processing: Reweight samples to correct historical imbalances
- Data collection: Gather new, representative data
- Hybrid: Use recent data more heavily than historical data
- Documentation: Acknowledge data limitations transparently

**Long-term Solution**: Update data collection practices to ensure ongoing representativeness.

---

## The Causal Analysis Process

### Step 1: Construct the Causal Diagram (Directed Acyclic Graph)

A causal diagram visually maps how variables influence each other in your system.

#### 1.1 Identify Key Variables

**Protected Attributes** (What we're protecting):
- Demographics: Gender, race, age, disability status
- Legally protected characteristics in your jurisdiction

**Outcome** (What we're predicting):
- Binary: Loan approved/denied, hired/not hired
- Continuous: Risk score, salary offered
- Multiple: Job level, treatment plan

**Features** (What the model uses):
- Demographic proxies: Zip code, first name, language
- Mediators: Income, education, employment history
- Legitimate predictors: Credit history, skills tests, medical indicators

**Unmeasured Variables** (Context we must infer):
- Historical discrimination
- Structural inequality
- Cultural factors
- Economic conditions

#### 1.2 Draw Causal Arrows

An arrow X → Y means "X causally influences Y"

**Rules for Drawing Arrows**:
1. **Temporal ordering**: Cause precedes effect
   - Birth year → Current age ✓
   - Current age → Birth year ✗

2. **Interventional test**: If we changed X, would Y change?
   - Income → Ability to repay loan ✓ (if income increased, repayment improves)
   - Zip code → Creditworthiness ✗ (moving doesn't change creditworthiness)

3. **Domain expertise**: Consult experts who understand real mechanisms
   - Medical: Doctor confirms symptom → diagnosis
   - Lending: Economist confirms employment → income

4. **No cycles**: DAGs cannot have loops
   - Employment → Income → Credit Score (valid)
   - Credit Score → Income → Credit Score (invalid - must break cycle)

#### Example Causal Diagram: Loan Approval System

```
                    Historical Gender Discrimination
                                  ↓
                               Gender
                                ↙  ↘
                   Wage Gap ↙      ↘ Caregiving
                          ↓           ↓
                      Income    Employment Gaps
                          ↓           ↓
                    Debt-to-    Employment
                    Income      History Score
                    Ratio             ↓
                          ↘         ↙
                        Credit Score
                              ↓
                       Loan Approval


    Proxy Path:
    Gender → Zip Code (due to housing discrimination)
            Zip Code → Loan Approval
```

**Key Insights from This Diagram**:
1. **Direct path**: None (gender not directly used)
2. **Mediator paths**: Gender → Income → Credit Score → Approval
3. **Proxy path**: Gender → Zip Code → Approval (unmeasured: housing discrimination)
4. **Selection bias**: Historical lending data reflects discriminatory practices

---

### Step 2: Classify Each Pathway

For each path from protected attribute to outcome, determine its type:

#### Classification Framework

**For each pathway, ask**:

**Q1: Does the protected attribute directly influence the outcome?**
- YES → **Direct Discrimination** (highest priority)
- NO → Continue to Q2

**Q2: Is there a feature that correlates with the protected attribute?**
- YES → Continue to Q2a
- NO → Check for selection bias

**Q2a: Does the feature have a legitimate causal relationship with the outcome?**
- YES → Check if it's a **Mediator** (Q2b)
- NO → **Proxy Discrimination** (remove or transform)

**Q2b: Is the feature causally influenced by the protected attribute?**
- YES → **Mediator Discrimination** (complex trade-off)
- NO → Legitimate predictor (keep it)

**Q3: Does training data reflect historical discrimination?**
- YES → **Selection Bias** (reweight or collect new data)

#### Worked Example: Loan Approval

**Pathway 1: Gender → Loan Approval**
- Q1: Direct influence? NO (gender not a feature)
- Classification: ✓ No direct discrimination

**Pathway 2: Gender → Income → Approval**
- Q2: Correlated feature? YES (income)
- Q2a: Legitimate causal link? YES (income → repayment ability)
- Q2b: Protected attribute influences feature? YES (gender wage gap)
- Classification: ⚠️ **Mediator Discrimination**
- Action: Use income but add fairness constraint

**Pathway 3: Gender → Zip Code → Approval**
- Q2: Correlated feature? YES (zip code)
- Q2a: Legitimate causal link? PARTIAL (contains both economic signal and discrimination)
- Domain expert: "Zip code proxies for race/gender due to housing discrimination"
- Classification: ⚠️ **Proxy Discrimination** (remove or transform)

**Pathway 4: Historical data from 1980-2000**
- Q3: Training data reflects discrimination? YES (pre-Fair Housing Act)
- Classification: ⚠️ **Selection Bias**
- Action: Reweight recent data, collect new samples

---

### Step 3: Quantify Each Pathway's Contribution

Estimate how much each mechanism contributes to the overall disparity.

#### 3.1 Calculate Overall Disparity

**For binary outcomes** (approved/denied):
```
Overall Gap = Approval_Rate(Male) - Approval_Rate(Female)
Example: 76% - 58% = 18% gap
```

**For continuous outcomes** (risk scores):
```
Overall Gap = Mean_Score(Male) - Mean_Score(Female)
Example: 650 - 580 = 70 point gap
```

#### 3.2 Decompose the Gap by Pathway

Use **path-specific effects** to isolate each mechanism's contribution:

**Method: Controlled Experiments (Thought Experiments)**

For each pathway, imagine an intervention:

**Pathway: Gender → Income → Approval**

*Experiment*: What if we equalized income across genders, keeping everything else the same?

1. Take all female applicants
2. Replace their income with the male income distribution (matched on other features)
3. Re-run the model
4. Calculate new gap

```
Original gap: 18%
After equalizing income: 10%
Pathway contribution: 18% - 10% = 8% (44% of total gap)
```

**Pathway: Gender → Zip Code → Approval**

*Experiment*: What if we removed zip code from the model?

1. Retrain model without zip code
2. Calculate new gap

```
Gap with zip code: 18%
Gap without zip code: 15%
Pathway contribution: 18% - 15% = 3% (17% of total gap)
```

**Selection Bias**

*Experiment*: What if we used only recent data (2020-2024)?

```
Gap with full historical data: 18%
Gap with recent data only: 11%
Selection bias contribution: 18% - 11% = 7% (39% of total gap)
```

#### 3.3 Summary Table

| Mechanism | Pathway | Contribution | % of Total Gap | Priority |
|-----------|---------|--------------|----------------|----------|
| Direct | Gender → Approval | 0% | 0% | N/A |
| Proxy | Gender → Zip Code → Approval | 3% | 17% | High |
| Mediator | Gender → Income → Approval | 8% | 44% | Complex |
| Selection | Historical discrimination in data | 7% | 39% | High |
| **Total** | - | **18%** | **100%** | - |

---

### Step 4: Counterfactual Analysis (Individual Fairness)

Beyond group-level disparities, examine individual-level fairness: Would a specific person's outcome change if only their protected attribute differed?

#### 4.1 Define the Counterfactual Query

**Standard Form**: "What would happen to Individual X if we changed their protected attribute from A to B, holding everything else equal?"

**Example Queries**:
- "Would Applicant #47 (female) be approved if she were male, with all other characteristics unchanged?"
- "Would Candidate #183 (Black) receive a job offer if they were white, keeping qualifications identical?"

#### 4.2 Identify Affected Variables

When changing a protected attribute in a counterfactual, which other variables must change?

**Two Types of Variables**:

**Type 1: Causally Dependent** (must change with protected attribute)
- Gender → Income: If we flip gender, wage gap implies income would change
- Race → Zip Code: If we flip race, residential segregation patterns suggest zip code might change

**Type 2: Causally Independent** (remain fixed)
- Years of education: Completed education doesn't change if gender flips
- Credit score from 5 years ago: Historical fact remains unchanged

**Constructing the Counterfactual**:
```
Original:
  Applicant: Female, Income: $45K, Zip: 10001, Credit: 680

Counterfactual (Gender → Male):
  Applicant: Male, Income: $52K (adjusted for wage gap),
             Zip: 10001 (independent), Credit: 680 (independent)

Model Prediction:
  Original (Female): DENIED (score: 0.48)
  Counterfactual (Male): APPROVED (score: 0.53)

Conclusion: Gender indirectly caused denial via income pathway
```

#### 4.3 Scale Counterfactual Analysis

For each demographic group, calculate:

**Counterfactual Approval Rate**: % of denied applicants who would be approved if their protected attribute changed

```
Female Denials: 420 applicants
Counterfactual (as male): 180 would be approved (43%)

Conclusion: 43% of female denials are attributable to gender
           via causal pathways (primarily income/employment)
```

**Intervention Targeting**:
- High counterfactual flip rate (>30%) → Strong evidence of discrimination → High priority
- Low counterfactual flip rate (<10%) → Other factors dominate → Lower priority

---

### Step 5: Generate Intervention Recommendations

Based on causal analysis, create a prioritized intervention plan.

#### 5.1 Match Mechanisms to Interventions

| Mechanism Detected | Recommended Intervention | Component | Effort | Expected Impact |
|-------------------|-------------------------|-----------|--------|-----------------|
| Direct Discrimination | Remove protected attribute, Add constraints | Pre/In-Processing | Low | 100% of direct path |
| Proxy Discrimination | Feature transformation, Adversarial debiasing | Pre/In-Processing | Medium | 60-80% of proxy path |
| Mediator Discrimination | Constrained optimization, Multi-objective training | In-Processing | High | 40-60% of mediator path |
| Selection Bias | Reweighting, New data collection | Pre-Processing | Medium-High | 70-90% of selection bias |

#### 5.2 Prioritization Framework

**Priority Scoring** = (Gap Contribution) × (Legal Risk) × (Intervention Feasibility)

**Legal Risk**:
- Direct discrimination: 10 (illegal, immediate liability)
- Proxy discrimination: 7 (disparate impact, regulatory scrutiny)
- Mediator discrimination: 4 (complex, depends on context)
- Selection bias: 6 (systemic, affects all decisions)

**Intervention Feasibility**:
- Post-processing (deployed model): 10 (days to implement)
- Pre-processing (data available): 7 (weeks to implement)
- In-processing (can retrain): 5 (months to implement)
- Data collection (new data needed): 3 (years to implement)

**Example Scoring**:
```
Proxy Pathway (Zip Code):
  Gap Contribution: 3%
  Legal Risk: 7
  Feasibility (Pre-processing): 7
  Priority Score: 3 × 7 × 7 = 147

Mediator Pathway (Income):
  Gap Contribution: 8%
  Legal Risk: 4
  Feasibility (In-processing): 5
  Priority Score: 8 × 4 × 5 = 160 (highest priority)

Selection Bias:
  Gap Contribution: 7%
  Legal Risk: 6
  Feasibility (Reweighting): 7
  Priority Score: 7 × 6 × 7 = 294 (critical priority)
```

#### 5.3 Create Intervention Roadmap

**Immediate Actions (Week 1-2)**:
1. Address selection bias: Reweight training data
2. Remove/transform proxy features (zip code)

**Short-term (Month 1-3)**:
- If model deployed: Apply post-processing (threshold optimization)
- Expected: 40-60% gap reduction

**Medium-term (Month 3-6)**:
- When retraining allowed: Implement constrained optimization for mediator pathways
- Expected: Additional 20-40% gap reduction

**Long-term (Year 1-2)**:
- Collect new representative data
- Address root causes (partner with HR on pay equity, etc.)
- Expected: 80-95% total gap reduction

---

## Outputs and Documentation

### Deliverable 1: Causal Analysis Report

**Executive Summary** (1 page):
- Overall disparity magnitude
- Top 3 causal mechanisms identified
- Recommended intervention priorities
- Expected outcomes and timeline

**Causal Diagram** (visual):
- Complete DAG with all relevant pathways
- Color-coded by mechanism type
- Annotated with contribution percentages

**Detailed Findings** (3-5 pages):
- Each pathway analyzed separately
- Quantified contributions
- Supporting evidence (statistical tests, domain expertise)
- Uncertainty acknowledgment

**Intervention Recommendations** (2 pages):
- Prioritized intervention list
- Effort estimates and expected impact
- Risks and trade-offs
- Success criteria

### Deliverable 2: Counterfactual Analysis

**Individual-Level Results**:
- Number and % of individuals affected by each pathway
- Distribution of counterfactual prediction changes
- Identification of most vulnerable subgroups

**Case Examples** (5-10 anonymized examples):
```
Case #1: Female, Age 28, Income $43K
  Actual: DENIED (score: 0.46)
  Counterfactual (as male): APPROVED (score: 0.54)
  Primary pathway: Gender → Income (wage gap) → Approval
  Intervention: Income-blind thresholds or equalized opportunity
```

### Deliverable 3: Intervention Selector Output

Structured recommendation for next steps:

```yaml
intervention_plan:
  immediate:
    - component: preprocessing
      technique: sample_reweighting
      target: selection_bias
      effort_weeks: 2
      expected_gap_reduction: 35-45%
      
  short_term:
    - component: postprocessing
      technique: threshold_optimization
      target: mediator_pathways
      effort_weeks: 3
      expected_gap_reduction: additional 25-35%
      
  medium_term:
    - component: inprocessing
      technique: constrained_optimization
      target: all_pathways
      effort_weeks: 12
      expected_gap_reduction: total 70-85%
```

---

## Best Practices and Common Pitfalls

### Best Practices

1. **Involve Domain Experts Early**
   - Data scientists identify correlations
   - Domain experts explain causal mechanisms
   - Together they build accurate causal models

2. **Document Assumptions Explicitly**
   - "We assume income causally influences repayment (confirmed by economics literature)"
   - "We assume zip code does NOT causally influence creditworthiness (confirmed by domain experts)"

3. **Validate with Multiple Data Sources**
   - Historical trends support causal claim?
   - External studies confirm mechanism?
   - Quasi-experimental evidence available?

4. **Consider Multiple Plausible Models**
   - Construct 2-3 alternative causal graphs
   - Test intervention recommendations' robustness across models
   - Document which interventions work regardless of uncertainty

5. **Revisit Analysis Quarterly**
   - Causal relationships change over time
   - New data may reveal previously hidden pathways
   - Regulations evolve, changing what's considered discriminatory

### Common Pitfalls

**Pitfall 1: Confusing Correlation with Causation**
- ❌ "Zip code correlates with defaults, so it must cause them"
- ✓ "Zip code correlates with defaults, but this likely reflects unmeasured socioeconomic factors"

**Pitfall 2: Ignoring Unmeasured Confounders**
- ❌ Drawing arrow: Education → Income (direct)
- ✓ Recognizing: Family wealth → Education AND Income (confounder)

**Pitfall 3: Over-Simplifying Mediator Decisions**
- ❌ "Income is a mediator, so remove it entirely"
- ✓ "Income is a mediator with legitimate predictive value; use constrained optimization"

**Pitfall 4: Neglecting Intersectionality**
- ❌ Analyzing gender and race separately
- ✓ Examining gender × race interactions (young Black women face unique pathways)

**Pitfall 5: Treating Causal Analysis as One-Time**
- ❌ "We did causal analysis in 2023, we're good"
- ✓ "We review our causal model quarterly as data and context evolve"

---

## Integration with Other Components

The Causal Fairness Toolkit feeds into all subsequent interventions:

**To Pre-Processing Toolkit**:
- Proxy pathways identified → Feature transformation needed
- Selection bias quantified → Reweighting parameters determined

**To In-Processing Toolkit**:
- Mediator pathways mapped → Constraints specified
- Trade-off magnitude estimated → Lambda parameter range suggested

**To Post-Processing Toolkit**:
- Counterfactual flip rates calculated → Threshold adjustments targeted
- Group-specific mechanisms → Calibration approach selected

**From Validation Framework**:
- Intervention effectiveness measured → Causal model validated
- Unexpected results → Causal model revisited and refined

---

## Summary

The Causal Fairness Toolkit provides the diagnostic foundation for all fairness interventions. By understanding the causal mechanisms creating disparities—direct discrimination, proxy discrimination, mediator discrimination, and selection bias—teams can target interventions effectively rather than applying generic fixes.

**Key Takeaways**:
1. Causal analysis is non-negotiable; skipping it leads to 3x higher intervention failure rates
2. Four mechanism types require different intervention approaches
3. Quantifying each pathway's contribution enables optimal resource allocation
4. Counterfactual analysis identifies individuals most affected by discrimination
5. Causal insights directly inform pre-processing, in-processing, and post-processing strategies

**Time Investment**: 3-5 days of analysis saves 2-3 weeks of trial-and-error interventions.



