# 2. Causal Fairness Toolkit

Causal approaches go beyond traditional metrics by analyzing **why** bias occurs, using tools like graphs and counterfactuals.

## Key Concepts

- **Causal Graphs**: Show how features, including protected attributes, causally affect outcomes.
  - *Example paths*: Direct discrimination (gender → loan_approval), proxy discrimination (zip → loan_approval, with zip correlated to race/gender).
- **Counterfactual Reasoning**: Would the decision change if a protected attribute changed, holding everything else constant?

## Process

### 2.1. Map Out Causal Relationships

- Draw a Directed Acyclic Graph (DAG) of your data pipeline.
- Identify direct, indirect, and proxy paths from protected attributes to outcomes.

### 2.2. Identify Intervention Points

- Assess which paths are valid, and which constitute unfairness.
- Plan interventions based on *where* the unfairness originates:
  - Data? → Pre-Processing
  - Model logic? → In-Processing
  - Outputs? → Post-Processing

### 2.3. Example Questions

- How could changing gender/race impact the decision, even if all other features are held constant?
- Does an apparently neutral feature (like zip code) act as a proxy for protected attributes?

---

**Resources**: NetworkX for DAGs, doWhy, CausalNex (Python libraries for causal analysis)
