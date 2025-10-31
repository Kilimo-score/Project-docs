# Methodology

This document outlines the methodology behind the Holistic Farmer Trust Scoring (HFTS) system.

## 1. Data Sources

The HFTS system leverages a variety of alternative data sources to build a comprehensive financial profile of smallholder farmers. These include:

*   **Mobile Money Transactions:** M-Pesa transaction history is a primary data source, providing insights into cash flow, bill payment consistency, and overall financial health.
*   **Satellite Imagery:** Satellite data is used to assess farm size, crop health (using NDVI), and environmental factors like drought risk.
*   **Cooperative Records (SACCO):** Data from Savings and Credit Co-operative Organizations (SACCOs) provides information on institutional credit history and savings behavior.
*   **Farm Metrics:** Self-reported and verified data on farm efficiency, yield, and income.

## 2. Feature Engineering

A three-tiered feature hierarchy is used to weigh the importance of different data points. This ensures that the most predictive and reliable indicators have the greatest impact on the final score.

*   **Tier 1: High Impact (Weight: 1.00):** These are the most critical features, directly reflecting financial behavior and farm performance.
    *   Bill Payment Consistency
    *   Farm Efficiency Score
    *   M-Pesa Average Balance
*   **Tier 2: Medium Impact (Weight: 0.67):** These features provide important context and secondary validation.
    *   SACCO Institutional Score
    *   NDVI Crop Health Index
    *   Drought Risk Probability
    *   Account Age
*   **Tier 3: Low Impact (Weight: 0.33):** These features offer additional data points but are less predictive on their own.
    *   Farm Size
    *   Land Value
    *   Net Income

## 3. Modeling

### 3.1. Model Selection

A LightGBM (Light Gradient Boosting Machine) model is used for its high performance, efficiency, and ability to handle large datasets. Gradient boosting models are well-suited for classification tasks and are known for their accuracy.

### 3.2. Training Data

The model is trained on a synthetically generated dataset of 10,000 farmer profiles. This data is designed to represent a realistic distribution of farmer archetypes, from "Excellent" to "High Risk," each with a corresponding default probability.

### 3.3. Default Probability Calculation

The default probability is calculated using a weighted scoring function that combines the base archetype score with the impact of individual features. Gaussian noise is added to simulate real-world variance.

```
default_probability = sigmoid(base_archetype_score + Σ(feature_i × weight_i × normalization_i) + noise)
```

## 4. Risk Classification

The calculated default probability is used to classify farmers into one of four risk categories:

*   **Very Low Risk (< 10% probability):** Auto-approved for larger loans with favorable terms.
*   **Low Risk (10-25% probability):** Auto-approved for moderate loans.
*   **Medium Risk (25-50% probability):** Flagged for manual review, eligible for smaller loans.
*   **High Risk (> 50% probability):** Denied, but may be eligible for very small, short-term loans.

## 5. Explainability

The HFTS system incorporates SHAP (SHapley Additive exPlanations) to provide explainable AI. This allows for the analysis of feature importance, making the model's decisions transparent and interpretable for loan officers and auditors.
