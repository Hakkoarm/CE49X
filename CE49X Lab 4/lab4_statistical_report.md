# Lab 4: Statistical Analysis Report

## Task 1: Concrete Strength Analysis

### Descriptive Statistics Table (`strength_mpa`)

| Statistic      | Value     |
|----------------|-----------|
| Count          | 100.0000 |
| Mean           | 35.0600 |
| Median         | 34.2000 |
| Mode           | 31.8000 |
| Std dev        | 6.1627 |
| Variance       | 37.9788 |
| Min            | 21.8000 |
| Max            | 47.1000 |
| Range          | 25.3000 |
| Q1 25th        | 31.1000 |
| Q2 50th        | 34.2000 |
| Q3 75th        | 39.1250 |
| Iqr            | 8.0250 |
| Skewness       | 0.0473 |
| Kurtosis       | -0.6747 |

### Key Findings & Engineering Implications

* **Central Tendency:** The Mean (35.06 MPa) and Median (34.20 MPa) are very close. This suggests the data is symmetric and not skewed by outliers.
* **Shape:** The Skewness (0.0473) is close to 0, confirming the symmetry. This is good for quality control, as it fits assumptions for normal distribution-based process control.
* **Variability:** The Standard Deviation (6.16 MPa) is the key metric for consistency. All design codes (e.g., ACI) use this value to determine the 'specified compressive strength' (f'c) required to meet a target mean strength. A lower std dev means less over-design is needed, saving costs.

## Task 3: Probability Modeling Scenarios

| Scenario       | Question                           | Probability |
|----------------|------------------------------------|-------------|
| Binomial       | P(X = 3)                           | 0.1396 |
| Binomial       | P(X <= 5)                          | 0.6160 |
| Poisson        | P(X = 8)                           | 0.1126 |
| Poisson        | P(X > 15)                          | 0.0487 |
| Normal         | P(X > 280)                         | 0.0228 |
| Normal         | 95th Percentile                    | 274.6728 |
| Exponential    | P(X < 500)                         | 0.3935 |
| Exponential    | P(X > 1500)                        | 0.2231 |

### Engineering Implications

* **Binomial:** The probability of 5 or fewer defects (0.6160) is high. This model allows setting acceptance criteria (e.g., 'reject batch if > 5 defects') with a known probability of error.
* **Normal:** The 95th percentile strength (274.67 MPa) can be used as a characteristic strength for design, representing a value that 95% of the material is expected to exceed.

## Task 4: Bayes' Theorem Application

Scenario: Probability of structural damage given a positive test.

* **Prior Probability (P(Damage)):** 0.05 (5%)
* **Test Sensitivity (P(Pos | Damage)):** 0.95 (95%)
* **Test Specificity (P(Neg | No Damage)):** 0.90 (90%)

**Resulting Posterior Probability (P(Damage | Positive Test)):**
## 0.3333 (or 33.33%)**

### Engineering Implications

This is a critical finding. A highly sensitive test (95%) still produces a high number of false positives when the base rate of the defect is low. A positive test **does not** confirm damage; it only raises the probability from 5% to ~32% (in this specific problem). This implies that a positive test must be followed by more detailed, and likely more expensive, secondary inspections before ordering costly repairs.
