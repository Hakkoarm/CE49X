# CE49X – Lab 5: Bias–Variance Tradeoff using the Air Quality Dataset

**Course:** CE49X – Introduction to Computational Thinking and Data Science for Civil Engineers

**Instructor:** Dr. Eyuphan Koç

**Semester:** Fall 2025

**Lab Duration:** 1 week

**Dataset:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

---

## 🎯 Learning Objectives

By completing this lab, you will:

- Understand the **bias–variance tradeoff** in machine learning.
- Implement and compare **linear** and **polynomial regression** models.
- Visualize **training** and **testing errors** as model complexity changes.
- Interpret **underfitting** and **overfitting** phenomena using real environmental data.

---

## 🧠 Background

The **bias–variance tradeoff** describes how increasing model complexity affects prediction performance:

- **High bias (underfitting):** Model is too simple → misses key patterns.
- **High variance (overfitting):** Model is too complex → fits noise rather than signal.
- The goal is to find the **sweet spot** with minimum test error.

In this lab, you will explore this concept using air quality measurements from a monitoring station in Italy.

---

## 🧩 Dataset Description

**Source:** UCI Machine Learning Repository – Air Quality Dataset

**URL:** [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)

**Description:**

- Contains hourly responses from a gas multisensor device.
- Records air pollutants (CO, NOx, NO2) and meteorological variables.
- We will predict **CO(GT)** (true CO concentration in mg/m³) from other features.

**Key columns:**

| Variable    | Description               | Unit      |
| ----------- | ------------------------- | --------- |
| CO(GT)      | True CO concentration     | mg/m³    |
| PT08.S1(CO) | Tin oxide sensor response | arbitrary |
| NMHC(GT)    | Non-methane hydrocarbons  | µg/m³   |
| NOx(GT)     | Nitrogen oxides           | ppb       |
| NO2(GT)     | Nitrogen dioxide          | µg/m³   |
| T           | Temperature               | °C       |
| RH          | Relative Humidity         | %         |
| AH          | Absolute Humidity         | g/m³     |

---

## ⚙️ Instructions

### Step 1 — Load and Prepare the Data

1. Download the dataset (`AirQualityUCI.csv`) from the UCI page.
2. Load it using **pandas**.
3. Handle missing values (`-200` indicates missing data).
4. Select the following columns for modeling:

   ```python
   features = ['T', 'RH', 'AH']
   target = 'CO(GT)'
   ```
5. Split into training (70%) and testing (30%) sets using `train_test_split`.

### Step 2 — Fit Models of Increasing Complexity

Start with a Linear Regression model:

```python
from sklearn.linear_model import LinearRegression
```

Create Polynomial Regression models of degree 1 to 10 using:

```python
from sklearn.preprocessing import PolynomialFeatures
```

For each degree:

1. Transform features using `PolynomialFeatures(degree=d)`.
2. Train a `LinearRegression()` model.
3. Compute training error and testing error (e.g., MSE or RMSE).

### Step 3 — Plot the Validation Curve

Plot model degree (x-axis) vs. error (y-axis).

Include both training and testing curves:

```python
import matplotlib.pyplot as plt

plt.plot(degrees, train_errors, label='Training Error')
plt.plot(degrees, test_errors, label='Testing Error')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Bias–Variance Tradeoff')
plt.show()
```

Label the regions of underfitting, optimal complexity, and overfitting.

### Step 4 — Discussion

Answer the following in your report:

1. Which polynomial degree gives the best generalization?
2. Describe how the training and testing errors change as degree increases.
3. Explain how bias and variance manifest in this dataset.
4. How might sensor noise or missing data affect the bias–variance tradeoff?

---

## 🧾 Deliverables

- **Jupyter Notebook** (`Lab5_BiasVariance.ipynb`)

Include:

- Code cells (with comments)
- Plots
- Answers to discussion questions

---

## 💡 Bonus (Optional)

Try using cross-validation (`cross_val_score`) instead of a simple train/test split. Compare results and comment on any differences in the optimal model degree.

---

## 🧮 Expected Outcome

You should observe:

- Low-degree models have high bias and poor fit.
- High-degree models have high variance and overfit training data.
- An intermediate degree yields the lowest test error.

---

## 📚 References

- UCI Machine Learning Repository: Air Quality Dataset (De Vito et al., 2008)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Scikit-learn documentation: https://scikit-learn.org/stable/
