# Lab 4: Statistical Analysis - Descriptive Statistics and Probability Distributions

**Objective:** In this lab, you will:

- Apply descriptive statistics to analyze engineering data (concrete strength, structural loads, material properties).
- Calculate and interpret measures of central tendency, spread, and shape.
- Work with probability distributions (discrete and continuous) relevant to engineering problems.
- Visualize distributions and interpret statistical properties.
- Practice using Python libraries (NumPy, Pandas, SciPy, Matplotlib, Seaborn) for statistical analysis.
- Apply probability concepts to engineering decision-making scenarios.

---

## 1. Overview

This lab focuses on statistical analysis methods essential for civil engineering practice. You will work with real-world engineering datasets, compute descriptive statistics, explore probability distributions, and apply statistical reasoning to engineering problems. The lab covers both theoretical concepts and practical implementation using Python.

**Course Repository:** [https://github.com/merenozcetin/CE49X-Fall25.git](https://github.com/merenozcetin/CE49X-Fall25.git)

---

## 2. Learning Objectives

By completing this lab, you will:

- Understand and compute measures of central tendency (mean, median, mode) and when to use each.
- Calculate and interpret measures of spread (variance, standard deviation, IQR).
- Analyze distribution shape using skewness and kurtosis.
- Work with discrete distributions (Bernoulli, Binomial, Poisson).
- Work with continuous distributions (Uniform, Normal, Exponential).
- Apply probability concepts (conditional probability, Bayes' theorem) to engineering scenarios.
- Visualize distributions and create publication-quality statistical plots.
- Use Python libraries for statistical computations and analysis.

---

## 3. Prerequisites

Before starting, ensure that you have:

- Completed Labs 1-3 and have a working Python environment.
- Python 3.10 or later installed.
- Required Python libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scipy
  ```
- A basic understanding of probability and statistics concepts from Lecture 6.
- A local Git repository for your course work.

---

## 4. Datasets

For this lab, you will work with synthetic engineering datasets that simulate real-world scenarios. All datasets are located in the **`datasets/` folder at the root level of the course repository**:

1. **Concrete Strength Data** (`../datasets/concrete_strength.csv`):

   - Contains compressive strength measurements (MPa) from multiple concrete batches
   - Includes batch ID, age (days), mix type, and strength values
   - Used for descriptive statistics and distribution analysis
   - File contains 100 samples across multiple batches and mix types
2. **Structural Load Data** (`../datasets/structural_loads.csv`):

   - Contains load measurements from structural monitoring
   - Includes time stamps, load values (kN), and component types
   - Used for probability analysis and distribution fitting
   - File contains 200 hourly measurements from structural components
3. **Material Property Data** (`../datasets/material_properties.csv`):

   - Contains yield strength measurements for different materials
   - Includes material type, test number, and strength values (MPa)
   - Used for comparative statistical analysis
   - File contains 200 samples (50 per material: Steel, Concrete, Aluminum, Composite)

---

## 5. Instructions

### 5.1. Create the Python Script

Create a new file named `lab4_statistical_analysis.py` in the `/labs/lab4/` directory.

### 5.2. Script Requirements

Your script should include the following functionality:

#### Part 1: Descriptive Statistics (40 points)

1. **Data Loading and Exploration:**

   - Load the concrete strength dataset from the root-level `datasets/` folder.
   - The `load_data()` function automatically handles the path to the datasets folder at the repository root.
   - Display basic information (shape, columns, data types).
   - Handle missing values appropriately.
   - Display first few rows and summary statistics.
   - Load other datasets from the root-level `datasets/` folder as needed for different tasks.
2. **Measures of Central Tendency:**

   - Calculate mean, median, and mode for concrete strength.
   - Compare these measures and explain when each is appropriate.
   - Visualize the distribution with mean, median, and mode marked.
3. **Measures of Spread:**

   - Calculate variance, standard deviation, range, and IQR.
   - Interpret these measures in engineering context.
   - Create a visualization showing the distribution with ±1σ, ±2σ, ±3σ marked.
4. **Shape Measures:**

   - Calculate skewness and kurtosis.
   - Interpret the values (symmetric, skewed, heavy-tailed, etc.).
   - Visualize the distribution shape using histograms and density plots.
5. **Quantiles and Percentiles:**

   - Calculate Q1, Q2 (median), Q3.
   - Create a five-number summary (Min, Q1, Median, Q3, Max).
   - Create boxplots showing quartiles and outliers.

#### Part 2: Probability Distributions (35 points)

6. **Discrete Distributions:**

   - **Bernoulli:** Model component pass/fail scenarios (e.g., quality control).
   - **Binomial:** Model number of defective items in a batch.
   - **Poisson:** Model number of events (e.g., truck arrivals, defects per unit).
   - For each distribution:
     - Generate random samples.
     - Plot PMF and CDF.
     - Calculate mean and variance.
     - Apply to a real engineering scenario.
7. **Continuous Distributions:**

   - **Uniform:** Model random positions or uniformly distributed loads.
   - **Normal:** Model material strengths, loads (central limit theorem).
   - **Exponential:** Model time until failure, waiting times.
   - For each distribution:
     - Generate random samples.
     - Plot PDF and CDF.
     - Calculate mean and variance.
     - Apply to a real engineering scenario.
8. **Distribution Fitting:**

   - Fit a normal distribution to concrete strength data.
   - Visualize fitted distribution overlaid on histogram.
   - Compare fitted distribution parameters with sample statistics.

#### Part 3: Probability Applications (25 points)

9. **Conditional Probability:**

   - Solve an engineering problem involving conditional probability.
   - Example: Probability of failure given a defect is detected.
   - Use probability trees to visualize the problem.
10. **Bayes' Theorem:**

    - Apply Bayes' theorem to an engineering diagnostic problem.
    - Example: Probability of structural damage given a positive test result.
    - Interpret prior, likelihood, and posterior probabilities.
11. **Basic Comparison:**

    - Compare distributions across different groups (e.g., material strength by material type).
    - Use descriptive statistics and visual comparisons.

#### Part 4: Visualization and Reporting (20 points)

12. **Comprehensive Visualizations:**

    - Create at least 5 meaningful visualizations:
      - Histogram with overlayed normal distribution
      - Boxplots comparing multiple groups
      - Probability distribution comparisons (PDF/CDF plots)
      - Statistical summary dashboard
      - Probability tree diagram (for Bayes' theorem)
13. **Statistical Report:**

    - Generate a summary report with:
      - Descriptive statistics table
      - Distribution parameters
      - Key findings and interpretations
      - Engineering implications

#### Part 5: Code Quality (10 points)

14. **Modular Code:**

    - Organize code into functions with clear purposes.
    - Add appropriate comments and docstrings.
    - Include error handling for file operations and data processing.
    - Follow PEP 8 style guidelines.
15. **Function Requirements:**
    Create at least the following functions:

    ```python
    def load_data(file_path)
    def calculate_descriptive_stats(data, column='strength_mpa')
    def plot_distribution(data, column, title, save_path=None)
    def fit_distribution(data, column, distribution_type='normal')
    def calculate_probability_binomial(n, p, k)
    def calculate_probability_normal(mean, std, x_lower=None, x_upper=None)
    def calculate_probability_poisson(lambda_param, k)
    def calculate_probability_exponential(mean, x)
    def apply_bayes_theorem(prior, sensitivity, specificity)
    def plot_material_comparison(data, column, group_column, save_path=None)
    def plot_distribution_fitting(data, column, fitted_dist=None, save_path=None)
    def create_statistical_report(data, output_file='lab4_statistical_report.txt')
    ```

---

## 6. Detailed Tasks

### Task 1: Concrete Strength Analysis

Analyze concrete strength data:

- Calculate all descriptive statistics (mean, median, mode, std, variance, skewness, kurtosis).
- Create visualizations:
  - Histogram with normal curve overlay
  - Boxplot showing quartiles
- Assess the shape of the distribution (symmetric, skewed, kurtosis).
- Interpret results in engineering context (e.g., quality control, design specifications).

### Task 2: Material Comparison

Compare strength distributions across different materials:

- Load material property data from the root-level `datasets/` folder.
- Calculate statistics for each material type.
- Create comparative boxplots.
- Compare means and standard deviations across materials.
- Interpret which material has higher variability.

### Task 3: Probability Modeling

Model engineering scenarios using probability distributions:

1. **Binomial Scenario:**

   - Quality control: 100 components tested, 5% defect rate.
   - What is probability of exactly 3 defective components?
   - What is probability of ≤ 5 defective components?
2. **Poisson Scenario:**

   - Bridge load events: Average 10 heavy trucks per hour.
   - What is probability of exactly 8 trucks in an hour?
   - What is probability of > 15 trucks in an hour?
3. **Normal Scenario:**

   - Steel yield strength: Mean = 250 MPa, Std = 15 MPa.
   - What percentage exceeds 280 MPa?
   - What is the 95th percentile?
4. **Exponential Scenario:**

   - Component lifetime: Mean = 1000 hours.
   - What is probability of failure before 500 hours?
   - What is probability of surviving beyond 1500 hours?

### Task 4: Bayes' Theorem Application

**Scenario:** Structural damage detection

- Base rate: 5% of structures have damage
- Test sensitivity: 95% (detects damage when present)
- Test specificity: 90% (correctly identifies no damage)
- If test is positive, what is probability of actual damage?
- Visualize using probability tree.
- Discuss implications for engineering decision-making.

### Task 5: Distribution Fitting and Validation

1. Fit a normal distribution to concrete strength data:

   - Estimate distribution parameters (mean and standard deviation) from the data.
   - Compare fitted distribution parameters with sample statistics.
   - Visualize fitted distribution overlaid on histogram.
2. Generate synthetic data from fitted normal distribution:

   - Compare synthetic vs. real data visually (histogram overlay).
   - Validate that mean and standard deviation of synthetic data match the fitted parameters.

---

## 7. Expected Outputs

Your script should produce:

1. **Console Output:**

   - Summary statistics tables
   - Distribution parameters
   - Probability calculations
   - Comparative statistics across groups
2. **Visualizations (save as PNG files):**

   - `concrete_strength_distribution.png`
   - `material_comparison_boxplot.png`
   - `probability_distributions.png`
   - `distribution_fitting.png`
   - `statistical_summary_dashboard.png`
3. **Report File:**

   - `lab4_statistical_report.txt` or `lab4_statistical_report.md`
   - Contains summary statistics, interpretations, and findings

---

## 8. Sample Code Structure

```python
"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon

def load_data(file_path):
    """Load dataset from CSV file in root-level datasets/ folder."""
    # Example: load_data('concrete_strength.csv') will load from 
    #          root-level datasets/concrete_strength.csv
    #          The function automatically navigates from labs/lab4/ to ../datasets/
    pass

def calculate_descriptive_stats(data):
    """Calculate all descriptive statistics."""
    pass

def plot_distribution(data, title, save_path=None):
    """Create distribution plot with statistics."""
    pass

def fit_distribution(data, distribution_type='normal'):
    """Fit probability distribution to data."""
    pass

def calculate_probability_binomial(n, p, k):
    """Calculate binomial probabilities."""
    pass

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """Calculate normal probabilities."""
    pass

def calculate_probability_poisson(lambda_param, k):
    """Calculate Poisson probabilities."""
    pass

def calculate_probability_exponential(mean, x):
    """Calculate exponential probabilities."""
    pass

def apply_bayes_theorem(prior, sensitivity, specificity):
    """Apply Bayes' theorem for diagnostic test scenario."""
    pass

def plot_material_comparison(data, column, group_column, save_path=None):
    """Create comparative boxplot for material types."""
    pass

def plot_distribution_fitting(data, column, fitted_dist=None, save_path=None):
    """Visualize fitted distribution with synthetic data comparison."""
    pass

def create_statistical_report(data, output_file='lab4_statistical_report.txt'):
    """Create a statistical report summarizing findings."""
    pass

def main():
    """Main execution function."""
    # Load data
    # Perform analyses
    # Generate visualizations
    # Create report
  
if __name__ == "__main__":
    main()
```

---

## 9. Running Your Script

From the `/labs/lab4/` directory, run your script by executing:

```bash
python lab4_statistical_analysis.py
```

Ensure that the script runs without errors and produces all required outputs.

---

## 10. Version Control: Committing and Pushing Your Code

Once you have completed the lab, follow these steps to push your work:

**Note:** Make sure you are working in the course repository: [https://github.com/merenozcetin/CE49X-Fall25.git](https://github.com/merenozcetin/CE49X-Fall25.git)

1. **Check your repository status:**

   ```bash
   git status
   ```
2. **Stage your changes:**

   ```bash
   git add labs/lab4/lab4_statistical_analysis.py
   git add datasets/*.csv
   git add labs/lab4/*.png
   git add labs/lab4/lab4_statistical_report.*
   ```
3. **Commit your changes with a descriptive message:**

   ```bash
   git commit -m "Lab 4: Add statistical analysis - descriptive statistics and probability distributions"
   ```
4. **Push your changes to your repository:**

   ```bash
   git push origin main
   ```

   > **Note:** If you are working on a separate branch, push that branch instead.
   >

---

## 11. Submission Instructions

When you have completed Lab 4, send an email to **eyuphan.koc@bogazici.edu.tr (also cc mustafa.ozcetin@std.bogazici.edu.tr)** with the subject line:

```
Name, LastName, Lab 4 Completed
```

Include the following in your email:

- A link to your GitHub repository
- A brief summary (5-7 sentences) covering:
  - Key findings from descriptive statistics analysis
  - Insights from probability distribution analysis
  - Most interesting probability application (Bayes' theorem or other)
  - Engineering implications of your statistical findings
- Answer the following questions:
  1. Which measure of central tendency is most appropriate for your concrete strength data? Why?
  2. Does your data follow a normal distribution? How did you determine this?
  3. What are the engineering implications of your probability calculations?

---

## 12. Grading Rubric

| Component                 | Points | Criteria                                                  |
| ------------------------- | ------ | --------------------------------------------------------- |
| Descriptive Statistics    | 25     | All measures calculated correctly, proper interpretation  |
| Visualizations            | 15     | Quality plots, appropriate choices, clear labels          |
| Probability Distributions | 20     | Correct application, proper calculations, interpretations |
| Bayes' Theorem            | 10     | Correct application, clear explanation                    |
| Code Quality              | 10     | Modular, documented, follows PEP 8                        |
| Distribution Fitting      | 10     | Proper fitting, validation, interpretation                |
| Report                    | 10     | Clear summary, engineering insights, professional format  |

**Total: 100 points**

---

## 13. Additional Resources

- **SciPy Statistical Functions:** [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- **Pandas Statistical Functions:** [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
- **Matplotlib Visualization:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- **Seaborn Statistical Visualization:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- **Probability Distributions Reference:** [https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
- **Engineering Statistics Textbook:** Recommended readings from course materials

---

## 14. Tips and Hints

1. **Start with Data Exploration:** Always begin by exploring your data - check for missing values, outliers, and data types.
2. **Use Appropriate Distributions:** Choose distributions based on the problem context:

   - Counts → Binomial, Poisson
   - Measurements → Normal, Uniform
   - Times → Exponential
3. **Visualize Before Calculating:** Visualizations help you understand the data and choose appropriate statistical methods.
4. **Interpret Results:** Always provide engineering interpretations, not just numerical results.
5. **Understand Distribution Shape:** Assess skewness and kurtosis to understand the distribution shape before choosing appropriate statistical methods.
6. **Use SciPy:** The `scipy.stats` module provides many useful functions for statistical analysis.
7. **Document Your Code:** Clear comments help you and others understand your analysis.

---

## 15. Challenge Problems (Optional, Extra Credit)

For students seeking additional challenge:

1. **Bootstrap Confidence Intervals:** Use bootstrap resampling to estimate confidence intervals for mean and standard deviation.
2. **Monte Carlo Simulation:** Simulate structural reliability using probability distributions.
3. **Advanced Distribution Fitting:** Fit mixture distributions (e.g., bimodal normal) to identify multiple populations.
4. **Bayesian Analysis:** Implement Bayesian parameter estimation for distribution fitting.
5. **Statistical Process Control:** Create control charts for quality monitoring.

---

Good luck with Lab 4! If you have any questions or encounter issues, please post on the class discussion board or reach out to the instructor.
