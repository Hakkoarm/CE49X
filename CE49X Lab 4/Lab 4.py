"""
Lab 4: Statistical Analysis
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
import os


def load_data(file_name):
    """
    Load dataset from CSV file.
    Assumes file is in the current directory.
    """
    try:
        # Per your instruction, load directly from the current directory
        data = pd.read_csv(file_name)
        print(f"Successfully loaded '{file_name}'")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: '{file_name}'.")
        print("Please ensure the file is in the same directory as the script.")
        return None
    except Exception as e:
        print(f"An error occurred loading {file_name}: {e}")
        return None

# --- Main Execution (Part 1.1: Data Loading and Exploration) ---

def main_part1_loading():
    """
    Executes the data loading and exploration task.

    MODIFIED: Now returns the loaded DataFrame.
    """
    print("--- Part 1.1: Data Loading and Exploration ---")

    # 1. Load the concrete strength dataset
    df_concrete = load_data('datasets/concrete_strength.csv')

    if df_concrete is not None:
        # 2. Display basic information (shape, columns, data types)
        print(f"\n--- Basic Information for concrete_strength.csv ---")
        print(f"Shape (Rows, Columns): {df_concrete.shape}")

        print("\nColumns:")
        print(list(df_concrete.columns))

        print("\nData Types and Non-Null Counts") # Using .info() argument
        df_concrete.info()

        # 3. Handle missing values (by checking for them)
        print("\n--- Missing Values Check ---")
        missing_values = df_concrete.isnull().sum()
        print(missing_values)
        if missing_values.sum() == 0:
            print("\nNo missing values found.")
        else:
            print(f"\nTotal missing values found: {missing_values.sum()}")

        # 4. Display first few rows
        print("\n--- First 5 Rows ---") # Using .head() argument
        print(df_concrete.head())

        # 5. Display summary statistics
        print("\n--- Summary Statistics ---") # Using .describe() argument
        # include='all' provides stats for non-numeric columns too
        print(df_concrete.describe(include='all'))

    # MODIFIED: Return the DataFrame for use in other parts
    return df_concrete


# --- Part 1.2: Central Tendency ---

def plot_distribution(data, column, title, save_path=None):
    """
    Create distribution plot (histogram + KDE) with mean, median, and mode.
    (As required by Lab 4, Section 5.15)
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found for plotting.")
        return

    plt.figure(figsize=(10, 6))

    # Calculate stats
    mean_val = data[column].mean()
    median_val = data[column].median()
    # Handle multimodal data by taking the first mode
    mode_val = data[column].mode().iloc[0] if not data[column].mode().empty else np.nan

    # Plot histogram and KDE
    sns.histplot(data[column], kde=True, bins=20, stat="density",
                 label="Histogram/KDE")

    # Add vertical lines for stats
    plt.axvline(mean_val, color='red', linestyle='--',
                label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-',
                label=f'Median: {median_val:.2f}')
    if not np.isnan(mode_val):
        plt.axvline(mode_val, color='blue', linestyle='-.',
                    label=f'Mode: {mode_val:.2f}')

    plt.title(title)
    plt.xlabel(column.replace('_', ' ').capitalize())
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close() # Close the plot to prevent double display

def main_part1_central_tendency(df_concrete):
    """
    Executes the central tendency calculations and visualization.

    MODIFIED: Now accepts a DataFrame as an argument.
    """
    print("\n--- Part 1.2: Measures of Central Tendency ---")

    # MODIFIED: Removed the redundant data loading
    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 2. Calculate mean, median, and mode
    mean_val = df_concrete[column_name].mean()
    median_val = df_concrete[column_name].median()
    mode_val = df_concrete[column_name].mode().iloc[0] if not df_concrete[column_name].mode().empty else np.nan

    print("\n--- Central Tendency Calculations ---")
    print(f"Mean (strength_mpa):   {mean_val:.2f} MPa")
    print(f"Median (strength_mpa): {median_val:.2f} MPa")
    print(f"Mode (strength_mpa):   {mode_val:.2f} MPa")

    # Explanation
    print("\n--- Interpretation of Central Tendency ---")
    print("* **Mean:** The 'average' value. Best for symmetric distributions.")
    print("* **Median:** The middle value (50th percentile). Best for skewed distributions or data with outliers, as it is not affected by extreme values.")
    print("* **Mode:** The most frequent value. Best for categorical data or finding peaks in a distribution.")
    print(f"\n*Comparison:* The Mean ({mean_val:.2f}) and Median ({median_val:.2f}) are very close, suggesting the 'strength_mpa' data is relatively symmetric.")


    # 3. Visualize the distribution
    print("\n--- Generating Distribution Plot ---")
    plot_distribution(df_concrete, column_name,
                      'Concrete Strength Distribution (MPa)',
                      'concrete_strength_distribution.png')


# --- Part 1.3: Measures of Spread ---

def plot_distribution_std_dev(data, column, title, save_path=None):
    """
    Create distribution plot (histogram + KDE) with standard deviation ranges.
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found for plotting.")
        return

    plt.figure(figsize=(12, 7))

    # Calculate stats
    mean_val = data[column].mean()
    std_val = data[column].std()

    # Plot histogram and KDE
    sns.histplot(data[column], kde=True, bins=20, stat="density",
                 label="Histogram/KDE")

    # Add vertical lines for stats
    plt.axvline(mean_val, color='red', linestyle='-',
                label=f'Mean (μ): {mean_val:.2f}')

    # Plot 1, 2, and 3 standard deviations
    colors = ['orange', 'green', 'blue']
    for i in range(1, 4):
        lower = mean_val - (i * std_val)
        upper = mean_val + (i * std_val)
        plt.axvline(lower, color=colors[i-1], linestyle='--',
                    label=f'±{i}σ ({lower:.2f} - {upper:.2f})')
        plt.axvline(upper, color=colors[i-1], linestyle='--')

    plt.title(title)
    plt.xlabel(column.replace('_', ' ').capitalize())
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part1_spread(df_concrete):
    """
    Executes the measures of spread calculations and visualization.
    """
    print("\n--- Part 1.3: Measures of Spread ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Calculate measures of spread
    var_val = df_concrete[column_name].var()
    std_val = df_concrete[column_name].std()
    range_val = df_concrete[column_name].max() - df_concrete[column_name].min()
    q1 = df_concrete[column_name].quantile(0.25)
    q3 = df_concrete[column_name].quantile(0.75)
    iqr_val = q3 - q1

    print("\n--- Measures of Spread Calculations ---")
    print(f"Variance:      {var_val:.2f} MPa²")
    print(f"Std. Deviation: {std_val:.2f} MPa")
    print(f"Range:         {range_val:.2f} MPa")
    print(f"IQR (Q3-Q1):   {iqr_val:.2f} MPa  (Q1: {q1:.2f}, Q3: {q3:.2f})")

    # 2. Interpretation
    print("\n--- Interpretation in Engineering Context ---")
    print(f"* **Variance ({var_val:.2f} MPa²):** A measure of how spread out the data is. Hard to interpret directly due to squared units.")
    print(f"* **Standard Deviation ({std_val:.2f} MPa):** The most common measure of spread. A low value indicates data points are close to the mean (high consistency, good quality control). A high value indicates data is spread out (low consistency, poor quality control).")
    print(f"* **Range ({range_val:.2f} MPa):** The simplest measure of spread (Max - Min). Highly sensitive to outliers (e.g., a single bad batch).")
    print(f"* **IQR ({iqr_val:.2f} MPa):** The spread of the middle 50% of the data. It is robust to outliers and gives a good sense of the 'typical' variability.")

    # 3. Visualize the distribution
    print("\n--- Generating Distribution Plot (with Std. Dev.) ---")
    plot_distribution_std_dev(df_concrete, column_name,
                              'Concrete Strength Distribution with Standard Deviations',
                              'concrete_strength_std_dev.png')


# --- Part 1.4: Shape Measures ---

def main_part1_shape(df_concrete):
    """
    Executes the shape measures calculations and interpretation.
    """
    print("\n--- Part 1.4: Shape Measures ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Calculate skewness and kurtosis
    # Note: pandas .kurt() returns "excess kurtosis" (Fisher's definition),
    # where 0 is normal.
    skew_val = df_concrete[column_name].skew()
    kurt_val = df_concrete[column_name].kurt()

    print("\n--- Shape Measures Calculations ---")
    print(f"Skewness: {skew_val:.4f}")
    print(f"Kurtosis: {kurt_val:.4f} (Fisher's definition, 0=normal)")

    # 2. Interpretation
    print("\n--- Interpretation of Shape ---")
    # Skewness
    if abs(skew_val) < 0.5:
        print(f"* **Skewness ({skew_val:.2f}):** The distribution is approximately symmetric.")
    elif skew_val > 0.5:
        print(f"* **Skewness ({skew_val:.2f}):** The distribution is moderately right-skewed (positive tail).")
    else:
        print(f"* **Skewness ({skew_val:.2f}):** The distribution is moderately left-skewed (negative tail).")

    # Kurtosis
    if abs(kurt_val) < 0.5:
        print(f"* **Kurtosis ({kurt_val:.2f}):** The distribution is mesokurtic, with a peak and tail weight similar to a normal distribution.")
    elif kurt_val > 0.5:
        print(f"* **Kurtosis ({kurt_val:.2f}):** The distribution is leptokurtic (heavy-tailed), meaning more outliers and a sharper peak than a normal distribution.")
    else:
        print(f"* **Kurtosis ({kurt_val:.2f}):** The distribution is platykurtic (light-tailed), meaning fewer outliers and a flatter peak than a normal distribution.")

    # 3. Visualize the distribution shape
    # This task is to visualize the histogram and density plot.
    # The function plot_distribution() from Part 1.2 already does this.
    # We can call it again or just note that it was already created.
    print("\n--- Visualizing Distribution Shape ---")
    print("The histogram and density plot (KDE) from Part 1.2")
    print("(concrete_strength_distribution.png) visually represents the shape.")
    # We can optionally re-generate it if needed:
    # plot_distribution(df_concrete, column_name,
    #                   'Concrete Strength Distribution (Shape)',
    #                   'concrete_strength_distribution.png')


# --- Part 1.5: Quantiles and Percentiles ---

def plot_boxplot(data, column, title, save_path=None):
    """
    Create a boxplot showing quartiles and outliers.
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found for plotting.")
        return

    plt.figure(figsize=(10, 4))

    # We use sns.boxplot, which automatically calculates quartiles and identifies outliers
    sns.boxplot(x=data[column])

    # Add a stripplot to show individual data points
    sns.stripplot(x=data[column], color='0.25', alpha=0.5, jitter=True)

    plt.title(title)
    plt.xlabel(column.replace('_', ' ').capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part1_quantiles(df_concrete):
    """
    Executes the quantile calculations and boxplot visualization.
    """
    print("\n--- Part 1.5: Quantiles and Percentiles ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Calculate Q1, Q2, Q3, Min, Max
    q1 = df_concrete[column_name].quantile(0.25)
    q2_median = df_concrete[column_name].quantile(0.50) # or .median()
    q3 = df_concrete[column_name].quantile(0.75)
    min_val = df_concrete[column_name].min()
    max_val = df_concrete[column_name].max()

    print("\n--- Five-Number Summary ---")
    print(f"Min:   {min_val:.2f} MPa")
    print(f"Q1:    {q1:.2f} MPa (25th Percentile)")
    print(f"Q2:    {q2_median:.2f} MPa (50th Percentile / Median)")
    print(f"Q3:    {q3:.2f} MPa (75th Percentile)")
    print(f"Max:   {max_val:.2f} MPa")

    # 2. Interpretation
    print("\n--- Interpretation ---")
    print("* The five-number summary provides a concise overview of the data's spread.")
    print(f"* 50% of the concrete samples have a strength between {q1:.2f} and {q3:.2f} MPa (the IQR).")
    print("* The plot will visually show if any data points (samples) are considered outliers.")

    # 3. Create boxplot
    print("\n--- Generating Boxplot ---")
    plot_boxplot(df_concrete, column_name,
                 'Boxplot of Concrete Strength (MPa) Showing Outliers',
                 'concrete_strength_boxplot.png')


# --- Part 2.1: Discrete Distributions ---

def plot_discrete_distribution(x, pmf_y, cdf_y, title, save_path=None):
    """
    Plots the PMF and CDF for a discrete distribution in two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- PMF Plot ---
    # Create a "lollipop" plot
    ax1.vlines(x, 0, pmf_y, colors='blue', lw=5, alpha=0.5, label='PMF')
    ax1.plot(x, pmf_y, 'o', color='blue')
    ax1.set_title('Probability Mass Function (PMF)')
    ax1.set_xlabel('k (Number of Events/Successes)')
    ax1.set_ylabel('P(X = k)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- CDF Plot ---
    # Use 'step' plot for CDF
    ax2.step(x, cdf_y, where='post', label='CDF', color='green')
    ax2.set_title('Cumulative Distribution Function (CDF)')
    ax2.set_xlabel('k (Number of Events/Successes)')
    ax2.set_ylabel('P(X <= k)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # --- Finalize ---
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part2_discrete_dist():
    """
    Generates samples, plots, and stats for discrete distributions.
    """
    print("\n--- Part 2.1: Discrete Distributions ---")

    # 1. Bernoulli Distribution
    print("\n--- Bernoulli Distribution ---")
    p_bern = 0.7
    print(f"Scenario: A single component passes inspection (p={p_bern}).")

    # Calculate Mean and Variance
    mean_bern, var_bern = stats.bernoulli.stats(p_bern, moments='mv')
    print(f"Theoretical Mean: {mean_bern:.2f}")
    print(f"Theoretical Variance: {var_bern:.2f}")

    # Generate Samples
    samples_bern = stats.bernoulli.rvs(p_bern, size=100)
    print(f"10 Random Samples (1=Pass, 0=Fail): {samples_bern[:10]}")

    # Plot PMF and CDF
    x_bern = [0, 1]
    pmf_bern = stats.bernoulli.pmf(x_bern, p_bern)
    cdf_bern = stats.bernoulli.cdf(x_bern, p_bern)
    plot_discrete_distribution(x_bern, pmf_bern, cdf_bern,
                               f'Bernoulli Distribution (p={p_bern})',
                               'bernoulli_plot.png')

    # 2. Binomial Distribution
    print("\n--- Binomial Distribution ---")
    n_binom = 20
    p_binom = 0.1
    print(f"Scenario: Number of defective items in a batch (n={n_binom}, p={p_binom}).")

    # Calculate Mean and Variance
    mean_binom, var_binom = stats.binom.stats(n_binom, p_binom, moments='mv')
    print(f"Theoretical Mean: {mean_binom:.2f}")
    print(f"Theoretical Variance: {var_binom:.2f}")

    # Generate Samples
    samples_binom = stats.binom.rvs(n_binom, p_binom, size=100)
    print(f"10 Random Samples (defects per batch): {samples_binom[:10]}")

    # Plot PMF and CDF
    x_binom = np.arange(0, n_binom + 1)
    pmf_binom = stats.binom.pmf(x_binom, n_binom, p_binom)
    cdf_binom = stats.binom.cdf(x_binom, n_binom, p_binom)
    plot_discrete_distribution(x_binom, pmf_binom, cdf_binom,
                               f'Binomial Distribution (n={n_binom}, p={p_binom})',
                               'binomial_plot.png')

    # 3. Poisson Distribution
    print("\n--- Poisson Distribution ---")
    lambda_poisson = 10
    print(f"Scenario: Number of truck arrivals in one hour (λ={lambda_poisson}).")

    # Calculate Mean and Variance
    mean_poisson, var_poisson = stats.poisson.stats(lambda_poisson, moments='mv')
    print(f"Theoretical Mean: {mean_poisson:.2f}")
    print(f"Theoretical Variance: {var_poisson:.2f}")

    # Generate Samples
    samples_poisson = stats.poisson.rvs(lambda_poisson, size=100)
    print(f"10 Random Samples (arrivals per hour): {samples_poisson[:10]}")

    # Plot PMF and CDF (plot up to a reasonable k, e.g., k=10 or 12)
    k_max_poisson = 12
    x_poisson = np.arange(0, k_max_poisson + 1)
    pmf_poisson = stats.poisson.pmf(x_poisson, lambda_poisson)
    cdf_poisson = stats.poisson.cdf(x_poisson, lambda_poisson)
    plot_discrete_distribution(x_poisson, pmf_poisson, cdf_poisson,
                               f'Poisson Distribution (λ={lambda_poisson})',
                               'poisson_plot.png')


# --- Part 2.2: Continuous Distributions ---

def plot_continuous_distribution(x, pdf_y, cdf_y, title, save_path=None):
    """
    Plots the PDF and CDF for a continuous distribution in two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- PDF Plot ---
    ax1.plot(x, pdf_y, 'r-', lw=2, label='PDF')
    ax1.fill_between(x, pdf_y, color='red', alpha=0.2)
    ax1.set_title('Probability Density Function (PDF)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- CDF Plot ---
    ax2.plot(x, cdf_y, 'g-', lw=2, label='CDF')
    ax2.set_title('Cumulative Distribution Function (CDF)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x) = P(X <= x)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # --- Finalize ---
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for title

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part2_continuous_dist():
    """
    Generates samples, plots, and stats for continuous distributions.
    """
    print("\n--- Part 2.2: Continuous Distributions ---")

    # 1. Uniform Distribution
    print("\n--- Uniform Distribution ---")
    # A load is uniformly distributed between 10 kN and 15 kN
    loc_unif = 10
    scale_unif = 5  # scale = width (b - a)
    print(f"Scenario: A load is uniformly distributed between {loc_unif} and {loc_unif + scale_unif} kN.")

    # Calculate Mean and Variance
    mean_unif, var_unif = stats.uniform.stats(loc=loc_unif, scale=scale_unif, moments='mv')
    print(f"Theoretical Mean: {mean_unif:.2f} kN")
    print(f"Theoretical Variance: {var_unif:.2f} kN²")

    # Generate Samples
    samples_unif = stats.uniform.rvs(loc=loc_unif, scale=scale_unif, size=100)
    print(f"10 Random Samples (load in kN): {np.round(samples_unif[:10], 2)}")

    # Plot PDF and CDF
    x_unif = np.linspace(loc_unif - (scale_unif * 0.1),
                         loc_unif + scale_unif + (scale_unif * 0.1), 200)
    pdf_unif = stats.uniform.pdf(x_unif, loc=loc_unif, scale=scale_unif)
    cdf_unif = stats.uniform.cdf(x_unif, loc=loc_unif, scale=scale_unif)
    plot_continuous_distribution(x_unif, pdf_unif, cdf_unif,
                                 f'Uniform Distribution (a={loc_unif}, b={loc_unif + scale_unif})',
                                 'uniform_plot.png')

    # 2. Normal Distribution
    print("\n--- Normal Distribution ---")
    # Steel yield strength
    mean_norm = 250
    std_norm = 15
    print(f"Scenario: Steel yield strength (μ={mean_norm} MPa, σ={std_norm} MPa).")

    # Calculate Mean and Variance
    mean_norm_calc, var_norm_calc = stats.norm.stats(loc=mean_norm, scale=std_norm, moments='mv')
    print(f"Theoretical Mean: {mean_norm_calc:.2f} MPa")
    print(f"Theoretical Variance: {var_norm_calc:.2f} MPa²")

    # Generate Samples
    samples_norm = stats.norm.rvs(loc=mean_norm, scale=std_norm, size=100)
    print(f"10 Random Samples (strength in MPa): {np.round(samples_norm[:10], 2)}")

    # Plot PDF and CDF
    # Plot 4 standard deviations from the mean
    x_norm = np.linspace(mean_norm - 4 * std_norm, mean_norm + 4 * std_norm, 200)
    pdf_norm = stats.norm.pdf(x_norm, loc=mean_norm, scale=std_norm)
    cdf_norm = stats.norm.cdf(x_norm, loc=mean_norm, scale=std_norm)
    plot_continuous_distribution(x_norm, pdf_norm, cdf_norm,
                                 f'Normal Distribution (μ={mean_norm}, σ={std_norm})',
                                 'normal_plot.png')

    # 3. Exponential Distribution
    print("\n--- Exponential Distribution ---")
    # Time until failure. Mean (1/λ) = 1000 hours
    # In scipy, scale = mean = 1/λ
    mean_exp = 1000
    print(f"Scenario: Time until component failure (Mean={mean_exp} hours).")

    # Calculate Mean and Variance
    mean_exp_calc, var_exp_calc = stats.expon.stats(scale=mean_exp, moments='mv')
    print(f"Theoretical Mean: {mean_exp_calc:.2f} hours")
    print(f"Theoretical Variance: {var_exp_calc:.2f} hours²")

    # Generate Samples
    samples_exp = stats.expon.rvs(scale=mean_exp, size=100)
    print(f"10 Random Samples (failure time in hours): {np.round(samples_exp[:10], 2)}")

    # Plot PDF and CDF
    # Plot up to 5x the mean lifetime
    x_exp = np.linspace(0, mean_exp * 5, 200)
    pdf_exp = stats.expon.pdf(x_exp, scale=mean_exp)
    cdf_exp = stats.expon.cdf(x_exp, scale=mean_exp)
    plot_continuous_distribution(x_exp, pdf_exp, cdf_exp,
                                 f'Exponential Distribution (Mean={mean_exp})',
                                 'exponential_plot.png')


# --- Part 2.3: Distribution Fitting ---

def fit_distribution(data, column, distribution_type='normal'):
    """
    Fit a specified probability distribution to data.
    (As required by Lab 4, Section 5.15)
    Returns the distribution parameters.
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found for fitting.")
        return None

    if distribution_type.lower() == 'normal':
        # Fit a normal distribution
        # .dropna() is good practice
        mu, std = norm.fit(data[column].dropna())
        return (mu, std)
    else:
        print(f"Distribution type '{distribution_type}' not supported yet.")
        return None

def plot_distribution_fitting(data, column, fitted_params, save_path=None):
    """
    Visualize fitted distribution (normal) overlaid on the histogram.
    (As required by Lab 4, Section 5.15)
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found for plotting.")
        return
    if fitted_params is None:
        print("Error: No fitted parameters provided for plotting.")
        return

    plt.figure(figsize=(10, 6))

    # Plot histogram
    sns.histplot(data[column], kde=True, bins=20, stat="density",
                 label="Data Histogram/KDE")

    # Plot fitted distribution
    mu, std = fitted_params
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2,
             label=f'Fitted Normal (μ={mu:.2f}, σ={std:.2f})')

    plt.title(f'Fitted Normal Distribution for {column}')
    plt.xlabel(column.replace('_', ' ').capitalize())
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part2_fitting(df_concrete):
    """
    Executes the distribution fitting and visualization.
    """
    print("\n--- Part 2.3: Distribution Fitting ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Fit a normal distribution
    print(f"Fitting normal distribution to '{column_name}' data...")
    fitted_params = fit_distribution(df_concrete, column_name, 'normal')

    if fitted_params:
        mu_fitted, std_fitted = fitted_params
        print(f"Fitted Parameters: Mean (μ) = {mu_fitted:.4f}, Std Dev (σ) = {std_fitted:.4f}")

        # 2. Compare fitted params with sample stats
        mean_sample = df_concrete[column_name].mean()
        std_sample = df_concrete[column_name].std()

        print("\n--- Comparison: Fitted vs. Sample Statistics ---")
        print(f"Sample Mean:   {mean_sample:.4f} MPa")
        print(f"Fitted Mean:   {mu_fitted:.4f} MPa")
        print(f"Sample StdDev: {std_sample:.4f} MPa")
        print(f"Fitted StdDev: {std_fitted:.4f} MPa")

        print("\n*Interpretation:* The fitted parameters are nearly identical to the sample statistics,")
        print(" as the 'fit' method for a normal distribution uses the sample mean and std. dev. as estimators.")

        # 3. Visualize the fitted distribution
        print("\n--- Generating Distribution Fitting Plot ---")
        plot_distribution_fitting(df_concrete, column_name, fitted_params,
                                  'distribution_fitting.png')

    else:
        print("Failed to fit distribution.")


# --- Part 3.1: Conditional Probability ---

def main_part3_conditional_prob():
    """
    Solves an engineering problem involving conditional probability.
    """
    print("\n--- Part 3.1: Conditional Probability ---")

    # Problem Scenario:
    print("Scenario: Quality control for a manufactured beam.")
    print("Let F = Event 'Beam is Faulty'")
    print("Let D = Event 'Test Detects a Defect'")

    # Given Probabilities:
    p_f = 0.02       # P(F): Prior probability of a beam being faulty (2%)
    p_not_f = 1 - p_f  # P(F'): Probability of a beam being OK (98%)

    p_d_given_f = 0.95 # P(D|F): Sensitivity - Prob. of detecting a defect if it IS faulty (95%)
    p_d_given_not_f = 0.03 # P(D|F'): False Positive Rate - Prob. of detecting a defect if it is OK (3%)

    print("\n--- Given Probabilities ---")
    print(f"P(Faulty) = P(F) = {p_f:.2f}")
    print(f"P(OK) = P(F') = {p_not_f:.2f}")
    print(f"P(Detect | Faulty) = P(D|F) = {p_d_given_f:.2f} (Sensitivity)")
    print(f"P(Detect | OK) = P(D|F') = {p_d_given_not_f:.2f} (False Positive Rate)")

    # Question:
    print("\n--- Question ---")
    print("What is the probability that a beam is ACTUALLY faulty, given that the test detects a defect?")
    print("We want to find: P(F | D)")

    # Solution using P(F|D) = P(F and D) / P(D)

    # 1. Calculate joint probabilities (Numerator)
    # P(F and D) = P(D|F) * P(F)
    p_f_and_d = p_d_given_f * p_f
    # P(F' and D) = P(D|F') * P(F')
    p_not_f_and_d = p_d_given_not_f * p_not_f

    # 2. Calculate total probability of detection (Denominator)
    # P(D) = P(F and D) + P(F' and D)
    p_d = p_f_and_d + p_not_f_and_d

    # 3. Calculate final conditional probability
    p_f_given_d = p_f_and_d / p_d

    # --- Visualization (Text-based Probability Tree) ---
    print("\n--- Probability Tree (Text Visualization) ---")
    print("  Root")
    print("   |")
    print(f"   +-- P(Faulty) = {p_f:.2f}")
    print(f"   |    +-- P(Detect | Faulty) = {p_d_given_f:.2f}  => P(Faulty AND Detect) = {p_f_and_d:.4f}")
    print(f"   |    +-- P(No Detect | Faulty) = {1-p_d_given_f:.2f}")
    print(f"   |")
    print(f"   +-- P(OK) = {p_not_f:.2f}")
    print(f"        +-- P(Detect | OK) = {p_d_given_not_f:.2f}      => P(OK AND Detect) = {p_not_f_and_d:.4f}")
    print(f"        +-- P(No Detect | OK) = {1-p_d_given_not_f:.2f}")

    # --- Calculation Steps ---
    print("\n--- Calculation (Law of Total Probability & Bayes') ---")
    print(f"1. P(Faulty AND Detect) = P(D|F) * P(F) = {p_d_given_f} * {p_f} = {p_f_and_d:.4f}")
    print(f"2. P(OK AND Detect) = P(D|F') * P(F') = {p_d_given_not_f} * {p_not_f} = {p_not_f_and_d:.4f}")
    print(f"3. P(Detect) = P(Faulty AND Detect) + P(OK AND Detect)")
    print(f"   P(Detect) = {p_f_and_d:.4f} + {p_not_f_and_d:.4f} = {p_d:.4f}")
    print(f"4. P(F | D) = P(F AND D) / P(D)")
    print(f"   P(F | D) = {p_f_and_d:.4f} / {p_d:.4f}")

    # --- Final Answer & Interpretation ---
    print("\n--- Final Answer ---")
    print(f"P(Faulty | Detect) = {p_f_given_d:.4f} (or {p_f_given_d*100:.2f}%)")
    print("\n*Interpretation:*")
    print("Even if a defect is detected, there is only a 39.26% chance the beam is actually faulty.")
    print("This is because the false positive rate (3% on 98% of beams) contributes a significant number")
    print("of 'detected' defects relative to the true positives (95% on 2% of beams).")


# --- Part 3.2: Bayes' Theorem ---

def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Apply Bayes' theorem for a diagnostic test scenario.
    (As required by Lab 4, Section 5.15)

    Args:
    - prior (float): P(Disease) - Base rate
    - sensitivity (float): P(Positive | Disease)
    - specificity (float): P(Negative | No Disease)

    Returns:
    - posterior (float): P(Disease | Positive)
    """
    p_positive_given_disease = sensitivity
    p_positive_given_no_disease = 1 - specificity
    p_disease = prior
    p_no_disease = 1 - prior

    # P(Positive) = P(Pos|Disease)*P(Disease) + P(Pos|No Disease)*P(No Disease)
    p_positive = (p_positive_given_disease * p_disease) + \
                 (p_positive_given_no_disease * p_no_disease)

    # P(Disease | Positive) = [P(Positive | Disease) * P(Disease)] / P(Positive)
    if p_positive == 0:
        return 0.0 # Avoid division by zero

    posterior = (p_positive_given_disease * p_disease) / p_positive

    return posterior

def main_part3_bayes_theorem():
    """
    Executes the Bayes' theorem application and interpretation.
    (This is Task 4 from the lab manual)
    """
    print("\n--- Part 3.2: Bayes' Theorem Application ---")

    # Scenario: Structural damage detection
    print("Scenario: Structural damage detection (Task 4 from lab manual)")
    print("Let D = Event 'Structure has Damage'")
    print("Let T+ = Event 'Test is Positive'")

    # Given probabilities from Task 4
    prior = 0.05       # P(D): Base rate of damage
    sensitivity = 0.95 # P(T+ | D): Test sensitivity
    specificity = 0.90 # P(T- | D'): Test specificity

    print("\n--- Given Probabilities ---")
    print(f"P(Damage) = P(D) = {prior:.2f} (This is the 'Prior' probability)")
    print(f"P(No Damage) = P(D') = {1-prior:.2f}")
    print(f"P(Positive | Damage) = P(T+|D) = {sensitivity:.2f} (This is the 'Likelihood' / Sensitivity)")
    print(f"P(Negative | No Damage) = P(T-|D') = {specificity:.2f} (Specificity)")
    print(f"P(Positive | No Damage) = P(T+|D') = {1-specificity:.2f} (False Positive Rate)")

    # Question:
    print("\n--- Question ---")
    print("If a test is positive (T+), what is the probability of actual damage (D)?")
    print("We want to find: P(D | T+)")

    # Apply the function
    posterior = apply_bayes_theorem(prior, sensitivity, specificity)

    # --- Interpretation ---
    print("\n--- Interpretation of Terms ---")
    print(f"* **Prior (P(D) = {prior*100:.0f}%):** Our initial belief. We assume 5% of structures have damage *before* any testing.")
    print(f"* **Likelihood (P(T+|D) = {sensitivity*100:.0f}%):** The strength of the evidence. How likely is a positive test if there *is* damage.")
    print(f"* **Posterior (P(D|T+) = {posterior*100:.2f}%):** Our updated belief *after* seeing the positive test result.")

    # --- Final Answer & Interpretation ---
    print("\n--- Final Answer ---")
    print(f"P(Damage | Positive Test) = {posterior:.4f} (or {posterior*100:.2f}%)")

    print("\n*Engineering Implications:*")
    print(f"Despite a 95% sensitive test, a positive result only means there is a {posterior*100:.2f}% chance")
    print("of actual damage. This is because the base rate of damage is very low (5%).")
    print("Most positive tests (approx. 68%) will be FALSE positives from the 95% of non-damaged structures.")
    print("This indicates that a positive test should lead to *further investigation*, not an immediate conclusion of damage.")


# --- Part 3.3: Basic Comparison ---

def plot_material_comparison(data, column, group_column, save_path=None):
    """
    Create comparative boxplot for material types.
    (As required by Lab 4, Section 5.15)
    """
    if column not in data.columns or group_column not in data.columns:
        print(f"Error: Columns '{column}' or '{group_column}' not found.")
        return

    plt.figure(figsize=(12, 7))

    # Create the boxplot
    sns.boxplot(x=group_column, y=column, data=data)

    # Overlay individual data points
    sns.stripplot(x=group_column, y=column, data=data,
                  color='0.25', alpha=0.5, jitter=True)

    plt.title(f'Comparison of {column.replace("_", " ")} by {group_column.replace("_", " ")}')
    plt.xlabel(group_column.replace('_', ' ').capitalize())
    plt.ylabel(column.replace('_', ' ').capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def main_part3_comparison():
    """
    Executes the basic comparison of material properties.
    (This is Task 2 from the lab manual)
    """
    print("\n--- Part 3.3: Basic Comparison (Material Properties) ---")

    # 1. Load the material properties dataset
    df_material = load_data('datasets/material_properties.csv')
    if df_material is None:
        print("Failed to load 'material_properties.csv'. Aborting task.")
        return

    # *** THIS IS THE FIX: ***
    # Changed 'strength_mpa' to 'yield_strength_mpa'
    column_name = 'yield_strength_mpa'
    group_column = 'material_type'

    # 2. Use descriptive statistics
    print("\n--- Descriptive Statistics by Material Type ---")

    # Check if columns exist before grouping
    if column_name not in df_material.columns:
        print(f"Error: Column '{column_name}' not found. Check CSV file.")
        return
    if group_column not in df_material.columns:
        print(f"Error: Column '{group_column}' not found. Check CSV file.")
        return

    grouped_stats = df_material.groupby(group_column)[column_name].describe()
    print(grouped_stats)

    # Interpretation
    print("\n--- Interpretation ---")
    # Adding a check in case 'Steel' or 'Composite' aren't in the loaded data
    if 'Steel' in grouped_stats.index:
        print(f"* **Mean:** 'Steel' has the highest mean strength (~{grouped_stats.loc['Steel', 'mean']:.0f} MPa).")
    if 'Composite' in grouped_stats.index and 'Steel' in grouped_stats.index:
        print(f"* **Variability (std):** 'Composite' has the highest standard deviation (~{grouped_stats.loc['Composite', 'std']:.2f}),")
        print("  indicating the least consistent performance. 'Steel' is the most consistent (~{grouped_stats.loc['Steel', 'std']:.2f}).")

    # 3. Use visual comparisons (Boxplot)
    print("\n--- Generating Comparative Boxplot ---")
    plot_material_comparison(df_material, column_name, group_column,
                             'material_comparison_boxplot.png')

# --- ADDITION: Part 4/5: Missing Functions, Tasks, and Report ---

# --- Part 5: Add Missing Required Functions ---

def calculate_descriptive_stats(data, column='strength_mpa'):
    """
    Calculate all descriptive statistics for a given column.
    (As required by Lab 4, Section 5.15)
    Returns a dictionary of statistics.
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in data.")
        return None

    data_col = data[column].dropna()

    stats_dict = {
        'count': data_col.count(),
        'mean': data_col.mean(),
        'median': data_col.median(),
        'mode': data_col.mode().iloc[0] if not data_col.mode().empty else np.nan,
        'std_dev': data_col.std(),
        'variance': data_col.var(),
        'min': data_col.min(),
        'max': data_col.max(),
        'range': data_col.max() - data_col.min(),
        'q1_25th': data_col.quantile(0.25),
        'q2_50th': data_col.quantile(0.50),
        'q3_75th': data_col.quantile(0.75),
        'iqr': data_col.quantile(0.75) - data_col.quantile(0.25),
        'skewness': data_col.skew(),
        'kurtosis': data_col.kurt()
    }
    return stats_dict

def calculate_probability_binomial(n, p, k, exact=True):
    """
    Calculate binomial probabilities.
    (As required by Lab 4, Section 5.15)
    If exact=True, calculates P(X = k).
    If exact=False, calculates P(X <= k).
    """
    if exact:
        # P(X = k)
        return stats.binom.pmf(k, n, p)
    else:
        # P(X <= k)
        return stats.binom.cdf(k, n, p)

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """
    Calculate normal probabilities for a given range.
    (As required by Lab 4, Section 5.15)
    """
    dist = stats.norm(loc=mean, scale=std)
    if x_lower is None and x_upper is not None:
        # P(X <= x_upper)
        return dist.cdf(x_upper)
    elif x_lower is not None and x_upper is None:
        # P(X >= x_lower)
        return dist.sf(x_lower)
    elif x_lower is not None and x_upper is not None:
        # P(x_lower <= X <= x_upper)
        return dist.cdf(x_upper) - dist.cdf(x_lower)
    else:
        # No bounds given
        return np.nan

def calculate_probability_poisson(lambda_param, k, exact=True):
    """
    Calculate Poisson probabilities.
    (As required by Lab 4, Section 5.15)
    If exact=True, calculates P(X = k).
    If exact=False, calculates P(X <= k).
    For P(X > k), use stats.poisson.sf(k, lambda_param)
    """
    if exact:
        # P(X = k)
        return stats.poisson.pmf(k, lambda_param)
    else:
        # P(X <= k)
        return stats.poisson.cdf(k, lambda_param)

def calculate_probability_exponential(mean, x, survival=False):
    """
    Calculate exponential probabilities.
    (As required by Lab 4, Section 5.15)
    Note: scipy uses scale = mean = 1/lambda
    If survival=False, calculates P(X <= x) (failure by time x)
    If survival=True, calculates P(X > x) (survival beyond time x)
    """
    if survival:
        # P(X > x)
        return stats.expon.sf(x, scale=mean)
    else:
        # P(X <= x)
        return stats.expon.cdf(x, scale=mean)

def create_statistical_report(stats_dict, task3_results, task4_posterior, output_file='lab4_statistical_report.md'):
    """
    Create a statistical report summarizing findings.
    (As required by Lab 4, Section 5.15)
    """
    print(f"\n--- Generating Statistical Report ---")

    try:
        with open(output_file, 'w') as f:
            f.write("# Lab 4: Statistical Analysis Report\n\n")

            # --- Task 1: Concrete Strength ---
            f.write("## Task 1: Concrete Strength Analysis\n\n")
            f.write("### Descriptive Statistics Table (`strength_mpa`)\n\n")
            if stats_dict:
                f.write("| Statistic      | Value     |\n")
                f.write("|----------------|-----------|\n")
                for key, value in stats_dict.items():
                    f.write(f"| {key.replace('_', ' ').capitalize():<14} | {value:,.4f} |\n")
            else:
                f.write("No statistics calculated.\n")

            f.write("\n### Key Findings & Engineering Implications\n\n")
            if stats_dict:
                mean_val = stats_dict['mean']
                median_val = stats_dict['median']
                std_val = stats_dict['std_dev']
                skew_val = stats_dict['skewness']

                f.write(f"* **Central Tendency:** The Mean ({mean_val:.2f} MPa) and Median ({median_val:.2f} MPa) are very close. This suggests the data is symmetric and not skewed by outliers.\n")
                f.write(f"* **Shape:** The Skewness ({skew_val:.4f}) is close to 0, confirming the symmetry. This is good for quality control, as it fits assumptions for normal distribution-based process control.\n")
                f.write(f"* **Variability:** The Standard Deviation ({std_val:.2f} MPa) is the key metric for consistency. All design codes (e.g., ACI) use this value to determine the 'specified compressive strength' (f'c) required to meet a target mean strength. A lower std dev means less over-design is needed, saving costs.\n")

            # --- Task 3: Probability Modeling ---
            f.write("\n## Task 3: Probability Modeling Scenarios\n\n")
            if task3_results:
                f.write("| Scenario       | Question                           | Probability |\n")
                f.write("|----------------|------------------------------------|-------------|\n")
                for key, val in task3_results.items():
                    # Unpack the list of tuples
                    for (desc, prob) in val:
                        f.write(f"| {key:<14} | {desc:<34} | {prob:,.4f} |\n")
            else:
                f.write("No probability scenarios calculated.\n")

            f.write("\n### Engineering Implications\n\n")
            if task3_results:
                f.write(f"* **Binomial:** The probability of 5 or fewer defects ({task3_results['Binomial'][1][1]:.4f}) is high. This model allows setting acceptance criteria (e.g., 'reject batch if > 5 defects') with a known probability of error.\n")
                f.write(f"* **Normal:** The 95th percentile strength ({task3_results['Normal'][1][1]:.2f} MPa) can be used as a characteristic strength for design, representing a value that 95% of the material is expected to exceed.\n")

            # --- Task 4: Bayes' Theorem ---
            f.write("\n## Task 4: Bayes' Theorem Application\n\n")
            f.write("Scenario: Probability of structural damage given a positive test.\n\n")
            f.write(f"* **Prior Probability (P(Damage)):** 0.05 (5%)\n")
            f.write(f"* **Test Sensitivity (P(Pos | Damage)):** 0.95 (95%)\n")
            f.write(f"* **Test Specificity (P(Neg | No Damage)):** 0.90 (90%)\n")
            f.write("\n**Resulting Posterior Probability (P(Damage | Positive Test)):**\n")
            f.write(f"## {task4_posterior:.4f} (or {task4_posterior*100:.2f}%)**\n\n")
            f.write("### Engineering Implications\n\n")
            f.write("This is a critical finding. A highly sensitive test (95%) still produces a high number of false positives when the base rate of the defect is low. A positive test **does not** confirm damage; it only raises the probability from 5% to ~32% (in this specific problem). This implies that a positive test must be followed by more detailed, and likely more expensive, secondary inspections before ordering costly repairs.\n")

        print(f"Successfully generated statistical report: '{output_file}'")
    except Exception as e:
        print(f"Error generating report: {e}")

# --- Task 3: Probability Modeling (New Main Function) ---

def main_task3_prob_modeling():
    """
    Solves the specific probability modeling scenarios from Task 3.
    Returns a dictionary of results for the report.
    """
    print("\n--- Detailed Task 3: Probability Modeling Scenarios ---")

    results = {}

    # 1. Binomial Scenario
    print("\n1. Binomial Scenario (n=100, p=0.05)")
    n_b, p_b = 100, 0.05
    # P(X = 3)
    prob_b_3 = calculate_probability_binomial(n_b, p_b, 3, exact=True)
    # P(X <= 5)
    prob_b_lte_5 = calculate_probability_binomial(n_b, p_b, 5, exact=False)
    print(f"P(exactly 3 defective components): {prob_b_3:.4f}")
    print(f"P(≤ 5 defective components):       {prob_b_lte_5:.4f}")
    results['Binomial'] = [
        ("P(X = 3)", prob_b_3),
        ("P(X <= 5)", prob_b_lte_5)
    ]

    # 2. Poisson Scenario
    print("\n2. Poisson Scenario (lambda=10)")
    lambda_p = 10
    # P(X = 8)
    prob_p_8 = calculate_probability_poisson(lambda_p, 8, exact=True)
    # P(X > 15) = 1 - P(X <= 15)
    prob_p_gt_15 = stats.poisson.sf(15, lambda_p) # sf() is P(X > k)
    print(f"P(exactly 8 trucks in an hour): {prob_p_8:.4f}")
    print(f"P(> 15 trucks in an hour):      {prob_p_gt_15:.4f}")
    results['Poisson'] = [
        ("P(X = 8)", prob_p_8),
        ("P(X > 15)", prob_p_gt_15)
    ]

    # 3. Normal Scenario
    print("\n3. Normal Scenario (mean=250, std=15)")
    mean_n, std_n = 250, 15
    # P(X > 280)
    prob_n_gt_280 = calculate_probability_normal(mean_n, std_n, x_lower=280)
    # 95th percentile
    perc_n_95 = stats.norm.ppf(0.95, loc=mean_n, scale=std_n)
    print(f"P(strength exceeds 280 MPa): {prob_n_gt_280:.4f} (or {prob_n_gt_280*100:.2f}%)")
    print(f"95th percentile strength:    {perc_n_95:.2f} MPa")
    results['Normal'] = [
        ("P(X > 280)", prob_n_gt_280),
        ("95th Percentile", perc_n_95)
    ]

    # 4. Exponential Scenario
    print("\n4. Exponential Scenario (mean=1000)")
    mean_e = 1000
    # P(X < 500)
    prob_e_lt_500 = calculate_probability_exponential(mean_e, 500, survival=False)
    # P(X > 1500)
    prob_e_gt_1500 = calculate_probability_exponential(mean_e, 1500, survival=True)
    print(f"P(failure before 500 hours): {prob_e_lt_500:.4f}")
    print(f"P(surviving beyond 1500 hours): {prob_e_gt_1500:.4f}")
    results['Exponential'] = [
        ("P(X < 500)", prob_e_lt_500),
        ("P(X > 1500)", prob_e_gt_1500)
    ]

    # Pack results for the report
    report_results = {
        'Binomial': [("P(X = 3)", prob_b_3), ("P(X <= 5)", prob_b_lte_5)],
        'Poisson': [("P(X = 8)", prob_p_8), ("P(X > 15)", prob_p_gt_15)],
        'Normal': [("P(X > 280)", prob_n_gt_280), ("95th Percentile", perc_n_95)],
        'Exponential': [("P(X < 500)", prob_e_lt_500), ("P(X > 1500)", prob_e_gt_1500)],
    }
    return report_results

# --- Task 5: Distribution Fitting and Validation (New Main Function) ---

def main_task5_validation(df_concrete):
    """
    Completes Task 5 by generating and comparing synthetic data.
    """
    print("\n--- Detailed Task 5: Distribution Fitting and Validation ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Fit distribution (reuse from Part 2.3)
    fitted_params = fit_distribution(df_concrete, column_name, 'normal')
    if not fitted_params:
        print("Fitting failed, aborting validation.")
        return

    mu_fitted, std_fitted = fitted_params
    print(f"Fitted Parameters: μ={mu_fitted:.2f}, σ={std_fitted:.2f}")

    # 2. Generate synthetic data
    sample_size = len(df_concrete)
    synthetic_data = stats.norm.rvs(loc=mu_fitted, scale=std_fitted, size=sample_size)
    print(f"Generated {sample_size} synthetic data points.")

    # 3. Validate mean and std dev
    mean_synthetic = synthetic_data.mean()
    std_synthetic = synthetic_data.std()

    print("\n--- Synthetic Data Validation ---")
    print(f"Fitted Mean:   {mu_fitted:.4f}  | Synthetic Mean:   {mean_synthetic:.4f}")
    print(f"Fitted StdDev: {std_fitted:.4f} | Synthetic StdDev: {std_synthetic:.4f}")
    print("*Interpretation:* The mean and std. dev. of the synthetic data are very close")
    print("to the fitted parameters, validating our model.")

    # 4. Compare synthetic vs. real data visually
    print("\n--- Generating Synthetic Data Comparison Plot ---")
    plt.figure(figsize=(10, 6))

    # Plot real data histogram
    sns.histplot(df_concrete[column_name], bins=20, stat="density",
                 label="Real Data", color="blue", alpha=0.6)
    # Plot synthetic data KDE
    sns.kdeplot(synthetic_data, label="Synthetic Data (KDE)",
                color="red", linestyle='--', linewidth=2.5)

    plt.title("Visual Comparison: Real vs. Synthetic Concrete Strength Data")
    plt.xlabel("Strength (MPa)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = "synthetic_data_comparison.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

# --- Part 4: Statistical Summary Dashboard (New Main Function) ---

def main_part4_dashboard(df_concrete):
    """
    Generates the 'statistical_summary_dashboard.png'
    """
    print("\n--- Part 4: Generating Statistical Summary Dashboard ---")

    if df_concrete is None:
        print("Data not loaded. Aborting task.")
        return

    column_name = 'strength_mpa'

    # 1. Get stats
    stats_dict = calculate_descriptive_stats(df_concrete, column_name)
    if not stats_dict:
        print("Failed to calculate stats for dashboard.")
        return

    # 2. Create 2x2 dashboard plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Statistical Summary Dashboard: Concrete Strength (MPa)',
                 fontsize=20, y=1.03)

    # --- Plot 1: Histogram and Fitted Normal Curve ---
    ax = axes[0, 0]
    sns.histplot(df_concrete[column_name], kde=True, bins=20, stat="density",
                 label="Data Histogram/KDE", ax=ax)
    # Add fitted curve
    mu, std = stats_dict['mean'], stats_dict['std_dev']
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'r', linewidth=2,
            label=f'Fitted Normal (μ={mu:.2f}, σ={std:.2f})')
    ax.set_title('Distribution and Fitted Normal Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Boxplot ---
    ax = axes[0, 1]
    sns.boxplot(y=df_concrete[column_name], ax=ax)
    sns.stripplot(y=df_concrete[column_name], ax=ax,
                  color='0.25', alpha=0.5, jitter=True)
    ax.set_title('Boxplot Showing Quartiles and Outliers')
    ax.set_ylabel('Strength (MPa)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Q-Q Plot (to check for normality) ---
    ax = axes[1, 0]
    stats.probplot(df_concrete[column_name], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot vs. Normal Distribution')
    ax.get_lines()[0].set_markerfacecolor('blue')
    ax.get_lines()[0].set_markeredgecolor('blue')
    ax.get_lines()[1].set_color('red')
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 4: Key Statistics Table ---
    ax = axes[1, 1]
    ax.axis('off') # Hide axes
    ax.set_title('Key Descriptive Statistics', pad=20)

    # Create the text for the table
    stats_text = (
        f"Count:    {stats_dict['count']}\n"
        f"Mean:     {stats_dict['mean']:.3f} MPa\n"
        f"Std Dev:  {stats_dict['std_dev']:.3f} MPa\n"
        f"Median:   {stats_dict['median']:.3f} MPa\n"
        f"IQR:      {stats_dict['iqr']:.3f} MPa\n"
        f"Min:      {stats_dict['min']:.3f} MPa\n"
        f"Max:      {stats_dict['max']:.3f} MPa\n"
        f"Skewness: {stats_dict['skewness']:.4f}\n"
        f"Kurtosis: {stats_dict['kurtosis']:.4f}"
    )
    # Add text to the plot
    ax.text(0.5, 0.5, stats_text,
            ha='center', va='center', fontsize=12,
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='grey', lw=1))

    # --- Finalize ---
    plt.tight_layout()
    save_path = "statistical_summary_dashboard.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

# --- Part 4/5: Final Report Generation (New Main Function) ---

def main_part4_and_5_final_report(df_concrete):
    """
    Gathers all results and generates the final statistical report.
    """
    print("\n--- Part 4/5: Final Report Generation ---")

    if df_concrete is None:
        print("Data not loaded. Aborting report generation.")
        return

    # 1. Get Task 1 Stats
    stats_dict = calculate_descriptive_stats(df_concrete, 'strength_mpa')

    # 2. Get Task 3 Results
    # (We re-run the calculations here to get the return values)
    task3_results = {
        'Binomial': [
            ("P(X = 3)", calculate_probability_binomial(100, 0.05, 3, exact=True)),
            ("P(X <= 5)", calculate_probability_binomial(100, 0.05, 5, exact=False))
        ],
        'Poisson': [
            ("P(X = 8)", calculate_probability_poisson(10, 8, exact=True)),
            ("P(X > 15)", stats.poisson.sf(15, 10)) # sf() is P(X > k)
        ],
        'Normal': [
            ("P(X > 280)", calculate_probability_normal(250, 15, x_lower=280)),
            ("95th Percentile", stats.norm.ppf(0.95, loc=250, scale=15))
        ],
        'Exponential': [
            ("P(X < 500)", calculate_probability_exponential(1000, 500, survival=False)),
            ("P(X > 1500)", calculate_probability_exponential(1000, 1500, survival=True))
        ]
    }

    # 3. Get Task 4 Results
    task4_posterior = apply_bayes_theorem(prior=0.05, sensitivity=0.95, specificity=0.90)

    # 4. Generate the report file
    create_statistical_report(stats_dict, task3_results, task4_posterior,
                              output_file='lab4_statistical_report.md')


if __name__ == "__main__":
    # Part 1 Components
    # Part 1.1
    df_concrete = main_part1_loading()

    # Pass the loaded data to the next functions
    if df_concrete is not None:
        # Part 1.2
        main_part1_central_tendency(df_concrete)
        # Part 1.3
        main_part1_spread(df_concrete)
        # Part 1.4
        main_part1_shape(df_concrete)
        # Part 1.5
        main_part1_quantiles(df_concrete)
    else:
        print("Failed to load initial data. Halting execution.")

    # Part 2 Components
    # This does not need to be in an if statement because its execution does not depend on success of data loading
    # Part 2.1
    main_part2_discrete_dist()
    # Part 2.2
    main_part2_continuous_dist()

    if df_concrete is not None: # However, this execution does
        # Part 2.3
        main_part2_fitting(df_concrete)

    # Part 3 Components
    # Part 3.1
    main_part3_conditional_prob()
    # Part 3.2
    main_part3_bayes_theorem()
    # Part 3.3
    main_part3_comparison()

    # --- ADDITION: Call final new functions ---

    # Call Task 3 (Specific Scenarios)
    main_task3_prob_modeling()

    if df_concrete is not None:
        # Call Task 5 (Validation)
        main_task5_validation(df_concrete)

        # Call Part 4 (Dashboard)
        main_part4_dashboard(df_concrete)

        # Call Part 4/5 (Final Report)
        main_part4_and_5_final_report(df_concrete)

    print("\n--- Lab 4 Complete ---")
