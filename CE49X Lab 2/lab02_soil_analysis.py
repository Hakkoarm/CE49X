# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: Hakan ARMAN
# Student ID: 2021403228
# Date: 15.10.2025

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print("-" * 50)
        print(f"Successfully loaded data from {file_path}")
        return df

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None


def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.

    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.

    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.

    Parameters:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    # TODO: Fill missing values in each specified column with the column mean
    columns_to_clean = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture'] # We do not clean the sample_id
    for column in columns_to_clean:
        if column in df_cleaned.columns:
            # Calculate the mean value of each column, ignoring NaN values
            mean_value = df_cleaned[column].mean()
            # Fill NaN values with the calculated mean
            df_cleaned[column] = df_cleaned[column].fillna(mean_value) # Fills the NaN values with mean_value
            print("-" * 50)
            print(f"Filled missing values in '{column}' with the mean ({mean_value:.2f}).")


    # Remove outliers for the 'soil_ph' column
    if 'soil_ph' in df_cleaned.columns:
        mean_ph = df_cleaned['soil_ph'].mean()
        std_ph = df_cleaned['soil_ph'].std()

        # Define the lower and upper bounds for outlier detection (mean ± 3 standard deviations)
        lower_bound = mean_ph - 3 * std_ph
        upper_bound = mean_ph + 3 * std_ph

        # Filter out rows where 'soil_ph' is outside the bounds
        original_count = len(df_cleaned)
        df_cleaned = df_cleaned[
            (df_cleaned['soil_ph'] >= lower_bound) &
            (df_cleaned['soil_ph'] <= upper_bound)
            # This line is kind of complex but thanks to Pandas we are able to create boolean series
            # (True/False data for each row). Since we use the & (and) operator, a row will be true
            # only if both conditions are satisfied. We then "mask" the Pandas dataframe using this series
            # only keeping True valued items (ie the ones between m-3sd <= x <= m+3sd)

        ]
        outlier_count = original_count - len(df_cleaned)
        print("-" * 50)
        print(f"Removed {outlier_count} outlier(s) from 'soil_ph' column.")
        print(f"Rows remaining after outlier removal: {len(df_cleaned)}")

    return df_cleaned



def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    # Check if the column exists in the DataFrame to prevent errors
    if df is None or column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame.")
        return

    # Calculate minimum value
    min_val = df[column].min()

    # Calculate maximum value
    max_val = df[column].max()

    # Calculate mean value
    mean_val = df[column].mean()

    # Calculate median value
    median_val = df[column].median()

    # Calculate standard deviation
    std_val = df[column].std()

    print("-" * 50)
    print(f"Descriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")


def main():
    """
    Main function to run the soil data analysis workflow.
    """
    # TODO: Update the file path to point to your soil_test.csv file
    file_path = 'soil_test.csv'

    # TODO: Load the dataset using the load_data function
    df_raw = load_data(file_path)

    if df_raw is not None:
        print("-" * 50)
        # Updated print statement to serve as the initial test/validation output
        print("First 5 rows of the loaded data for initial validation:")
        print(df_raw.head())
        # TODO: Clean the dataset using the clean_data function
        df_clean = clean_data(df_raw)
        print("-" * 50)
        print("First 5 rows of the cleaned data for testing purposes:")
        print(df_clean.head())
        if df_clean is not None and not df_clean.empty:

            # TODO: Compute and display statistics for the 'soil_ph' column
            compute_statistics(df_clean, 'soil_ph')

            # TODO: (Optional) Compute statistics for other columns
            compute_statistics(df_clean, 'nitrogen')
            compute_statistics(df_clean, 'phosphorus')
            compute_statistics(df_clean, 'moisture')
        else:
            print("No data to analyze after cleaning.")

if __name__ == "__main__":
    main()
    # This name check is required because it prevents the program from automatically starting after being imported

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: Optional part of removing outlier data.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: It could help assess ground conditions for foundation considerations.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: We could develop the program further by implementing GIS integration, allowing us to map test data to
# corresponding in situ coordination. We can also make 2 additional improvements.
# Firstly, instead of replacing NaN entries with the mean values we can use an AI model to more accurately
# predict the real condition by analyzing other parameters.
# Secondly, we can implement a preliminary alert mechanism when some parameters are abnormal.

# 4. How did error handling improve the robustness of your code?
# Answer: It prevents the program from outright crashing after reading unusual values.
# which are expected because we are directly reading from raw lab data files
