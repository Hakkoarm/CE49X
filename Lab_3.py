"""
Lab 3 - Step 1: Load and Explore the Datasets
---------------------------------------------
This script:
 - Loads ERA5 wind data for Berlin and Munich (from CSV files)
 - Ensures correct datetime parsing with pd.Timestamp
 - Displays dataset shape, columns, and data types
 - Handles missing values
 - Calculates and displays summary statistics
"""

import pandas as pd
import numpy as np
import sys
# I've heard it is better practice to use sys.exit() instead of the default quit() or exit() as it is more robust
# the (1) argument inside specifies there has been a general error and that is the reason for the halt.

print("==================================================")
print("Lab 3 - Step 1: Load and Explore the Datasets")
print("==================================================")
print()

# Writing out the file names for easier access later on
berlin_file_location = "berlin_era5_wind_20241231_20241231.csv"
munich_file_location = "munich_era5_wind_20241231_20241231.csv"

# --- Load the CSVs with Error Handling ---
print("Loading data...")
try:
    df_berlin = pd.read_csv(berlin_file_location, parse_dates=["timestamp"])
    df_munich = pd.read_csv(munich_file_location, parse_dates=["timestamp"])

    # Check for empty files (files with headers but no data)
    if df_berlin.empty or df_munich.empty:
        print("--------------------------------------------------")
        print("FATAL ERROR: A file loaded successfully but contains no data rows.")
        print("--------------------------------------------------")
        sys.exit(1)  # Exit the script

    print("Successfully loaded Berlin and Munich data files.\n")

    # Confirm the time column type
    print("Berlin time column type:", type(df_berlin.loc[0, "timestamp"]))
    print("Munich time column type:", type(df_munich.loc[0, "timestamp"]))
    print()

# Catches "File Not Found"
except FileNotFoundError as e:
    print("--------------------------------------------------")
    print(f"FATAL ERROR: File not found.")
    print(f"Details: {e}")
    print("Please make sure the CSV files are in the same directory as the script.")
    print("--------------------------------------------------")
    sys.exit(1)

# Catches "File is Empty" (0 bytes)
except pd.errors.EmptyDataError as e:
    print("--------------------------------------------------")
    print(f"FATAL ERROR: One of the files is empty.")
    print(f"Details: {e}")
    print("--------------------------------------------------")
    sys.exit(1)

# Catches "File is Corrupt" (badly formatted)
except pd.errors.ParserError as e:
    print("--------------------------------------------------")
    print(f"FATAL ERROR: File is corrupt or badly formatted.")
    print(f"Details: {e}")
    print("--------------------------------------------------")
    sys.exit(1)

# Catches all other errors (e.g., no permissions)
except Exception as e:
    print("--------------------------------------------------")
    print(f"FATAL ERROR: An unexpected error occurred during file loading.")
    print(f"Details: {e}")
    print("--------------------------------------------------")
    sys.exit(1)

# Confirm the time column type
print("Berlin time column type:", type(df_berlin.loc[0, "timestamp"]))
print("Munich time column type:", type(df_munich.loc[0, "timestamp"]))
print()

# --- Basic dataset info ---
print("=== BERLIN DATA ===")
print("Shape:", df_berlin.shape)
print("Columns:", df_berlin.columns.tolist())
print(df_berlin.dtypes)
print()

print("=== MUNICH DATA ===")
print("Shape:", df_munich.shape)
print("Columns:", df_munich.columns.tolist())
print(df_munich.dtypes)
print()

# --- Handle missing values ---
print("Missing values (Berlin):")
print(df_berlin.isna().sum())
print()
print("Missing values (Munich):")
print(df_munich.isna().sum())
print()

# Example handling: forward-fill short gaps and drop remaining NaNs
df_berlin = df_berlin.ffill(limit=3).dropna()
df_munich = df_munich.ffill(limit=3).dropna()

# --- Summary statistics ---
print("=== Summary Statistics for Berlin ===")
print(df_berlin.describe().T)
# .describe method summarizes given database, printing information such as std and mean
# .T method is used to transpose the dataset in order to increase readibility
print()

print("=== Summary Statistics for Munich ===")
print(df_munich.describe().T)
print()

# If the files have u and v components (e.g., u10, v10), calculate total wind speed.
df_berlin["wind_speed"] = np.sqrt(df_berlin["u10m"]**2 + df_berlin["v10m"]**2)
df_munich["wind_speed"] = np.sqrt(df_munich["u10m"]**2 + df_munich["v10m"]**2)
print("Added 'wind_speed' column for both datasets.\n")

print("Berlin wind speed stats:")
print(df_berlin["wind_speed"].describe())
print("\nMunich wind speed stats:")
print(df_munich["wind_speed"].describe())

print()
print("==================================================")
print("Lab 3 - Step 2: Temporal Aggregations")
print("==================================================")
print()


# --- Set Timestamp as Index ---
# This is crucial for using pandas time-series resampling functions for easier timestamp > month conversion.
try:
    df_berlin = df_berlin.set_index("timestamp")
    df_munich = df_munich.set_index("timestamp")
    print("Set 'timestamp' as the index for both DataFrames.\n")
except KeyError:
    print("Index already set.\n")


# --- 1. Calculate Monthly Averages ---
# We use .resample('ME') for "Month-End" to avoid the warning.
monthly_avg_berlin = df_berlin['wind_speed'].resample('ME').mean()
monthly_avg_munich = df_munich['wind_speed'].resample('ME').mean()

# --- Give the Series a name (this names the 'column') ---
monthly_avg_berlin.name = "Avg_Wind_Speed"
monthly_avg_munich.name = "Avg_Wind_Speed"

# --- Change the index to show Month Name ---
# We use .index.month_name() to get "April", "May", etc.
monthly_avg_berlin.index = monthly_avg_berlin.index.month_name()
monthly_avg_munich.index = monthly_avg_munich.index.month_name()

# --- Set the index name (this names the 'index column') ---
monthly_avg_berlin.index.name = "Month"
monthly_avg_munich.index.name = "Month"

print("--- Monthly Average Wind Speed (m/s) ---")
print("\nBerlin Monthly Average Wind Speed:")
print(monthly_avg_berlin)
print("\nMunich Monthly Average Wind Speed:")
print(monthly_avg_munich)
print("\n" + "-"*30 + "\n")

# --- 2. Calculate Seasonal Averages ---
# We use .resample('QE-NOV') for "Quarter-End ending in November"
# Winter starts with december, thus, a full circle of seasons ends at November (considering summer as the default start)
# This groups by meteorological seasons (DJF, MAM, JJA, SON)
seasonal_avg_berlin = df_berlin['wind_speed'].resample('QE-NOV').mean()
seasonal_avg_munich = df_munich['wind_speed'].resample('QE-NOV').mean()

print("--- Seasonal Average Wind Speed (m/s) ---")
# Rename the index to be clearer (e.g., 2024-02-29 becomes 'Winter 2024')
season_map = {2: 'Winter', 5: 'Spring', 8: 'Summer', 11: 'Fall'}
seasonal_avg_berlin.index = [f"{season_map[month]} {year}" for month, year in zip(seasonal_avg_berlin.index.month, seasonal_avg_berlin.index.year)]
seasonal_avg_munich.index = [f"{season_map[month]} {year}" for month, year in zip(seasonal_avg_munich.index.month, seasonal_avg_munich.index.year)]

print("\nBerlin Seasonal Average Wind Speed:")
print(seasonal_avg_berlin)
print("\nMunich Seasonal Average Wind speed:")
print(seasonal_avg_munich)
print("\n" + "-"*30 + "\n")


# --- 3. Compare Seasonal Patterns ---
# Create a single DataFrame for easy comparison
df_seasonal_compare = pd.DataFrame({
    'Berlin_Avg_Wind_Speed': seasonal_avg_berlin,
    'Munich_Avg_Wind_Speed': seasonal_avg_munich
})

# Add a column to see the difference
df_seasonal_compare['Difference (Berlin - Munich)'] = df_seasonal_compare['Berlin_Avg_Wind_Speed'] - df_seasonal_compare['Munich_Avg_Wind_Speed']
print("--- Seasonal Comparison Delta ---")
print(df_seasonal_compare)
print("\nAnalysis: A positive 'Difference' means Berlin had higher average wind speed for that season.")

print()
print("==================================================")
print("Lab 3 - Step 3: Statistical Analysis")
print("==================================================")
print()

# --- 1. Identify Extreme Weather (Highest Wind Speed Moments) ---
# We sort the original DataFrame by 'wind_speed' in descending order
top_5_windy_days_berlin = df_berlin.sort_values('wind_speed', ascending=False).head(5)
top_5_windy_days_munich = df_munich.sort_values('wind_speed', ascending=False).head(5)

print("--- Top 5 Windiest Moments (Berlin) ---")
# We print the wind speed and the u/v components for analysis
print(top_5_windy_days_berlin[['wind_speed', 'u10m', 'v10m']])
print("\n--- Top 5 Windiest Moments (Munich) ---")
print(top_5_windy_days_munich[['wind_speed', 'u10m', 'v10m']])
print("\n" + "-"*30 + "\n")


# --- 2. Calculate Diurnal (Daily) Patterns ---
# We group the data by the hour of the day (0-23)
# df_berlin.index.hour gives us the hour for each timestamp
diurnal_pattern_berlin = df_berlin.groupby(df_berlin.index.hour)['wind_speed'].mean()
diurnal_pattern_munich = df_munich.groupby(df_munich.index.hour)['wind_speed'].mean()

# Combine into a single DataFrame for easier analysis
df_diurnal = pd.DataFrame({
    'Berlin_Avg_Wind_Speed': diurnal_pattern_berlin,
    'Munich_Avg_Wind_Speed': diurnal_pattern_munich
})
df_diurnal.index.name = 'Hour of Day'

print("--- Diurnal (Hourly) Average Wind Speed (m/s) ---")
print(df_diurnal)
print("\n" + "-"*30 + "\n")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print()
print("==================================================")
print("Lab 3 - Step 4: Visualization")
print("==================================================")
print()

# --- Visualization 1: Monthly Average Wind Speed Trend Plot ---
# We use the Series with month names as the index.
# Matplotlib will treat the index (April, May, etc.) as categorical labels.
# We also use 'monthly_avg_berlin' from Step 2, which already has the name "Avg_Wind_Speed".
# We'll combine them into a DataFrame for easier plotting.

df_monthly_compare = pd.DataFrame({
    'Berlin': monthly_avg_berlin,
    'Munich': monthly_avg_munich
})
df_monthly_compare.index.name = "Month"

print("Creating plot 1: 'monthly_wind_speed_comparison.png'")
try:
    # Use pandas' built-in plotting
    ax = df_monthly_compare.plot(
        kind='line',
        style=['-o', '--s'],  # line style and marker for each column
        figsize=(12, 6)
    )

    # --- Formatting the plot ---
    ax.set_title('Monthly Average Wind Speed (2024)', fontsize=16)
    ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.legend(['Berlin', 'Munich'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('monthly_wind_speed_comparison.png')

    print("Successfully saved 'monthly_wind_speed_comparison.png'")
    plt.close()  # Close the figure to free up memory

except Exception as e:
    print(f"Error creating plot 1: {e}")

print("\nCreating plot 2: 'seasonal_wind_speed_comparison.png'")
# --- Visualization 2: Seasonal Comparison Bar Chart ---
# We use the 'df_seasonal_compare' DataFrame created in Step 2
try:
    # Use pandas' built-in plotting for easy bar charts
    ax = df_seasonal_compare.plot(
        kind='bar',
        y=['Berlin_Avg_Wind_Speed', 'Munich_Avg_Wind_Speed'],
        figsize=(10, 6),
        rot=0  # Keep season names horizontal
    )

    # --- Formatting the plot ---
    ax.set_title('Seasonal Average Wind Speed Comparison', fontsize=16)
    ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
    ax.set_xlabel('Season', fontsize=12)
    ax.legend(['Berlin', 'Munich'])
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('seasonal_wind_speed_comparison.png')

    print("Successfully saved 'seasonal_wind_speed_comparison.png'")
    plt.close()  # Close the figure

except Exception as e:
    print(f"Error creating plot 2: {e}")

import seaborn as sns

print("\nCreating plot 3: 'seasonal_wind_speed_boxplot.png'")
# --- Visualization 3: Seasonal Comparison Box Plot ---

try:
    # --- Prepare Data for Box Plot ---
    # A box plot needs the full distribution, so we use the original DataFrames.
    # We first create a mapping from month number to season.
    season_map_names = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall',
        12: 'Winter'
    }

    # Add a 'Season' column to each DataFrame
    df_berlin['Season'] = df_berlin.index.month.map(season_map_names)
    df_munich['Season'] = df_munich.index.month.map(season_map_names)

    # For a side-by-side plot with Seaborn, it's easiest to combine them
    # into one "long-form" DataFrame.

    # Add a 'City' column to each before combining
    df_berlin_plot = df_berlin[['wind_speed', 'Season']].copy()
    df_berlin_plot['City'] = 'Berlin'

    df_munich_plot = df_munich[['wind_speed', 'Season']].copy()
    df_munich_plot['City'] = 'Munich'

    # Concatenate them
    df_combined_plot = pd.concat([df_berlin_plot, df_munich_plot])

    # Define the order of seasons (based on your data, April -> Dec)
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']

    # --- Create the Plot ---
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df_combined_plot,
        x='Season',
        y='wind_speed',
        hue='City',
        order=season_order,
        palette="Set2"  # Use a color-blind friendly palette
    )

    # --- Formatting the plot ---
    plt.title('Seasonal Wind Speed Distribution', fontsize=16)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Wind Speed (m/s)', fontsize=12)
    plt.legend(title='City', loc='upper right')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('seasonal_wind_speed_boxplot.png')

    print("Successfully saved 'seasonal_wind_speed_boxplot.png'")
    plt.close()  # Close the figure

except Exception as e:
    print(f"Error creating plot 3: {e}")

print("\nCreating plot 4: 'wind_component_scatter.png'")
# --- Visualization 4: Wind Component Scatter Plot (u vs. v) ---

try:
    # --- Create the Subplots (1 row, 2 columns) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- Find common limits for a fair comparison ---
    # Find the most extreme value across all u/v components in both cities
    min_val = min(df_berlin['u10m'].min(), df_berlin['v10m'].min(), df_munich['u10m'].min(), df_munich['v10m'].min())
    max_val = max(df_berlin['u10m'].max(), df_berlin['v10m'].max(), df_munich['u10m'].max(), df_munich['v10m'].max())
    # Create a symmetrical limit based on the largest absolute value, add 1 for padding
    limit = max(abs(min_val), abs(max_val)) + 1

    # --- Plot 1: Berlin ---
    ax1.scatter(df_berlin['u10m'], df_berlin['v10m'], alpha=0.3, label='Berlin', color='blue')

    # --- Formatting for ax1 ---
    ax1.set_title('Berlin Wind Components')
    ax1.set_xlabel('u10m (East-West Component)')
    ax1.set_ylabel('v10m (South-North Component)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Add 0-lines to show quadrants
    ax1.axhline(0, color='black', linewidth=0.7)
    ax1.axvline(0, color='black', linewidth=0.7)
    # Set limits and make aspect ratio equal
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_aspect('equal')

    # --- Plot 2: Munich ---
    ax2.scatter(df_munich['u10m'], df_munich['v10m'], alpha=0.3, label='Munich', color='green')

    # --- Formatting for ax2 ---
    ax2.set_title('Munich Wind Components')
    ax2.set_xlabel('u10m (East-West Component)')
    ax2.set_ylabel('v10m (South-North Component)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    # Add 0-lines to show quadrants
    ax2.axhline(0, color='black', linewidth=0.7)
    ax2.axvline(0, color='black', linewidth=0.7)
    # Set limits and make aspect ratio equal
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)
    ax2.set_aspect('equal')

    # --- Final Touches ---
    fig.suptitle('Wind Component Scatter Plots (m/s)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

    # Save the figure
    plt.savefig('wind_component_scatter.png')

    print("Successfully saved 'wind_component_scatter.png'")
    plt.close()  # Close the figure

except Exception as e:
    print(f"Error creating plot 4: {e}")

print("\nCreating plot 5: 'monthly_wind_difference.png'")
# --- Visualization 5: Monthly Difference Line Plot ---
# We will add a 'Difference' column to our df_monthly_compare DataFrame

try:
    df_monthly_compare['Difference (Berlin - Munich)'] = df_monthly_compare['Berlin'] - df_monthly_compare['Munich']

    # --- Create the Plot ---
    plt.figure(figsize=(12, 6))
    ax = df_monthly_compare['Difference (Berlin - Munich)'].plot(
        kind='line',
        marker='o',
        color='purple'
    )

    # --- Formatting the plot ---
    ax.set_title('Monthly Wind Speed Difference (Berlin - Munich)', fontsize=16)
    ax.set_ylabel('Wind Speed Difference (m/s)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)

    # Add a horizontal line at 0 to clearly show positive/negative
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add text to explain the plot
    ax.text(
        0.02, 0.05,
        'Positive = Berlin windier\nNegative = Munich windier',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    )

    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('monthly_wind_difference.png')

    print("Successfully saved 'monthly_wind_difference.png'")
    plt.close()  # Close the figure

except Exception as e:
    print(f"Error creating plot 5: {e}")

print("\nCreating plot 6: 'wind_rose.png'")
# --- Visualization 6: Wind Rose Diagram ---
# This plot combines direction (from u/v) and magnitude (wind_speed)

import numpy as np


def plot_wind_rose(df, ax, title):
    """
    Helper function to plot a full, speed-binned wind rose.
    """
    # --- 1. Calculate Meteorological Wind Direction ---
    # We use atan2(u, v) and add 180 to get the "coming from" direction
    # 0=North, 90=East, 180=South, 270=West
    direction_rad = np.arctan2(df['u10m'], df['v10m'])
    direction_deg = (np.degrees(direction_rad) + 180) % 360

    # --- 2. Bin Wind Direction (16 bins) ---
    # 16 bins, 22.5 degrees each
    dir_bins = np.arange(0, 360 + 22.5, 22.5)
    # Use pandas.cut to bin the directions
    df['dir_bin'] = pd.cut(direction_deg, bins=dir_bins, right=False, include_lowest=True)

    # --- 3. Bin Wind Speed ---
    speed_bins = [0, 2, 4, 6, 8, np.inf]
    speed_labels = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8+ m/s']
    df['speed_bin'] = pd.cut(df['wind_speed'], bins=speed_bins, labels=speed_labels, right=False)

    # --- 4. Aggregate Data ---
    # Create a crosstab to count frequencies for each direction/speed bin
    # We normalize by the total number of observations to get frequency %
    crosstab = pd.crosstab(index=df['dir_bin'], columns=df['speed_bin'], normalize='all') * 100
    # Ensure columns are in the correct order for stacking
    crosstab = crosstab.reindex(columns=speed_labels)

    # --- 5. Plot the Stacked Bars ---
    # Get the bin centers (in radians) for plotting
    bin_centers_deg = np.arange(22.5 / 2, 360, 22.5)
    bin_centers_rad = np.radians(bin_centers_deg)
    width = np.radians(22.5)  # Width of each bar

    bottom = np.zeros(len(bin_centers_rad))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(speed_labels)))

    for i, speed_cat in enumerate(speed_labels):
        values = crosstab[speed_cat].fillna(0).values
        ax.bar(bin_centers_rad, values, width=width, bottom=bottom, label=speed_cat, color=colors[i], edgecolor='k',
               linewidth=0.5)
        bottom += values  # Stack the next bar on top

    # --- 6. Format the Polar Plot ---
    ax.set_theta_zero_location('N')  # Set 0 degrees to be North

    # --- THIS IS THE CORRECTED LINE ---
    ax.set_theta_direction(-1)  # Set direction to be clockwise

    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    ax.set_rlabel_position(90)  # Move radial labels to the East
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title, pad=20)


try:
    # --- Create the Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': 'polar'})

    # Plot for Berlin (pass a copy to avoid modifying the original df)
    plot_wind_rose(df_berlin.copy(), ax1, 'Berlin Wind Rose')

    # Plot for Munich
    plot_wind_rose(df_munich.copy(), ax2, 'Munich Wind Rose')

    # --- Add a single legend for the whole figure ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Wind Speed (m/s)')

    fig.suptitle('Wind Rose (Direction and Speed)', fontsize=20)
    # Adjust layout to make room for the legend (rect=[left, bottom, right, top])
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

    # Save the figure
    plt.savefig('wind_rose.png')

    print("Successfully saved 'wind_rose.png'")
    plt.close()  # Close the figure

except Exception as e:
    print(f"Error creating plot 6: {e}")

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt  # Ensure this is imported if not already

print("\nCreating plot 7: 'wind_speed_calendar_heatmap.png'")


# --- Visualization 7: Wind Speed Calendar Heatmap ---

def plot_calendar_heatmap(df, ax, title):
    """
    Helper function to create a calendar-style heatmap for wind speed.
    """
    # --- 1. Resample to Daily Maximum Wind Speed ---
    daily_max = df['wind_speed'].resample('D').max().dropna()

    # --- 2. Define Categories (Calm, Normal, Extreme) ---
    q_low = daily_max.quantile(0.25)
    q_high = daily_max.quantile(0.75)

    bins = [0, q_low, q_high, np.inf]
    labels = [0, 1, 2]  # 0=Calm, 1=Normal, 2=Extreme

    df_daily = pd.DataFrame(daily_max)
    df_daily['Category'] = pd.cut(df_daily['wind_speed'], bins=bins, labels=labels, include_lowest=True)

    # --- 3. Prepare Data for Pivot ---
    df_daily['Month'] = df_daily.index.month_name()
    df_daily['Day'] = df_daily.index.day

    month_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    # --- 4. Pivot the Data ---
    try:
        heatmap_data = df_daily.pivot(index='Day', columns='Month', values='Category')
        heatmap_data = heatmap_data.reindex(columns=month_order)
    except Exception as e:
        print(f"Error pivoting data for {title}: {e}")
        df_daily['YearMonth'] = df_daily.index.to_period('M')
        heatmap_data = df_daily.pivot_table(index='Day', columns='YearMonth', values='Category', aggfunc='mean')

    heatmap_data = heatmap_data.astype(float)

    # --- 5. DEFINE COLORS AND NORMALIZATION ---
    # Define the discrete colors
    cmap = ListedColormap(['#0077b6', '#2ca02c', '#d62728'])  # 0=Blue, 1=Green, 2=Red

    # Define the boundaries for the colors. This creates 3 "bins":
    # [0-1], [1-2], [2-3]
    boundaries = [0, 1, 2, 3]

    # Create a normalization object to map values to colors
    # cmap.N is 3 (the number of colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Define the tick locations for the labels (center of each bin)
    ticks = [0.5, 1.5, 2.5]

    # --- 6. Plot the Heatmap ---
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        norm=norm,  # <-- ADD THIS to force the heatmap to use the norm
        linewidths=0.5,
        linecolor='white',
        cbar=False  # We'll draw the colorbar manually
    )

    # --- 7. Format the Plot and Colorbar ---
    ax.set_title(title, pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Day of Month')

    # Create a new "mappable" object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is a small hack needed for ScalarMappable

    cbar_ax = ax.figure.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])

    # Build the colorbar using the mappable, boundaries, and ticks
    cb = plt.colorbar(sm, cax=cbar_ax, ticks=ticks, boundaries=boundaries)

    cb.set_ticklabels(['Calm', 'Normal', 'Extreme'])
    cb.set_label('Wind Intensity', rotation=270, labelpad=15)

    # Remove the small tick marks on the colorbar
    cb.ax.tick_params(length=0)


try:
    # --- Create the Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    fig.subplots_adjust(right=0.85, wspace=0.3)

    plot_calendar_heatmap(df_berlin, ax1, 'Berlin Daily Max Wind Speed')
    plot_calendar_heatmap(df_munich, ax2, 'Munich Daily Max Wind Speed')

    fig.suptitle('Wind Speed Calendar Heatmap (Calm, Normal, Extreme)', fontsize=20)

    plt.savefig('wind_speed_calendar_heatmap.png')

    print("Successfully saved 'wind_speed_calendar_heatmap.png'")
    plt.close()

except Exception as e:
    print(f"Error creating plot 7: {e}")