import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points per dataset
num_points = 1000

# Generate 'Actual' values uniformly distributed between 50 and 150
actual_values = np.random.uniform(5, 150, num_points)

# Dataset 1: Predicted deviates negatively from the 45-degree line
# Predicted = Actual * slope (slope < 1) + small noise
slope_neg = 0.9  # Slope less than 1 for negative deviation
noise_neg = np.random.normal(0, 5, num_points)  # Small noise
predicted_neg = actual_values * slope_neg + noise_neg

# Ensure no negative predictions
predicted_neg = np.maximum(predicted_neg, 0)

# Create DataFrame for Dataset 1
df_neg = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': predicted_neg
})

# Save Dataset 1 to CSV
df_neg.to_csv('dataset_negative_deviation.csv', index=False)

# Dataset 2: Predicted deviates positively from the 45-degree line
# Predicted = Actual * slope (slope > 1) + small noise
slope_pos = 1.1  # Slope greater than 1 for positive deviation
noise_pos = np.random.normal(0, 5, num_points)  # Small noise
predicted_pos = actual_values * slope_pos + noise_pos

# Create DataFrame for Dataset 2
df_pos = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': predicted_pos
})

# Save Dataset 2 to CSV
df_pos.to_csv('dataset_positive_deviation.csv', index=False)

# Dataset 3: Linear relationship with high scatter
# Predicted = Actual + large noise
noise_high = np.random.normal(0, 20, num_points)  # High variability
predicted_high_scatter = actual_values + noise_high

# Create DataFrame for Dataset 3
df_high_scatter = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': predicted_high_scatter
})

# Save Dataset 3 to CSV
df_high_scatter.to_csv('dataset_high_scatter.csv', index=False)

# Dataset 4: Linear relationship with low scatter
# Predicted = Actual + small noise
noise_low = np.random.normal(0, 5, num_points)  # Low variability
predicted_low_scatter = actual_values + noise_low

# Create DataFrame for Dataset 4
df_low_scatter = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': predicted_low_scatter
})

# Save Dataset 4 to CSV
df_low_scatter.to_csv('dataset_low_scatter.csv', index=False)

print("All datasets have been generated and saved as CSV files.")
