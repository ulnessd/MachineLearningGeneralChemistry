import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of kernels
n_samples = 10

# Generate X1, X2, X3 as random values
S = range(1, n_samples + 1)
EARitability = np.random.uniform(0, 10, n_samples)
aMAIZEingness = np.random.uniform(1, 10, n_samples)
X3 = np.random.uniform(0, 10, n_samples)  # Irrelevant feature
X4 = np.sqrt(EARitability) + np.random.normal(0, 0.1, n_samples)  # Non-linear, noisy version of X1

# Create CORNYness with more complex interactions and conditional logic
noise = np.random.normal(0, 0.1, n_samples)
CORNYness = np.where(EARitability > 5, EARitability**2 + aMAIZEingness, EARitability - np.sin(aMAIZEingness)) + 0.3 * X3 + noise

# Create a DataFrame and round values to 2 decimal places
df = pd.DataFrame({
    'Kernel': S,
    'EARitability': np.round(EARitability, 2),
    'aMAIZEingness': np.round(aMAIZEingness, 2),
    'CORNYness': np.round(CORNYness, 2)
})

# Save to CSV
df.to_csv('NewKernelData.csv', index=False)

print("CSV file 'NewKernelData.csv' created successfully!")

