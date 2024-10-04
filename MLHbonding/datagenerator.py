import numpy as np
import pandas as pd

# Define molecule set
molecules = ['R-O', 'R-N', 'R-C', 'R-O-H', 'R-N-H', 'R-C-H']

# Define pair categories
category_1_pairs = [('R-O', 'R-O'), ('R-N', 'R-N'), ('R-C', 'R-C'),
                    ('R-O-H', 'R-O-H'), ('R-N-H', 'R-N-H'), ('R-C-H', 'R-C-H')]

category_2_pairs = [('R-C', 'R-O-H'), ('R-C', 'R-N-H'), ('R-C', 'R-C-H'),
                    ('R-O', 'R-C-H'), ('R-N', 'R-C-H')]

category_3_pairs = [('R-O', 'R-O-H'), ('R-O', 'R-N-H'), ('R-N', 'R-O-H'),
                    ('R-N', 'R-N-H')]

# Remaining pairs assigned to Category 1 distributions
remaining_pairs = [('R-O', 'R-N'), ('R-O', 'R-C'), ('R-N', 'R-C'),
                   ('R-O-H', 'R-C-H'), ('R-N-H', 'R-C-H'), ('R-O-H', 'R-N-H')]

# Combine all pairs
all_pairs = category_1_pairs + category_2_pairs + category_3_pairs + remaining_pairs

# Initialize an empty list to store dataframes
df_list = []


# Function to generate data based on category
def generate_data(pair, category):
    n_samples = 1000
    if category == 1:
        # Uniform distributions for distance and angle
        distance = np.random.uniform(0, 1, n_samples)
        angle = np.random.uniform(0, 1, n_samples)
        energy = np.exp(-5 * (1 - np.random.uniform(0, 1, n_samples)))
    elif category == 2:
        # Uniform distributions for distance and angle
        distance = np.random.uniform(0, 1, n_samples)
        angle = np.random.uniform(0, 1, n_samples)
        x_energy = np.random.uniform(0, 1, n_samples)
        energy = np.exp(-20 * (x_energy - 0.6) ** 2)
    elif category == 3:
        # Gaussian distributions for distance and angle centered at 0.5
        distance = np.random.normal(0.5, np.sqrt(1 / (2 * 20)), n_samples)
        angle = np.random.normal(0.5, np.sqrt(1 / (2 * 20)), n_samples)
        distance = np.clip(distance, 0, 1)
        angle = np.clip(angle, 0, 1)
        energy = np.exp(-5 * np.random.uniform(0, 1, n_samples))
    else:
        # For remaining pairs, use Category 1 distributions
        distance = np.random.uniform(0, 1, n_samples)
        angle = np.random.uniform(0, 1, n_samples)
        energy = np.exp(-5 * (1 - np.random.uniform(0, 1, n_samples)))

    # Convert normalized units to physical units
    physical_distance = 4.5 * distance + 0.75  # in angstroms
    physical_angle = 180 * angle - 90  # in degrees

    # Create a dataframe
    df = pd.DataFrame({
        'Molecule_1': [pair[0]] * n_samples,
        'Molecule_2': [pair[1]] * n_samples,
        'Normalized_Distance': distance,
        'Normalized_Angle': angle,
        'Energy': energy,
        'Physical_Distance (Å)': physical_distance,
        'Physical_Angle (°)': physical_angle
    })
    return df


# Generate data for each pair
for pair in all_pairs:
    if pair in category_1_pairs:
        category = 1
    elif pair in category_2_pairs:
        category = 2
    elif pair in category_3_pairs:
        category = 3
    else:
        category = 1  # Assign remaining pairs to Category 1
    df_pair = generate_data(pair, category)
    df_list.append(df_pair)

# Concatenate all dataframes
final_df = pd.concat(df_list, ignore_index=True)

# Save to CSV
final_df.to_csv('small_molecule_interactions_dataset.csv', index=False)

print("Dataset generated and saved to 'molecule_interactions_dataset.csv'.")
