import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2
from sklearn.cluster import DBSCAN
import os
import csv


def generate_lattice_points(size, spacing):
    """Generate a list of lattice points."""
    lattice_points = []
    for i in range(0, size, spacing):
        for j in range(0, size, spacing):
            lattice_points.append((i, j))
    return lattice_points


def filter_lattice_points(lattice_points, threshold):
    """Filter lattice points based on a random threshold."""
    filtered_points = []
    for point in lattice_points:
        if np.random.rand() > threshold:
            filtered_points.append(point)
    return filtered_points


def place_gaussians(lattice_points, size, width):
    """Place Gaussians at the specified lattice points."""
    lattice = np.zeros((size, size))
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    X, Y = np.meshgrid(x, y)

    for point in lattice_points:
        lattice += np.exp(-((X - point[0]) ** 2 + (Y - point[1]) ** 2) / (2 * width ** 2))

    return lattice


def compute_defect_percentage(original_points, filtered_points):
    """Calculate the percentage of defects (missing points)."""
    defect_percentage = 100 * (len(original_points) - len(filtered_points)) / len(original_points)
    return defect_percentage


def measure_cluster_sizes_and_edges(lattice_points, spacing):
    """Measure the size of defect clusters using DBSCAN and calculate edge points."""
    if not lattice_points:
        return 0, 0, 0  # No clusters or edges if no points are left

    # Convert lattice points to a 2D array for clustering
    points_array = np.array(lattice_points)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=spacing, min_samples=2).fit(points_array)
    labels = clustering.labels_

    # Calculate cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = counts[unique_labels != -1]  # Exclude noise

    if len(cluster_sizes) == 0:
        return 0, 0, 0  # No clusters if only noise is detected

    avg_cluster_size = np.mean(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)

    # Calculate edge points
    total_edge_points = 0
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        cluster_points = points_array[labels == label]
        for point in cluster_points:
            neighbors = np.sum(np.linalg.norm(cluster_points - point, axis=1) < spacing * 1.5)
            if neighbors < 4:  # Assuming a 2D lattice with up to 4 neighbors in a perfect square lattice
                total_edge_points += 1

    return avg_cluster_size, max_cluster_size, total_edge_points


def compute_2d_ft(lattice):
    """Compute the 2D Fourier Transform of the lattice."""
    ft = fftshift(fft2(lattice))
    magnitude = np.abs(ft)
    return magnitude


def save_ft_image(ft_magnitude, filename):
    """Save the Fourier Transform as a PNG image."""
    plt.imshow(np.log(1 + ft_magnitude), cmap='gray', origin='lower')
    plt.axis('off')  # Hide axes for the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_zoomed_ft_image(ft_magnitude, filename, zoom_factor=8):
    """Save a zoomed-in region of the Fourier Transform as a PNG image."""
    center = ft_magnitude.shape[0] // 2
    half_size = ft_magnitude.shape[0] // (2 * zoom_factor)
    zoomed_region = ft_magnitude[center - half_size:center + half_size, center - half_size:center + half_size]

    plt.imshow(np.log(1 + zoomed_region), cmap='gray', origin='lower')
    plt.axis('off')  # Hide axes for the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# Parameters
size = 1000  # Size of the lattice grid
spacing = 32  # Spacing between lattice points
width = spacing / 5  # Width of the Gaussian
num_samples = 5000  # Number of training set entries to generate
start_sample = 0  # Start count, change this if you need to resume after a crash
output_dir = 'ft_images'  # Directory to save Fourier Transform images
csv_filename = 'ft_metadata.csv'  # CSV file to save metadata

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the CSV file for appending or writing if starting fresh
csv_mode = 'a' if start_sample > 0 else 'w'
with open(csv_filename, mode=csv_mode, newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header only if starting fresh
    if start_sample == 0:
        csv_writer.writerow([
            'FullFTFilename', 'ZoomedFTFilename',
            'DefectPercentage', 'AvgClusterSize', 'MaxClusterSize', 'TotalEdgePoints'
        ])

    for i in range(start_sample, num_samples):
        # Randomly choose a threshold for defect creation
        threshold = np.random.uniform(0.0005, 0.5)

        # Generate lattice points
        original_points = generate_lattice_points(size, spacing)

        # Filter lattice points based on the threshold
        filtered_points = filter_lattice_points(original_points, threshold)

        # Place Gaussians at the filtered lattice points
        lattice = place_gaussians(filtered_points, size, width)

        # Compute the defect percentage
        defect_percentage = compute_defect_percentage(original_points, filtered_points)

        # Measure cluster sizes and total edge points
        avg_cluster_size, max_cluster_size, total_edge_points = measure_cluster_sizes_and_edges(filtered_points,
                                                                                                spacing)

        # Compute the 2D Fourier Transform of the lattice
        ft_magnitude = compute_2d_ft(lattice)

        # Create unique filenames for the images
        full_ft_filename = f'ft_full_{i:03d}.png'
        zoomed_ft_filename = f'ft_zoomed_{i:03d}.png'
        full_ft_filepath = os.path.join(output_dir, full_ft_filename)
        zoomed_ft_filepath = os.path.join(output_dir, zoomed_ft_filename)

        # Save the full Fourier Transform image
        save_ft_image(ft_magnitude, full_ft_filepath)

        # Save the zoomed-in Fourier Transform image
        save_zoomed_ft_image(ft_magnitude, zoomed_ft_filepath)

        # Write the metadata to the CSV file
        csv_writer.writerow([
            full_ft_filename, zoomed_ft_filename,
            defect_percentage, avg_cluster_size, max_cluster_size, total_edge_points
        ])

        print(f"Sample {i + 1} saved: {full_ft_filename}, {zoomed_ft_filename}")

print(f'Training set generation completed. Images saved to "{output_dir}" and metadata saved to "{csv_filename}".')
