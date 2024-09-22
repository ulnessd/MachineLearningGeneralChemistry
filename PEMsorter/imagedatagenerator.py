import cv2
import numpy as np
import random
import os
from tqdm import tqdm  # For progress bar
import csv

def generate_bead_image(image_size=(256, 256),
                        small_bead_radius=(8, 10),
                        large_bead_radius=(18, 20),
                        num_beads=50,
                        category='mixed',
                        small_prob=0.5):
    """
    Generates a synthetic Polymer Electron Microscopy (PEM) image with beads.

    Parameters:
        image_size (tuple): Size of the image (height, width).
        small_bead_radius (tuple): Min and max radius for small beads.
        large_bead_radius (tuple): Min and max radius for large beads.
        num_beads (int): Number of beads to draw.
        category (str): Category of beads - 'pure_small', 'pure_large', or 'mixed'.
        small_prob (float): Probability of a bead being small in mixed images.

    Returns:
        np.ndarray: Generated synthetic PEM image.
    """
    # Validate category input
    if category not in ['pure_small', 'pure_large', 'mixed']:
        raise ValueError("Invalid category. Choose from 'pure_small', 'pure_large', or 'mixed'.")

    # Create a blank grayscale image (black background)
    image = np.zeros(image_size, dtype=np.uint8)  # Black background

    existing_beads = []  # To keep track of bead positions and radii for overlap checking

    for _ in range(num_beads):
        # Decide bead size based on category
        if category == 'pure_small':
            radius = random.randint(small_bead_radius[0], small_bead_radius[1])
        elif category == 'pure_large':
            radius = random.randint(large_bead_radius[0], large_bead_radius[1])
        elif category == 'mixed':
            if random.random() < small_prob:
                radius = random.randint(small_bead_radius[0], small_bead_radius[1])
            else:
                radius = random.randint(large_bead_radius[0], large_bead_radius[1])

        # Random position ensuring the bead is fully within the image boundaries
        x = random.randint(radius, image_size[1] - radius)
        y = random.randint(radius, image_size[0] - radius)

        # Optional: Prevent excessive overlapping (simple collision detection)
        overlap = False
        for bead in existing_beads:
            distance = np.sqrt((x - bead['x'])**2 + (y - bead['y'])**2)
            if distance < (radius + bead['radius'] + 2):  # Adding a small buffer
                overlap = True
                break
        if overlap:
            continue  # Skip this bead to prevent overlap

        # Add bead to existing_beads list
        existing_beads.append({'x': x, 'y': y, 'radius': radius})

        # Create a bead mask with a base intensity higher than the background
        bead = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)  # Start with black
        base_intensity = 50  # Increased base intensity for better visibility
        cv2.circle(bead, (radius, radius), radius, base_intensity, -1)  # Filled circle

        # Apply radial gradient for shading to simulate depth
        for i in range(1, radius):
            intensity = base_intensity + int((i / radius) * 200)  # Gradient from base_intensity to base_intensity + 100
            intensity = min(intensity, 255)  # Ensure intensity does not exceed 255
            cv2.circle(bead, (radius, radius), radius - i, intensity, 1)  # Draw concentric circles

        # Define the region of interest (ROI) on the main image where the bead will be placed
        roi = image[y - radius:y + radius, x - radius:x + radius]

        # Ensure ROI dimensions match bead dimensions
        if roi.shape[0] != 2 * radius or roi.shape[1] != 2 * radius:
            # Skip drawing beads that would go out of bounds
            continue

        # Combine the bead with the ROI using cv2.max to brighten the bead area
        combined = cv2.max(roi, bead)
        image[y - radius:y + radius, x - radius:x + radius] = combined

    # Add Gaussian noise to simulate real EM image imperfections
    noise_sigma = 0.15  # Adjusted standard deviation of the noise for better visibility
    noise = np.random.normal(0, noise_sigma, image_size).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    return noisy_image

def save_image(image, filename, output_dir='synthetic_images'):
    """
    Saves the generated image to the specified directory.

    Parameters:
        image (np.ndarray): The image to save.
        filename (str): The name of the file.
        output_dir (str): Directory where the image will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path
    filepath = os.path.join(output_dir, filename)

    # Save the image
    cv2.imwrite(filepath, image)
    # Optional: Print statement can be commented out to reduce console clutter
    # print(f"Image saved as '{filepath}'.")

def main():
    """
    Main function to generate and save 8,000 synthetic PEM images based on specified distributions,
    along with their metadata in a CSV file.
    """
    total_images = 64000
    pure_small_ratio = 0.25
    pure_large_ratio = 0.25
    mixed_ratio = 0.50

    num_pure_small = int(total_images * pure_small_ratio)  # 2000
    num_pure_large = int(total_images * pure_large_ratio)  # 2000
    num_mixed = total_images - num_pure_small - num_pure_large  # 4000

    # Define output directory
    base_output_dir = 'synthetic_images_2'
    os.makedirs(base_output_dir, exist_ok=True)

    # Initialize metadata list
    metadata = []

    # Progress bar setup
    with tqdm(total=total_images, desc="Generating Images") as pbar:
        # Generate Pure Small Beads
        for i in range(1, num_pure_small + 1):
            # Random bead count between 40 and 90
            num_beads = random.randint(40, 90)

            # Generate the image
            img = generate_bead_image(
                image_size=(256, 256),
                small_bead_radius=(8, 10),
                large_bead_radius=(18, 20),
                num_beads=num_beads,
                category='pure_small',
                small_prob=0.0  # Not used for pure categories
            )

            # Define unique filename
            filename = f"batch{str(i).zfill(5)}.png"  # e.g., batch00001.png

            # Save the image
            save_image(img, filename, output_dir=base_output_dir)

            # Determine sorting_bin
            sorting_bin = 'pure_small'

            # Collect metadata
            metadata.append({
                'filename': filename,
                'category': 'pure_small',
                'num_beads': num_beads,
                'small_prob': 0.0,
                'sorting_bin': sorting_bin
            })

            # Update progress bar
            pbar.update(1)

        # Generate Pure Large Beads
        for i in range(num_pure_small + 1, num_pure_small + num_pure_large + 1):
            # Random bead count between 40 and 90
            num_beads = random.randint(60, 115)

            # Generate the image
            img = generate_bead_image(
                image_size=(256, 256),
                small_bead_radius=(8, 10),
                large_bead_radius=(18, 20),
                num_beads=num_beads,
                category='pure_large',
                small_prob=0.0  # Not used for pure categories
            )

            # Define unique filename
            filename = f"batch{str(i).zfill(5)}.png"  # e.g., batch2001.png

            # Save the image
            save_image(img, filename, output_dir=base_output_dir)

            # Determine sorting_bin
            sorting_bin = 'pure_large'

            # Collect metadata
            metadata.append({
                'filename': filename,
                'category': 'pure_large',
                'num_beads': num_beads,
                'small_prob': 0.0,
                'sorting_bin': sorting_bin
            })

            # Update progress bar
            pbar.update(1)

        # Generate Mixed Beads
        for i in range(num_pure_small + num_pure_large + 1, total_images + 1):
            # Random bead count between 40 and 90
            num_beads = random.randint(40, 90)

            # Random small_prob between 0.05 and 0.95
            small_prob = random.uniform(0.05, 0.95)

            # Generate the image
            img = generate_bead_image(
                image_size=(256, 256),
                small_bead_radius=(8, 10),
                large_bead_radius=(18, 20),
                num_beads=num_beads,
                category='mixed',
                small_prob=small_prob
            )

            # Define unique filename
            filename = f"batch{str(i).zfill(5)}.png"  # e.g., batch4001.png

            # Save the image
            save_image(img, filename, output_dir=base_output_dir)

            # Determine sorting_bin based on small_prob
            if small_prob >= 0.92:
                sorting_bin = 'acceptable_small'  # <8% large beads
            elif small_prob <= 0.08:
                sorting_bin = 'acceptable_large'  # <8% small beads
            else:
                sorting_bin = 'reject'  # >8% small and large beads

            # Collect metadata
            metadata.append({
                'filename': filename,
                'category': 'mixed',
                'num_beads': num_beads,
                'small_prob': round(small_prob, 2),
                'sorting_bin': sorting_bin
            })

            # Update progress bar
            pbar.update(1)

    # Write metadata to CSV
    csv_filename = 'metadata02.csv'
    fieldnames = ['filename', 'category', 'num_beads', 'small_prob', 'sorting_bin']
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in metadata:
            writer.writerow(data)

    print(f"All {total_images} images have been generated and saved in the '{base_output_dir}' directory.")
    print(f"Metadata02 CSV '{csv_filename}' has been created.")

if __name__ == "__main__":
    main()
