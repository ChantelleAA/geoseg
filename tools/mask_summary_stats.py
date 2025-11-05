import os
import numpy as np
import pandas as pd
from PIL import Image


# before this you need to run biodiversity_mask_conver
# Directory containing segmentation label masks in PNG format
image_dir = "../data/Biodiversity_tiff/Train/masks_png_convert_rgb"

# Define RGB color codes corresponding to each semantic class
# The keys are class indices; values are RGB triplets
class_colors = {
    0: [11, 246, 210],    # ignore (background or unlabeled area)
    1: [250, 62, 119],    # forestland
    2: [168, 232, 84],    # grassland
    3: [242, 180, 92],    # cropland
    4: [116, 116, 116],   # settlement (urban areas)
    5: [255, 214, 33],    # seminatural grassland
}

# Get a list of all PNG mask filenames in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]


# Total number of classes (based on dictionary size)
n_classes = len(class_colors)

# Prepare an empty list to store per-image class statistics
records_prec = []
records_count = []

# Loop through every mask image
for fname in image_files:
    path = os.path.join(image_dir, fname)
    
    # Load image and convert to NumPy array (shape: [height, width, 3])
    img = np.array(Image.open(path))
    
    # Initialize a count array for each class in this image
    class_counts = np.zeros(n_classes, dtype=int)
    
    # Loop through each class and count how many pixels match its color
    for cls_idx, color in class_colors.items():
        # Create a boolean mask where pixels match the RGB color
        match = np.all(img == color, axis=-1)
        # Sum True values to get total pixel count for that class
        class_counts[cls_idx] = match.sum()
    
    # Compute total number of pixels in the image
    total_pixels = img.shape[0] * img.shape[1]

    # Compute proportion (fraction) of pixels per class
    class_props = class_counts / total_pixels
    
    # Store results for this image in a dictionary
    record_prec = {"filename": fname}
    record_count =  {"filename": fname}
    for c in range(n_classes):
        # Convert fraction to percentage and store under class_X_pct
        record_prec[f"class_{c}_pct"] = class_props[c]
        record_count[f"class_{c}_count"] = class_counts[c]

    # Append per-image record to the master list
    records_prec.append(record_prec)
    records_count.append(record_count)

df_prec = pd.DataFrame(records_prec)
df_count = pd.DataFrame(records_count)

# Compute mean percentage of each class across all images
prec_mean = df_prec[[f"class_{c}_pct" for c in range(n_classes)]].mean()
count_mean = df_count[[f"class_{c}_count" for c in range(n_classes)]].mean()

# Save per-image results to CSV
df_prec.to_csv("class_prec_per_image.csv", index=False)
df_count.to_csv("class_count_per_image.csv", index=False)
