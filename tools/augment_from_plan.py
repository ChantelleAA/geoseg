# this script also augments tifs

import os
import cv2
import albumentations as A
from tqdm import tqdm
import numpy as np
import rasterio
import pandas as pd


#  Directories 
mask_dir = "../data/Biodiversity_tiff/Train/masks"                # Corresponding masks
tif_image_dir = "../data/Biodiversity_tiff/Train/image"  # TIFF satellite images

output_mask_dir = "../data/Biodiversity_tiff/Train/augmented/masks"
output_tif_dir = "../data/Biodiversity_tiff/Train/augmented/image"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_tif_dir, exist_ok=True)


# augmentation plan (created by create_augmentation_plan.py)
plan_csv = "augmentation_plan.csv"
if not os.path.exists(plan_csv):
    raise FileNotFoundError(f"{plan_csv} not found.")
plan_df = pd.read_csv(plan_csv)


# transformation definition
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
    #A.GaussNoise(p=0.2)
], additional_targets={"tif": "image"})



for _, row in tqdm(plan_df.iterrows(), total=len(plan_df)):
    filename = row["filename"]
    n_augments = int(row["n_augments"])


    stem = os.path.splitext(filename)[0]
    mask_path = os.path.join(mask_dir, filename) 
    tif_path = os.path.join(tif_image_dir, f"{stem}.tif")


    if not os.path.exists(mask_path):
        print(f"[Warning] Missing mask for {filename}, skipping.")
        continue


    # read in mask
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"[Warning] Could not read image/mask for {filename}, skipping.")
        continue

    # Read full multi-band TIFF (preserve all layers)
    if not os.path.exists(tif_path):
        print(f"[Warning] No TIFF found for {filename}, skipping TIFF.")
        tif_data = None
    else:
        with rasterio.open(tif_path) as src:
            tif_data = src.read()  # Shape: (C, H, W)
            tif_meta = src.meta.copy()

        # Albumentations expects H×W×C format
        tif_data = np.transpose(tif_data, (1, 2, 0))



    # Apply augmentation n_augments times 
    for i in range(n_augments): 
        augmented = transform(image=tif_data, mask=mask) 
        aug_tif = augmented["image"] 
        aug_mask = augmented["mask"] 
        
        # Save augmented mask 
        aug_mask_filename = f"{stem}_aug{i+1}.png" 
        cv2.imwrite(os.path.join(output_mask_dir, aug_mask_filename), aug_mask) 

        # Save augmented TIFF 
        aug_tif = np.transpose(aug_tif, (2, 0, 1)) 
        aug_tif_filename = f"{stem}_aug{i+1}.tif" 
        with rasterio.open(os.path.join(output_tif_dir, aug_tif_filename), "w", **tif_meta) as dst: 
            dst.write(aug_tif)