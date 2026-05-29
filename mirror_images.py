import pandas as pd
import os
import shutil

# ---------------------- Paths ----------------------
ground_truth_csv = 'ground_truth.csv'           # Converted from ava.txt
dataset_csv = 'ground_truth_dataset.csv'       # CSV to mirror with
original_images = 'resized_images/'            # Folder with your original images
mirrored_images = 'images_mirrored/'                # Folder for mirrored images

# Create mirrored images folder if it doesn't exist
os.makedirs(mirrored_images, exist_ok=True)

# ---------------------- Load CSVs ----------------------
gt_df = pd.read_csv(ground_truth_csv)
dataset_df = pd.read_csv(dataset_csv)

# ---------------------- Standardize column names ----------------------
gt_df.columns = gt_df.columns.str.strip().str.lower()
dataset_df.columns = dataset_df.columns.str.strip().str.lower()

# Rename image_num → image_id
if 'image_num' in dataset_df.columns:
    dataset_df = dataset_df.rename(columns={'image_num': 'image_id'})

# ---------------------- Ensure compatibility ----------------------
if 'image_id' not in gt_df.columns or 'image_id' not in dataset_df.columns:
    raise ValueError("Both CSVs must have an 'image_id' column (or one renamed).")

# ---------------------- Mirror Process ----------------------
# Keep only rows in dataset_df that exist in gt_df
mirrored_df = dataset_df[dataset_df['image_id'].isin(gt_df['image_id'])]

# Save mirrored CSV
mirrored_csv_path = 'ground_truth_dataset_mirrored.csv'
mirrored_df.to_csv(mirrored_csv_path, index=False)
print(f"Mirrored CSV saved as {mirrored_csv_path}")

# ---------------------- Mirror Images ----------------------
for _, row in mirrored_df.iterrows():
    img_id = str(row['image_id']) + ".jpg"  # adjust extension if needed
    src = os.path.join(original_images, img_id)
    dst = os.path.join(mirrored_images, img_id)
    if os.path.exists(src):
        shutil.copy(src, dst)

print(f"Mirrored images saved in {mirrored_images}")
