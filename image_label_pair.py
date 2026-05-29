import os
import pandas as pd

# Paths
csv_file = "ground_truth_dataset_cleaned.csv"
image_folder = "resized_images"

# Load dataset
df = pd.read_csv(csv_file)

# Ensure image_id is str (for file matching)
df["image_id"] = df["image_id"].astype(str)

# Supported extensions
valid_exts = [".jpg", ".jpeg", ".png"]

# Build image-label pairs
pairs = []
for _, row in df.iterrows():
    image_id = row["image_id"]
    mean_score = row["mean_score"]

    # Try possible filenames
    found = False
    for ext in valid_exts:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            pairs.append((image_path, mean_score))
            found = True
            break
    
    if not found:
        print(f"⚠️ Image not found for ID: {image_id}")

# Save as CSV
pairs_df = pd.DataFrame(pairs, columns=["image_path", "label"])
pairs_df.to_csv("image_label_pairs.csv", index=False)

print(f"✅ Created {len(pairs_df)} image–label pairs -> image_label_pairs.csv")
