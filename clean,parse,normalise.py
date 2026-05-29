import pandas as pd

# Load dataset
df = pd.read_csv("ground_truth_dataset_mirrored.csv")

# --- 1. Standardize column names ---
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"image_num": "image_id"})

# --- 2. Drop duplicates & handle NaN ---
df = df.drop_duplicates(subset=["image_id"])
df = df.fillna(0)

# --- 3. Ensure numeric votes ---
vote_cols = [col for col in df.columns if col.startswith("vote_")]
df[vote_cols] = df[vote_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# --- 4. Normalize votes ---
df["total_votes"] = df[vote_cols].sum(axis=1)
df[vote_cols] = df[vote_cols].div(df["total_votes"].replace(0, 1), axis=0)

# --- 5. Compute mean_score (weighted average) ---
weights = list(range(1, 11))  # ratings 1–10
df["mean_score"] = df[vote_cols].mul(weights, axis=1).sum(axis=1)

# --- 6. Save cleaned dataset ---
df.to_csv("ground_truth_dataset_cleaned.csv", index=False)

print("✅ Saved cleaned & normalized dataset as ground_truth_dataset_cleaned.csv")
