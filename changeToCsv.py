import pandas as pd

# File paths
txt_file = 'ava.txt'
csv_file = 'ground_truth.csv'

# Column names
columns = [
    'index', 'image_id',
    'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5',
    'rating_6', 'rating_7', 'rating_8', 'rating_9', 'rating_10',
    'tag_1', 'tag_2',
    'challenge_id'
]

# Read space-separated txt
df = pd.read_csv(txt_file, sep=' ', header=None, names=columns)

# Save as CSV
df.to_csv(csv_file, index=False)

print(f"Saved {csv_file} with {len(df)} rows")
print(df.head())
