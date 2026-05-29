import pandas as pd
import matplotlib.pyplot as plt

# Load image-label pairs
df = pd.read_csv("image_label_pairs.csv")

# ✅ Histogram of scores
plt.figure(figsize=(8,5))
plt.hist(df["label"], bins=20, edgecolor="black")
plt.title("Distribution of Mean Scores")
plt.xlabel("Mean Score")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.5)
plt.show()

# ✅ If you want classification, bucket scores (example: 1-3=Low, 4-6=Medium, 7-10=High)
def categorize(score):
    if score <= 3.5:
        return "Low"
    elif score <= 6.5:
        return "Medium"
    else:
        return "High"

df["class"] = df["label"].apply(categorize)

# ✅ Class balance
class_counts = df["class"].value_counts()

plt.figure(figsize=(6,4))
class_counts.plot(kind="bar", color=["#66c2a5","#fc8d62","#8da0cb"])
plt.title("Class Balance")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

print("Class distribution:")
print(class_counts)
