import pandas as pd

csv_path = "class_prec_per_image.csv"   # input CSV
output_path = "augmentation_plan.csv"
target_share = 0.5                       # desired minimum % per class
max_multiplier = 10                     # cap augmentation multiplier
augmentation_budget_factor = 1           # fraction of dataset to add

df = pd.read_csv(csv_path)
image_col = df.columns[0]
class_cols = df.columns[1:]

# Compute mean share per class (Calculates the average share of each class across all images)
class_means = df[class_cols].mean()

# Finds classes that are under the target share
rare_classes = class_means[class_means < target_share].index

# Compute augmentation factors for rare classes only
augmentation_factors = (target_share / class_means[rare_classes]).clip(upper=max_multiplier)

# Compute per-image augmentation score for rare classes
# Each image gets a score based on how much it can help balance rare classes
df["augment_score"] = (df[rare_classes] * augmentation_factors).sum(axis=1)

# Only keep images with a positive augment score
df_to_augment = df[df["augment_score"] > 0].copy()

# Convert scores to integer augmentation counts
total_aug = int(len(df) * augmentation_budget_factor)
df_to_augment["n_augments"] = (df_to_augment["augment_score"] / df_to_augment["augment_score"].sum() * total_aug).round().astype(int)

# Save augmentation plan
df_to_augment[[image_col, "n_augments"]].to_csv(output_path, index=False)
print(f"Augmentation plan saved to {output_path}")
print(f"Number of images to augment: {len(df_to_augment)}")
