import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGFORMER_CSV = ROOT / "Segformer" / "results" / "model6_crowd" / "per_image_scores_long.csv"
UNET_CSV = ROOT / "U-Net" / "results-2" / "dust_metrics.csv"
csv_files = [SEGFORMER_CSV, UNET_CSV]

levels = {
    "clean": "weighted_iou_clean",
    "light dust": "weighted_iou_light",
    "medium dust": "weighted_iou_medium",
    "heavy dust": "weighted_iou_heavy"
}

dataframes = {}

for file in csv_files:
    df = pd.read_csv(file)

    # Remove suffixes after M1, if any.
    df["image"] = df["image"].astype(str).str.replace(r"(?<=M1).*", "", regex=True)

    # first_row = df.iloc[0]
    df = df[df["image"].astype(str).str.strip() != "image"].copy()
    # df = pd.concat([first_row, df], ignore_index=True)

    # Reshape to one row per image.
    df = df.pivot(index="image", columns="level", values="weighted_iou")
    df = df.reindex(columns=list(levels.keys()))
    df = df.rename(columns=levels).reset_index()

    dataframes[file] = df

# Find images with NaN in either CSV.
invalid_images = set()

for df in dataframes.values():
    missing = df[df.isna().any(axis=1)]
    invalid_images.update(missing["image"])

# Also remove images that are missing entirely from either CSV.
image_sets = [set(df["image"]) for df in dataframes.values()]
common_images = set.intersection(*image_sets)

valid_images = common_images - invalid_images

print(f"Images with NaN in either CSV: {len(invalid_images)}")
print(f"Images shared by both CSVs: {len(common_images)}")
print(f"Images remaining: {len(valid_images)}")

# Filter both CSVs and ensure identical row order.
for file, df in dataframes.items():
    df = df[df["image"].isin(valid_images)]
    df = df.sort_values("image").reset_index(drop=True)

    output = ROOT / "final_weighted_iou_comparison" / f"{file.stem}_collapsed.csv"
    df.to_csv(output, index=False)