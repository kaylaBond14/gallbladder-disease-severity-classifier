"""
PREPROCESS: Sort raw 9-class gallbladder dataset into 4 severity tiers
========================================================================
The downloaded Kaggle dataset organizes images into 9 folders, one per
disease class. This script copies those images into a new folder
structure organized by the 4 severity tiers used in this project,
then splits each tier into train/val sets.

HOW TO USE:
1. Download and unzip the dataset (see README.md) so you have a folder
   like `data_raw/` containing the 9 original class subfolders.
2. Run this script once WITHOUT the mapping filled in — it will print
   out the exact folder names it finds under RAW_DATA_DIR.
3. Copy those exact names into the TIER_MAPPING dictionary below,
   matching each original class to the correct severity tier.
4. Run the script again — it will copy (not move) images into the
   data/train/ and data/val/ structure the training script expects.
"""

import os
import shutil
import random

RAW_DATA_DIR = "data_raw/Gallblader Diseases Dataset"  # all 11 folders live here
OUTPUT_DIR = "data"            # where the sorted train/val folders will be created
VAL_SPLIT = 0.2                # 20% of each tier's images go to validation
RANDOM_SEED = 42                # keeps the train/val split reproducible

# ----------------------------------------------------------------------
# STEP 1: Tier mapping, filled in from your two datasets.
#
# Entries are (source_folder_path, tier). The source folder path is
# relative to this script's location.
# ----------------------------------------------------------------------
TIER_MAPPING = {
    # --- Tier 1: Normal (from the second dataset, "gallbladder_cancer_data") ---
    # Using both "nml" folders, since we do our own train/val split below
    # anyway — this just maximizes how many normal images we have to work with.
    os.path.join(RAW_DATA_DIR, "nml"): "tier1_normal",
    os.path.join(RAW_DATA_DIR, "nml2"): "tier1_normal",

    # --- Tier 2: Mild pathology (from UIdataGB) ---
    os.path.join(RAW_DATA_DIR, "1Gallstones"): "tier2_mild",
    os.path.join(RAW_DATA_DIR, "6Polyps and cholesterol crystals"): "tier2_mild",

    # --- Tier 3: Inflammation (from UIdataGB) ---
    os.path.join(RAW_DATA_DIR, "3cholecystitis"): "tier3_inflammation",
    os.path.join(RAW_DATA_DIR, "7Adenomyomatosis"): "tier3_inflammation",
    os.path.join(RAW_DATA_DIR, "9Various causes of gallbladder wall thickening"): "tier3_inflammation",

    # --- Tier 4: Severe (from UIdataGB) ---
    os.path.join(RAW_DATA_DIR, "4Membranous and gangrenous cholecystitis"): "tier4_severe",
    os.path.join(RAW_DATA_DIR, "5Perforation"): "tier4_severe",
    os.path.join(RAW_DATA_DIR, "8Carcinoma"): "tier4_severe",

    # Note: "2Abdomen and retroperitoneum" from UIdataGB is intentionally
    # left out here since it doesn't cleanly fit a severity tier. You can
    # add it to tier3_inflammation below if you'd rather keep all 9
    # UIdataGB classes represented:
    # os.path.join(RAW_DATA_DIR, "2Abdomen and retroperitoneum"): "tier3_inflammation",
}

TIERS = ["tier1_normal", "tier2_mild", "tier3_inflammation", "tier4_severe"]


def find_images(folder_path):
    """Recursively find all image files under folder_path, at any nesting depth.
    This handles datasets where images sit directly in the folder, or are
    nested one (or more) levels deeper inside a same-named subfolder."""
    image_paths = []
    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def list_raw_folders():
    """Helper: show folder names found under the raw data directory."""
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"Could not find '{RAW_DATA_DIR}'.")
        return

    folders = sorted([
        f for f in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, f))
    ])

    print(f"\nFound {len(folders)} folders under '{RAW_DATA_DIR}':\n")
    for f in folders:
        folder_path = os.path.join(RAW_DATA_DIR, f)
        n_images = len(find_images(folder_path))
        print(f"  '{f}'  ({n_images} images, including any nested subfolders)")


def sort_into_tiers():
    """Copy images from raw class folders into tier folders, split into train/val."""
    random.seed(RANDOM_SEED)

    # Create the output folder structure
    for split in ["train", "val"]:
        for tier in TIERS:
            os.makedirs(os.path.join(OUTPUT_DIR, split, tier), exist_ok=True)

    tier_counts = {tier: 0 for tier in TIERS}

    for class_path, tier in TIER_MAPPING.items():
        if not os.path.isdir(class_path):
            print(f"WARNING: '{class_path}' not found, skipping.")
            continue

        # Use the last two path segments (e.g. "training_nml" or "1Gallstones")
        # as a filename prefix, so files never collide even when multiple
        # source folders share the same name (like the two "nml" folders).
        path_parts = [p for p in class_path.rstrip("/").split(os.sep) if p]
        source_label = "_".join(path_parts[-2:]).replace(" ", "_")

        images = find_images(class_path)  # recursive — handles nested duplicate folders
        random.shuffle(images)

        split_point = int(len(images) * (1 - VAL_SPLIT))
        train_images = images[:split_point]
        val_images = images[split_point:]

        for i, img_path in enumerate(train_images):
            dst = os.path.join(OUTPUT_DIR, "train", tier, f"{source_label}_{i:04d}_{os.path.basename(img_path)}")
            shutil.copyfile(img_path, dst)

        for i, img_path in enumerate(val_images):
            dst = os.path.join(OUTPUT_DIR, "val", tier, f"{source_label}_{i:04d}_{os.path.basename(img_path)}")
            shutil.copyfile(img_path, dst)

        tier_counts[tier] += len(images)
        print(f"'{class_path}' -> {tier}: {len(train_images)} train, "
              f"{len(val_images)} val")

    print("\n=== Final tier totals ===")
    for tier, count in tier_counts.items():
        print(f"{tier}: {count} images")


if __name__ == "__main__":
    print("Checking your raw data folders...")
    list_raw_folders()
    print("\nIf the folder names/paths above don't match TIER_MAPPING at the "
          "top of this script, update them there first.\n")

    proceed = input("Proceed with sorting into tiers? (y/n): ").strip().lower()
    if proceed == "y":
        sort_into_tiers()
        print("\nDone. Your data/train and data/val folders are ready for train.py.")
    else:
        print("Stopped. Edit TIER_MAPPING as needed, then run this script again.")
