#!/usr/bin/env python3
"""Prepare the GroceryStoreDataset into ImageFolder format for our pipeline."""

import csv
import shutil
import sys
from pathlib import Path


def main():
    repo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/grocery_data")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw")

    dataset_dir = repo_dir / "dataset"

    # Load class names from classes.csv
    # Format: Class Name (str), Class ID (int), Coarse Class Name, Coarse Class ID, ...
    classes_file = dataset_dir / "classes.csv"
    class_names = {}
    if classes_file.exists():
        with open(classes_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    class_name = row[0].strip()
                    class_id = int(row[1].strip())
                    class_names[class_id] = class_name

    print(f"Loaded {len(class_names)} classes from {classes_file}")

    # Process each split
    # train.txt format: "train/Fruit/Apple/Golden-Delicious/img.jpg, 0, 0"
    for split in ["train", "val", "test"]:
        split_file = dataset_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping")
            continue

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                img_path = parts[0].strip()
                fine_label = int(parts[1].strip())

                # Get class name or use label number
                class_name = class_names.get(fine_label, f"class_{fine_label:03d}")

                # Source image path (relative to dataset dir)
                src = dataset_dir / img_path
                if not src.exists():
                    continue

                # Destination: output/split/class_name/image.jpg
                dst_dir = split_dir / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name

                shutil.copy2(src, dst)
                count += 1

        print(f"{split}: {count} images -> {split_dir}")

    # Summary
    train_dir = output_dir / "train"
    if train_dir.exists():
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"\nTotal classes: {len(classes)}")
        print(f"Sample: {classes[:10]}")


if __name__ == "__main__":
    main()
