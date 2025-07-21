import os
import shutil
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SRC_ROOT = Path('/home/cacc/Repositories/Dataset/PlantDoc-Object-Detection-Dataset')
DST_ROOT = Path('datasets/')
SRC_TRAIN_LABEL = SRC_ROOT / "train_labels.csv"
SRC_TRAIN_IMG_DIR = SRC_ROOT / "TRAIN"
SRC_TEST_LABEL = SRC_ROOT / "test_labels.csv"
SRC_TEST_IMG_DIR = SRC_ROOT / "TEST"


def convert_csv_to_yolo(
    csv_input,
    img_dir: Path,
    dst_root: Path,
    split: str,
    class_to_id: dict
):
    """
    Convert CSV annotations (from DataFrame or CSV path) to YOLO format and copy images.
    """
    if isinstance(csv_input, (str, Path)):
        df = pd.read_csv(csv_input)
    else:
        df = csv_input

    (dst_root / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (dst_root / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    for filename, group in df.groupby('filename'):
        width = group.iloc[0]['width']
        height = group.iloc[0]['height']
        yolo_lines = []

        for _, row in group.iterrows():
            class_id = class_to_id[row['class']]
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_w = (xmax - xmin) / width
            box_h = (ymax - ymin) / height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # Save label file
        label_file = dst_root / f"labels/{split}" / (Path(filename).stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        # Copy image
        src_img_path = img_dir / filename
        dst_img_path = dst_root / f"images/{split}" / filename
        if src_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Image missing: {img_dir / filename}")


def prepare_yolo_dataset():
    # -----------------------
    # Read and split train/val
    # -----------------------
    df = pd.read_csv(SRC_TRAIN_LABEL)
    class_names = sorted(df['class'].unique())
    class_to_id = {cls: i for i, cls in enumerate(class_names)}
    print("Class mapper:", class_to_id)

    # Train/Val split
    unique_images = df['filename'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    train_df = df[df['filename'].isin(train_images)]
    val_df = df[df['filename'].isin(val_images)]

    # -----------------------
    # Convert all splits
    # -----------------------
    convert_csv_to_yolo(train_df, SRC_TRAIN_IMG_DIR, DST_ROOT, "train", class_to_id)
    convert_csv_to_yolo(val_df, SRC_TRAIN_IMG_DIR, DST_ROOT, "val", class_to_id)
    convert_csv_to_yolo(SRC_TEST_LABEL, SRC_TEST_IMG_DIR, DST_ROOT, "test", class_to_id)

    # -----------------------
    # Write data.yaml
    # -----------------------
    yaml_path = DST_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {str(DST_ROOT / 'images/train')}\n")
        f.write(f"val: {str(DST_ROOT / 'images/val')}\n")
        f.write(f"test: {str(DST_ROOT / 'images/test')}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print("Yolo datasets convert completed!")


if __name__ == "__main__":
    prepare_yolo_dataset()
