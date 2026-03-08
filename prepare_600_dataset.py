import random
import shutil
from pathlib import Path

# correct paths
SOURCE_DATASET = "C:/Users/ishug/OneDrive/Desktop/flame"
OUTPUT_DATASET = "C:/Users/ishug/OneDrive/Desktop/flame_dataset_600"

TOTAL_IMAGES = 600
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

src = Path(SOURCE_DATASET)
out = Path(OUTPUT_DATASET)

# source folders
source_train_images = src / "train" / "images"
source_train_labels = src / "train" / "labels"

source_valid_images = src / "valid" / "images"
source_valid_labels = src / "valid" / "labels"

source_test_images = src / "test" / "images"
source_test_labels = src / "test" / "labels"

# output folders
train_img = out / "train" / "images"
train_lbl = out / "train" / "labels"

valid_img = out / "valid" / "images"
valid_lbl = out / "valid" / "labels"

test_img = out / "test" / "images"
test_lbl = out / "test" / "labels"

for folder in [train_img, train_lbl, valid_img, valid_lbl, test_img, test_lbl]:
    folder.mkdir(parents=True, exist_ok=True)

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

pairs = []

def collect_pairs(images_dir, labels_dir):
    collected = []
    if not images_dir.exists():
        print(f"Warning: images folder not found: {images_dir}")
        return collected
    if not labels_dir.exists():
        print(f"Warning: labels folder not found: {labels_dir}")
        return collected

    for img_path in images_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in image_exts:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                collected.append((img_path, label_path))
    return collected

pairs.extend(collect_pairs(source_train_images, source_train_labels))
pairs.extend(collect_pairs(source_valid_images, source_valid_labels))
pairs.extend(collect_pairs(source_test_images, source_test_labels))

print(f"Found {len(pairs)} image-label pairs.")

if len(pairs) < TOTAL_IMAGES:
    raise ValueError(f"Only {len(pairs)} valid image-label pairs found, less than {TOTAL_IMAGES}.")

random.shuffle(pairs)
selected = pairs[:TOTAL_IMAGES]

train_count = int(TOTAL_IMAGES * TRAIN_SPLIT)
valid_count = int(TOTAL_IMAGES * VAL_SPLIT)
test_count = TOTAL_IMAGES - train_count - valid_count

train_pairs = selected[:train_count]
valid_pairs = selected[train_count:train_count + valid_count]
test_pairs = selected[train_count + valid_count:]

def copy_pairs(pairs_list, img_dst, lbl_dst):
    for img_path, lbl_path in pairs_list:
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

copy_pairs(train_pairs, train_img, train_lbl)
copy_pairs(valid_pairs, valid_img, valid_lbl)
copy_pairs(test_pairs, test_img, test_lbl)

yaml_path = out.as_posix()

yaml_text = f"""path: {yaml_path}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['fire']
"""

with open(out / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

print("Dataset created successfully.")
print(f"Train: {len(train_pairs)} images")
print(f"Valid: {len(valid_pairs)} images")
print(f"Test: {len(test_pairs)} images")
print(f"Saved at: {out}")