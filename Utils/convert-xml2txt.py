import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split

# ✅ Path config
ROOT = Path(".")
ANNOTATIONS = [
    "/home/cacc/Repositories/Dataset/VOC2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
    "/home/cacc/Repositories/Dataset/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations"
]
IMAGES = [
    "/home/cacc/Repositories/Dataset/VOC2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    "/home/cacc/Repositories/Dataset/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
]
IMAGESETS_MAIN = "/home/cacc/Repositories/Dataset/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main"

# ✅ Create YOLO directory structure
for d in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# ✅ VOC class names
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}

# ✅ Parse annotation
def convert_voc_to_yolo(ann_file, out_file, img_w, img_h):
    tree = ET.parse(ann_file)
    root = tree.getroot()

    yolo_labels = []

    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in class_to_id:
            continue
        cls_id = class_to_id[cls]

        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(out_file, "w") as f:
        f.write("\n".join(yolo_labels))

# ✅ Collect image and annotation pairs
all_images = []
for img_dir, ann_dir in zip(IMAGES, ANNOTATIONS):
    for filename in os.listdir(ann_dir):
        img_id = filename.replace(".xml", "")
        img_path = os.path.join(img_dir, img_id + ".jpg")
        ann_path = os.path.join(ann_dir, filename)

        if os.path.exists(img_path):
            all_images.append((img_path, ann_path))

# ✅ Split into train and val
train_data, val_data = train_test_split(all_images, test_size=0.2, random_state=42)

# ✅ Process function
def process_data(data_split, split_name):
    for img_path, ann_path in data_split:
        img_id = Path(img_path).stem
        out_img = ROOT / f"images/{split_name}/{img_id}.jpg"
        out_label = ROOT / f"labels/{split_name}/{img_id}.txt"

        shutil.copy(img_path, out_img)

        # Get image size
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        convert_voc_to_yolo(ann_path, out_label, w, h)

process_data(train_data, "train")
process_data(val_data, "val")