import os
import shutil
import random

random.seed(42)

src_root = '/home/cacc/Repositories/Dataset/PlantDoc-Object-Detection-Dataset/'
dst_root = '/home/cacc/Repositories/Dataset/PlantDoc-Object-Detection-Dataset-Pascal/'
src_dir = src_root + 'TRAIN'

all_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
random.shuffle(all_files)

split_ratio = 0.8
split_index = int(len(all_files) * split_ratio)

train_files = all_files[:split_index]
val_files = all_files[split_index:]

def move_files(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for img_file in files:
        xml_file = img_file.rsplit('.', 1)[0] + '.xml'
        
        # Construct full paths for source image and XML
        src_img_path = os.path.join(src_dir, img_file)
        src_xml_path = os.path.join(src_dir, xml_file)
        
        # Check if both image and XML files exist
        if os.path.exists(src_img_path) and os.path.exists(src_xml_path):
            try:
                # Construct full paths for destination image and XML
                dst_img_path = os.path.join(target_dir, img_file)
                dst_xml_path = os.path.join(target_dir, xml_file)
                
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_xml_path, dst_xml_path)
                # print(f"Moved: {img_file} and {xml_file} to {target_dir}") # Optional: for debugging
            except Exception as e:
                print(f"Error moving {img_file} or {xml_file}: {e}")
        else:
            print(f"Skipping {img_file} (or its XML {xml_file}) because one or both are missing.")

move_files(train_files, dst_root + 'train')
move_files(val_files, dst_root + 'val')

print("Dataset splitting complete. Check the output directories.")