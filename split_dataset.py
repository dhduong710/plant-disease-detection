import os
import shutil
import random
from tqdm import tqdm

def split_dataset(input_dir, output_dir, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, seed = 42):
    random.seed(seed)

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok = True)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_imgs = images[ :n_train]
        val_imgs = images[n_train: n_train + n_val]
        test_imgs = images[n_train + n_val: ]

        for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok = True)
            for img in tqdm(img_list, desc=f"{cls} -> {split}"):
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_cls_dir, img)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    INPUT_DIR = r"D:\PlantVillage"    
    OUTPUT_DIR = r"D:\dataset_split"    

    split_dataset(INPUT_DIR, OUTPUT_DIR)
