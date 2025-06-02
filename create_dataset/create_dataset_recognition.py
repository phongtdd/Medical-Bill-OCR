import os
import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.utils import shuffle
from multiprocessing import Pool, cpu_count
from random import randint
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import gc

class TextRecognitionDataGenerator:
    def __init__(self, text_img_paths, text_labels, gen_img_dir, gen_csv_path, start_idx, proc_id):
        self.text_img_paths = text_img_paths
        self.text_labels = text_labels
        self.gen_img_dir = gen_img_dir
        self.gen_csv_path = gen_csv_path
        self.start_idx = start_idx
        self.proc_id = proc_id
        self.csv_data = []

        # Augmentations
        self.pre_padding_aug = A.Pad(padding = (10, 10, 5, 5), fill=(255, 255, 255), p=1.0)
        self.rotate_aug = A.Rotate(
            limit=4,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
        )
        
        
        self.scale_aug = A.RandomScale(scale_limit=0.2, interpolation=cv2.INTER_LINEAR, p=0.5)
        self.perspective_aug = A.Perspective(scale=(0.02, 0.05), fit_output=True, p=0.5)
        self.brightness_contrast_aug = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        
        self.random_erasing_aug = A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.1, 0.3),
            hole_width_range=(0.1, 0.2),
            fill=(255, 255, 255),
            p=1.0
        )

        self.elastic_aug = A.ElasticTransform(
            alpha=1.0, sigma=50.0,
            border_mode=cv2.BORDER_CONSTANT, p=0.3
        )

        self.downscale = A.Downscale(scale_range=(0.7, 0.9), p=0.5)

        self.padding_left = A.Pad(padding = (0, 0, 0, 5), fill=(255, 255, 255), p=1.0)
        self.padding_right = A.Pad(padding = (0, 0, 5, 0), fill=(255, 255, 255), p=1.0)
        self.padding_top = A.Pad(padding = (0, 5, 0, 0), fill=(255, 255, 255), p=1.0)
        self.padding_bot = A.Pad(padding = (5, 0, 0, 0), fill=(255, 255, 255), p=1.0)

        self.opt_aug = A.OpticalDistortion(
            distort_limit=0.02, interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT, p=0.4
        )

        self.augs = A.Compose(
            [
                A.Compose([self.pre_padding_aug, self.rotate_aug], p = 0.7),
                self.brightness_contrast_aug,              # Điều chỉnh độ sáng/độ tương phản
                self.random_erasing_aug,                   # Che ngẫu nhiên
                self.elastic_aug,                          # Biến dạng đàn hồi nhẹ
                self.downscale,                            # Downscale
                A.SomeOf([self.padding_left, self.padding_right, self.padding_top], p=0.5),       # Padding
                self.opt_aug                               # Biến dạng quang học
            ],
            p=1.0,
        )

        # Góc nghiêng
        self.distortion_scale = 0.1
        self.perspective_transformer = T.RandomPerspective(
            distortion_scale=self.distortion_scale,
            p=0.6,
            interpolation=T.InterpolationMode.NEAREST
        )


    @staticmethod
    def extract_image(image, startpoints, endpoints):
        transformed_img = F.perspective(image, startpoints, endpoints, fill=(255, 255, 255), interpolation=T.InterpolationMode.NEAREST)
        x_coords = [pt[0] for pt in endpoints]
        y_coords = [pt[1] for pt in endpoints]
        ymin, xmin = min(y_coords), min(x_coords)
        height = max(y_coords) - ymin
        width = max(x_coords) - xmin
        cropped = np.asarray(transformed_img)[ymin:ymin + height, xmin:xmin + width, :]
        return cropped

    def generate_perspective_images(self, image, shape, gen_count=6):
        W, H = shape
        seed = randint(0, 1000000)
        torch.manual_seed(seed)

        imgs = []
        for _ in range(gen_count):
            startpoints, endpoints = self.perspective_transformer.get_params(W, H, distortion_scale=self.distortion_scale)
            imgs.append(self.extract_image(image, startpoints, endpoints))

        return imgs

    def process(self):
        print(f"[INFO] Starting process {self.proc_id}")
        NUM_AUG_IMGS = 3
        temp_csv_path = f"{self.gen_csv_path}_temp_{self.proc_id}.csv"  # File CSV tạm thời
        temp_csv_data = []
        
        for idx, (img_path, label) in enumerate(zip(self.text_img_paths, self.text_labels), start=self.start_idx):
            text_img = Image.open(img_path).convert("RGB")
            W, H = text_img.size
            
            max_angle = 4
            padding = int(np.ceil(max(W, H) * np.sin(np.radians(max_angle))))
            self.pre_padding_aug.min_height = H + 2 * padding
            self.pre_padding_aug.min_width = W + 2 * padding
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            perspective_imgs = self.generate_perspective_images(
                image=text_img,
                shape=(W, H),
                gen_count=NUM_AUG_IMGS,
            )

            for i, aug_img in enumerate(perspective_imgs):
                    augmented = self.augs(image=aug_img)
                    final_image = augmented["image"]

                    final_image = final_image[:, :, ::-1]
                    save_name = f"{base_name}_{i:02d}.png"
                    save_path = os.path.join(self.gen_img_dir, save_name)
                    cv2.imwrite(save_path, final_image)

                    temp_csv_data.append({"text": label, "image_name": save_path})
                    
                    del aug_img, augmented, final_image
                    gc.collect()

        pd.DataFrame(temp_csv_data, columns=["text", "image_name"]).to_csv(temp_csv_path, index=False)
        print(f"[INFO] Finishing process {self.proc_id}")
        return temp_csv_path


def chunk_indices(length, n):
    for start in range(0, length, n):
        yield start, min(start + n, length)


def run_worker(payload):
    generator = TextRecognitionDataGenerator(**payload)
    return generator.process()


if __name__ == "__main__":
    start_time = time.perf_counter()
    CLASS_TYPE = "val"
    NUM_AUG_IMGS = 3
    procs = max(cpu_count() - 6, 10)
    
    CSV_PATH = f"vaipe_crops\\{CLASS_TYPE}.csv"
    
    GEN_FOLDER = r"generated_text_recognition"
    GEN_IMG_DIR = os.path.join(GEN_FOLDER, CLASS_TYPE)
    GEN_CSV_PATH = os.path.join(GEN_FOLDER, f"{CLASS_TYPE}.csv")

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {CSV_PATH}: {e}")
        exit(1)

    TEXT_IMGS_PATH = df["image_name"].tolist()
    TEXT_IMGS_PATH = [os.path.join(f"vaipe_crops\\{CLASS_TYPE}", text) for text in TEXT_IMGS_PATH]
    TEXT_LABELS = df["text"].tolist()

    print("[INFO] Number of using processes:", procs)

    os.makedirs(GEN_IMG_DIR, exist_ok=True)

    total_images = len(TEXT_IMGS_PATH)
    print("[INFO] Total text images:", total_images)

    chunk_size = int(np.ceil(total_images / procs))
    payloads = []
    for i in range(procs):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_images)
        payloads.append({
            "text_img_paths": TEXT_IMGS_PATH[start_idx:end_idx],
            "text_labels": TEXT_LABELS[start_idx:end_idx],
            "gen_img_dir": GEN_IMG_DIR,
            "gen_csv_path": GEN_CSV_PATH,
            "start_idx": start_idx,
            "proc_id": i,
        })

    with Pool(processes=procs) as pool:
        temp_csv_paths = pool.map(run_worker, payloads)
        
    all_csv_data = []
    for temp_csv_path in temp_csv_paths:
        temp_df = pd.read_csv(temp_csv_path)
        all_csv_data.append(temp_df)
        os.remove(temp_csv_path)
        
    augmented_df = pd.concat(all_csv_data, ignore_index=True)
    augmented_df.to_csv(GEN_CSV_PATH, index=False)
    print(f"[INFO] Saved augmented CSV to {GEN_CSV_PATH}")

    print("[INFO] Total time:", time.perf_counter() - start_time)