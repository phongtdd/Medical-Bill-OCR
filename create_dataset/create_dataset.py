import os
import cv2
import time
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.utils import shuffle
from multiprocessing import Pool, cpu_count
from random import uniform, randint

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class DataGenerator:
    def __init__(self, doc_img_paths, doc_msk_paths, bck_img_paths, gen_img_dir, gen_msk_dir, start_idx, proc_id):
        self.doc_img_paths = doc_img_paths
        self.doc_msk_paths = doc_msk_paths
        self.bck_img_paths = bck_img_paths
        self.gen_img_dir = gen_img_dir
        self.gen_msk_dir = gen_msk_dir
        self.start_idx = start_idx
        self.proc_id = proc_id

        # Augmentations
        self.motion_blur = A.MotionBlur(blur_limit=(3, 11), p=0.25)
        self.v_flip = A.VerticalFlip(p=1.0)
        self.h_flip = A.HorizontalFlip(p=1.0)
        self.rotate_aug = A.Rotate(limit=30, border_mode=1, interpolation=0, p=0.5)
        self.color_jitter = A.ColorJitter(hue=0.1, saturation=0.1, p=0.7)
        self.opt_aug = A.OpticalDistortion(distort_limit=0.02, interpolation=0, border_mode=0, p=0.6)
        self.grid_aug = A.GridDistortion(num_steps=5, distort_limit=(-0.5, 0.5), interpolation=0, border_mode=0, p=0.9)
        self.elastic_aug = A.ElasticTransform(alpha=150, sigma=13, interpolation=0, border_mode=0, p=0.9)
        self.compression_aug = A.ImageCompression(quality_range=(30, 80), p=1.0)
        self.noise = A.ISONoise(color_shift=(0.01, 0.05), p=0.75)
        self.shadow = A.RandomShadow(shadow_roi=(0.2, 0.2, 0.8, 0.8), shadow_dimension=3, p=0.7)
        self.sunflare = A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 1.0), src_radius=200, src_color=(255, 255, 255), p=0.6)
        self.rgb_shift = A.RGBShift(r_shift_limit=10, g_shift_limit=0, b_shift_limit=5, p=0.4)
        self.channel_shuffle = A.ChannelShuffle(p=0.2)
        self.contrast_1 = A.RandomBrightnessContrast(contrast_limit=(0.1, 0.34), p=0.5)
        self.contrast_2 = A.RandomBrightnessContrast(p=0.5)

        self.augs = A.Compose(
            [
                A.OneOf([self.v_flip, self.h_flip], p=0.6),
                self.rotate_aug,
                self.color_jitter,
                self.contrast_2,
                A.OneOf([self.opt_aug, self.grid_aug, self.elastic_aug], p=0.8),
                A.OneOf([self.noise, self.motion_blur, self.compression_aug], p=0.7),
            ],
            p=1.0,
        )

        self.distortion_scale = 0.55
        self.perspective_transformer = T.RandomPerspective(
            distortion_scale=self.distortion_scale,
            p=0.7,
            interpolation=T.InterpolationMode.NEAREST
        )

    @staticmethod
    def get_random_size(doc_height, doc_width, factor=(1.1, 1.4)):
        size_factor = uniform(factor[0], factor[1])
        new_h, new_w = int(size_factor * doc_height), int(size_factor * doc_width)
        return new_h, new_w

    @staticmethod
    def get_random_crop(image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width + 1
        max_y = image.shape[0] - crop_height + 1
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        return y, x, y + crop_height, x + crop_width

    @staticmethod
    def create_composite(cropped_bck_img, doc_img, doc_msk):
        doc_img = doc_img / 255.0
        mask_inv = np.where(doc_msk == 255, 0.0, 1.0)
        cropped_bck_img_masked = cropped_bck_img * mask_inv
        merged = ((cropped_bck_img_masked + doc_img) * 255).astype(np.int32)
        return merged

    @staticmethod
    def extract_image(image, startpoints, endpoints):
        transformed_img = F.perspective(image, startpoints, endpoints, fill=0, interpolation=T.InterpolationMode.NEAREST)
        x_coords = [pt[0] for pt in endpoints]
        y_coords = [pt[1] for pt in endpoints]
        ymin, xmin = min(y_coords), min(x_coords)
        height = max(y_coords) - ymin
        width = max(x_coords) - xmin
        cropped = np.asarray(transformed_img)[ymin:ymin + height, xmin:xmin + width, :]
        return cropped

    def generate_perspective_images(self, image, mask, shape, gen_count=6):
        W, H = shape
        seed = randint(0, 1000000)
        torch.manual_seed(seed)

        imgs = []
        for _ in range(gen_count):
            startpoints, endpoints = self.perspective_transformer.get_params(W, H, distortion_scale=self.distortion_scale)
            imgs.append(self.extract_image(image, startpoints, endpoints))

        torch.manual_seed(seed)
        msks = []
        for _ in range(gen_count):
            startpoints, endpoints = self.perspective_transformer.get_params(W, H, distortion_scale=self.distortion_scale)
            msks.append(self.extract_image(mask, startpoints, endpoints))

        imgs, msks = shuffle(imgs, msks, random_state=1)
        return imgs, msks

    def process(self):
        print(f"[INFO] Starting process {self.proc_id}")
        NUM_BCK_IMGS = 6
        total_bck_idxs = np.arange(len(self.bck_img_paths))

        for idx, (img_path, msk_path) in enumerate(zip(self.doc_img_paths, self.doc_msk_paths), start=self.start_idx):
            orig_img = Image.open(img_path).convert("RGB")
            orig_msk = Image.open(msk_path).convert("RGB")
            W, H = orig_img.size

            perspective_imgs, perspective_msks = self.generate_perspective_images(
                image=orig_img,
                mask=orig_msk,
                shape=(W, H),
                gen_count=NUM_BCK_IMGS,
            )

            chosen_bck_idxs = np.random.choice(total_bck_idxs, size=NUM_BCK_IMGS, replace=False)
            bck_imgs_chosen = self.bck_img_paths[chosen_bck_idxs]

            for i, bck_path in enumerate(bck_imgs_chosen):
                bck_img = cv2.imread(bck_path, cv2.IMREAD_COLOR)[:, :, ::-1]

                doc_img = perspective_imgs[i]
                # Apply contrast augmentation on document image only
                contrast_aug = self.contrast_2(image=doc_img)
                doc_img = contrast_aug["image"]

                doc_msk = perspective_msks[i].astype(np.int32)
                h, w = doc_img.shape[:2]

                new_h, new_w = self.get_random_size(h, w)
                bck_img = cv2.resize(bck_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                ymin, xmin, ymax, xmax = self.get_random_crop(bck_img, h, w)
                cropped_bck_img = bck_img[ymin:ymax, xmin:xmax, :] / 255.0

                final_image = self.create_composite(cropped_bck_img, doc_img, doc_msk)

                bck_img[ymin:ymax, xmin:xmax, :] = final_image

                new_mask = np.zeros_like(bck_img)
                new_mask[ymin:ymax, xmin:xmax, :] = doc_msk

                augmented = self.augs(image=bck_img, mask=new_mask)
                bck_img = augmented["image"]
                new_mask = augmented["mask"]

                bck_img = bck_img[:, :, ::-1]
                new_mask = new_mask.astype(np.uint8)

                assert len(np.unique(new_mask)) == 2, "Mask should be binary after augmentation"

                save_name = f"{idx:04d}_bck_{i:02d}.png"
                cv2.imwrite(os.path.join(self.gen_img_dir, save_name), bck_img)
                cv2.imwrite(os.path.join(self.gen_msk_dir, save_name), new_mask)

        print(f"[INFO] Finishing process {self.proc_id}")


def chunk_indices(length, n):
    for start in range(0, length, n):
        yield start, min(start + n, length)


def run_worker(payload):
    generator = DataGenerator(**payload)
    generator.process()


if __name__ == "__main__":
    start_time = time.perf_counter()

    DOC_IMG_PATH = r"vaipe-p\public_test\image"
    DOC_MSK_PATH = r"vaipe-p\public_test\mask"
    GEN_FOLDER = r"final_set"
    GEN_IMG_DIR = os.path.join(GEN_FOLDER, "test", "images")
    GEN_MSK_DIR = os.path.join(GEN_FOLDER, "test", "masks")
    BCK_IMGS_DIR = r"background"

    DOC_IMGS = sorted([os.path.join(DOC_IMG_PATH, f) for f in os.listdir(DOC_IMG_PATH)])
    DOC_MSKS = sorted([os.path.join(DOC_MSK_PATH, f) for f in os.listdir(DOC_MSK_PATH)])
    BCK_IMGS = np.array(sorted([os.path.join(BCK_IMGS_DIR, f) for f in os.listdir(BCK_IMGS_DIR)]))


    os.makedirs(GEN_IMG_DIR, exist_ok=True)
    os.makedirs(GEN_MSK_DIR, exist_ok=True)

    total_images = len(DOC_IMGS)
    print("[INFO] Total document images:", total_images)

    procs = max(cpu_count() - 6, 2)
    print("[INFO] Using processes:", procs)

    chunk_size = int(np.ceil(total_images / procs))
    payloads = []
    for i in range(procs):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_images)
        payloads.append({
            "doc_img_paths": DOC_IMGS[start_idx:end_idx],
            "doc_msk_paths": DOC_MSKS[start_idx:end_idx],
            "bck_img_paths": BCK_IMGS,
            "gen_img_dir": GEN_IMG_DIR,
            "gen_msk_dir": GEN_MSK_DIR,
            "start_idx": start_idx,
            "proc_id": i,
        })

    with Pool(processes=procs) as pool:
        pool.map(run_worker, payloads)

    print("[INFO] Total time:", time.perf_counter() - start_time)
