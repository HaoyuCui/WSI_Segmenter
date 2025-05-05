import cv2
import os
import re

import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, dataset_type='train', train_ratio=0.8):
        self.data_dir = data_dir
        self.file_pairs = self._find_file_pairs()

        # Split dataset into training and validation sets
        train_size = int(train_ratio * len(self.file_pairs))
        val_size = len(self.file_pairs) - train_size
        self.train_dataset, self.val_dataset = random_split(self.file_pairs, [train_size, val_size])

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Choose which dataset to load
        if dataset_type == 'train':
            self.dataset = self.train_dataset
        elif dataset_type == 'val':
            self.dataset = self.val_dataset
        else:
            self.dataset = self.file_pairs

    @staticmethod
    def _find_file_pairs(self):
        file_pairs = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if 'TCGA' in file:
                    continue
                if re.search(rf"\.jpeg$", file):
                    image_path = os.path.join(root, file)  # image ends with .jpeg
                    mask_path = os.path.join(root, file.replace(f".jpeg", ".png"))  # mask ends with .png

                    if os.path.exists(mask_path):
                        file_pairs.append((image_path, mask_path))

        return file_pairs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, mask_path = self.dataset[idx]

        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (256, 256))

        # convert mask to have only 0 and 1
        mask = mask / 255

        # when mask is in red, change the index value if needed
        mask = mask[:, :, 2]

        mask = mask.astype('uint8')
        mask = torch.FloatTensor(mask)
        mask = mask.unsqueeze(0)

        image = self.image_transform(image)

        return image, mask


# TSR-CRC was initially a classification dataset, here I use some tricks to make it a segmentation dataset
class TSR_CRC_Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        data_dirs = os.listdir(data_dir)
        assert 'train' in data_dirs and 'val' in data_dirs, 'Make sure you have downloaded the correct TSR-CRC dataset'

        self.data_dir = os.path.join(data_dir, mode)
        self.image_paths = []
        self.masks = []

        all_subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        assert 'TUM' in all_subdirs, 'TUM folder is missing in the dataset'

        tum_images = [os.path.join(self.data_dir, 'TUM', f) for f in os.listdir(os.path.join(self.data_dir, 'TUM')) if
                      f.endswith('.jpg')]
        tum_sample_count = len(tum_images)

        non_tum_images = []
        category_files = {}
        for subdir in all_subdirs:
            if subdir == 'TUM':
                continue
            subdir_images = [os.path.join(self.data_dir, subdir, f) for f in
                             os.listdir(os.path.join(self.data_dir, subdir)) if f.endswith('.jpg')]
            category_files[subdir] = subdir_images
            non_tum_images.extend(subdir_images)

        non_tum_total_count = len(non_tum_images)

        if non_tum_total_count <= tum_sample_count:
            sampled_non_tum_images = non_tum_images
        else:
            sampled_non_tum_images = []
            per_category_sample = tum_sample_count // len(category_files)

            for subdir, file_list in category_files.items():
                sample_size = min(per_category_sample, len(file_list))
                sampled_non_tum_images.extend(random.sample(file_list, sample_size))

            while len(sampled_non_tum_images) < tum_sample_count:
                remaining_files = list(set(non_tum_images) - set(sampled_non_tum_images))
                if not remaining_files:
                    break
                sampled_non_tum_images.append(random.choice(remaining_files))

        self.image_paths = tum_images + sampled_non_tum_images
        self.masks = [1] * len(tum_images) + [0] * len(sampled_non_tum_images)

        combined = list(zip(self.image_paths, self.masks))
        random.shuffle(combined)
        self.image_paths, self.masks = zip(*combined)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    @staticmethod
    def generate_tum_mask(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, binary_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        mask = 1 - (binary_mask / 255).astype(np.uint8)

        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_value = self.masks[idx]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_value == 1:
            mask = self.generate_tum_mask(image)
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)

        mask = torch.FloatTensor(mask).unsqueeze(0)
        image = self.image_transform(image)

        return image, mask


if __name__ == '__main__':
    dataset = SegmentationDataset(data_dir=r'', dataset_type='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)
    image, mask = next(dataloader_iter)
    print(image.shape, mask.shape)

