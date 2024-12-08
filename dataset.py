import cv2
import os
import re

import torch
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

        # Choose which dataset to load
        if dataset_type == 'train':
            self.dataset = self.train_dataset
        elif dataset_type == 'val':
            self.dataset = self.val_dataset
        else:
            self.dataset = self.file_pairs

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

        image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image = image_transform(image)

        return image, mask


if __name__ == '__main__':
    dataset = SegmentationDataset(data_dir=r'F:\Data\endometrium_pathology\annotations', dataset_type='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)
    image, mask = next(dataloader_iter)
    print(image.shape, mask.shape)

