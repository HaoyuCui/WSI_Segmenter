import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import time
import argparse

import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from dataset import SegmentationDataset


def compute_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)
    return dice.mean().item()


def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


if __name__ == '__main__':
    vis = visdom.Visdom()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'.\eg')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    data_dir = args.data_dir
    train_ratio = args.train_ratio
    epochs = args.epochs

    seg_model = deeplabv3_resnet50(weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=1).to('cuda')

    criterion = nn.BCELoss().to('cuda')
    optimizer = optim.Adam(seg_model.parameters(), lr=5e-4)
    dataset_train = SegmentationDataset(data_dir=data_dir,
                                        dataset_type='train',
                                        train_ratio=train_ratio)
    dataset_val = SegmentationDataset(data_dir=data_dir,
                                      dataset_type='val',
                                      train_ratio=train_ratio)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, drop_last=True)

    saving_index = 0
    for epoch in range(epochs):
        saving_index += 1
        index = 0
        epoch_loss = 0
        start = time.time()

        train_dice = 0
        train_iou = 0

        seg_model.train()

        epoch_pbar = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{epochs}', leave=True)

        for image, mask in epoch_pbar:
            index += 1
            start = time.time()
            image = image.to('cuda')
            mask = mask.to('cuda')

            optimizer.zero_grad()
            output = seg_model(image)
            output = torch.sigmoid(output['out'])

            loss = criterion(output, mask)

            loss.backward()
            iter_loss = loss.item()
            epoch_loss += iter_loss
            optimizer.step()

            # Compute Dice and IoU
            dice = compute_dice(output, mask)
            iou = compute_iou(output, mask)
            train_dice += dice
            train_iou += iou

            output_np = output.cpu().data.numpy().copy()
            y_np = mask.cpu().data.numpy().copy()
            image_np = image.cpu().data.numpy().copy()

            if np.mod(index, 20) == 1:
                epoch_pbar.set_description_str(f'epoch {epoch}, {index}/{len(dataloader_train)}, loss: {iter_loss:.4f}')

                # 转换为可视化格式
                output_grid = vutils.make_grid(torch.tensor(output_np), normalize=True)
                mask_grid = vutils.make_grid(torch.tensor(y_np), normalize=True)
                image_grid = vutils.make_grid(torch.tensor(image_np), normalize=True)

                vis.images(image_grid.numpy(), win='image', opts=dict(title='Input Image'))
                vis.images(output_grid.numpy(), win='pred', opts=dict(title='Prediction'))
                vis.images(mask_grid.numpy(), win='label', opts=dict(title='Ground Truth'))

        train_dice /= len(dataloader_train)
        train_iou /= len(dataloader_train)
        print('=========== Summary ===========')
        print('Train Dice: {:.4f}, Train IoU: {:.4f}'.format(train_dice, train_iou))
        print('Epoch loss = %f' % (epoch_loss / len(dataloader_train)))

        torch.save(seg_model.state_dict(), f'checkpoints/epoch_{epoch}.pt')
        print(f'Saving to checkpoints/epoch_{epoch}.pt')
        print('===============================')

        # validation
        seg_model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            for image, mask in dataloader_val:
                image = image.to('cuda')
                mask = mask.to('cuda')

                output = seg_model(image)
                output = torch.sigmoid(output['out'])

                loss = criterion(output, mask)
                val_loss += loss.item()

                # Compute Dice and IoU
                dice = compute_dice(output, mask)
                iou = compute_iou(output, mask)
                val_dice += dice
                val_iou += iou

        val_dice /= len(dataloader_val)
        val_iou /= len(dataloader_val)
        print('Validation Dice: {:.4f}, Validation IoU: {:.4f}'.format(val_dice, val_iou))
        print('Validation loss = %f' % (val_loss / len(dataloader_val)))
