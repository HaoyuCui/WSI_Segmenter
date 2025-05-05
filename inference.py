import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

target_dir = r'PATH/TO/PATCHES'
output_dir = r'PATH/TO/SEG&OVERLAY'

ALPHA = 0.5

if __name__ == '__main__':
    print('available GPUs:', torch.cuda.device_count())
    cuda_available = torch.cuda.is_available()
    print('use cuda : {}'.format(cuda_available))
    if cuda_available:
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = deeplabv3_resnet50(num_classes=1)
    model.load_state_dict(torch.load('checkpoints/tsr_crc.pt'))
    model = model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    for img_name in os.listdir(target_dir):
        if not img_name.endswith('.png'):
            continue

        img_path = os.path.join(target_dir, img_name)

        img = cv2.imread(img_path)
        # crop to 256x256
        # img = img[256:512, 256:512, :]
        img = cv2.resize(img, (256, 256))  # we strongly recommend to extract patches at 20x or higher
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = torch.sigmoid(output['out'])
            output_bool = output.mean() > 0.5  # we use a boolean value to define whether the tile contains tumor or not

        output = output.squeeze(0).cpu().numpy() * 255
        mask = output.astype(np.uint8)

        img_cv = np.array(img)  # (H, W, 3)
        mask_color = np.zeros_like(img_cv)
        mask_color[:, :, 1] = mask

        overlay = cv2.addWeighted(img_cv, 1 - ALPHA, mask_color, ALPHA, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f'Overlay: Tile contains tumor? {output_bool.item()}')
        plt.axis("off")

        plt.savefig(os.path.join(output_dir, img_name))

        print(f'{img_name} done!')
