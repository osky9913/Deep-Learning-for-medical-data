import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config_global import landmark_class, UNET_LANDMARK_IMAGE_SIZE
import matplotlib.pyplot as plt





class TeethDataset(Dataset):
    def __init__(self, imgs, transform=None , scale = 1):
        self.imgs = imgs
        self.transform = transform
        self.scale      = scale

    def __len__(self):
        return len(self.imgs)

    def preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, idx):
        size = UNET_LANDMARK_IMAGE_SIZE
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB").resize(size)
        img_path = self.imgs[idx]
        mask = np.zeros(size)


        for index in range(len(landmark_class)):
            mask_path = img_path.replace('input', landmark_class[index] + os.sep + 'binary_mask')
            #print(mask_path)
            if os.path.exists(mask_path):
                mask_bin = np.array(Image.open(mask_path).resize(size))
                mask_bin_fixed = np.where(mask_bin != 0, 1, 0)
                mask[mask_bin_fixed == 1] = index + 1

        """
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        plt.show()
        """

        mask = Image.fromarray(mask)
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}



