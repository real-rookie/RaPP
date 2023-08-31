import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
"""
class MVTec_AD(Dataset):
    # to be tested
    def __init__(self, img_dir, labels_path, transform=None, target_transform=None):
        # transform should at least have a resize
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(f"{self.img_dir}/{idx}.png")
        if image.shape[0] == 1:
            # some images have only 1 channel
            image = image.expand(3, -1, -1)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
            image = image.movedim(0,2)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_data(self):
        image, _ = self[0]
        tensor = torch.tensor(image.unsqueeze(0))
        for index in range(1, self.__len__()):
            image, label = self[index]
            print(label)
            tensor = torch.cat([tensor, image.unsqueeze(0)], 0)
        return tensor
"""

