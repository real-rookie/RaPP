import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms as T
import torch

class MVTec_AD(Dataset):
    def __init__(self, img_dir, labels_path, transform=None, target_transform=None):
        self.img_dir = img_dir + "MVTec-AD"
        self.img_labels = pd.read_csv(labels_path + "labels.csv")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(f"{self.img_dir}/{idx}.png")
        if image.shape[0] == 1:
            # some images have only 1 channel
            image = image.expand(3, -1, -1)
        image = T.functional.resize([1024, 1024], image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        image = image.movedim(0,2)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    @property
    def data_targets(self):
        images, labels = self[0]
        images = torch.tensor(images)
        labels = [labels]
        for index in range(1, self.__len__()):
            image, label = self[index]
            images = torch.stack([images, image], 0)
            labels.append(label)
        labels = torch.tensor(labels)
        return images, labels


