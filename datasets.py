from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class Dataset(Dataset):
    def __init__(self, data_dir, augment=False, image_size=256, channels=3, depth=8):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = list(
            filter(
                lambda x: os.path.splitext(x)[-1] in (".png", "tif", ".jpg", "jpeg"),
                sorted(glob(os.path.join(self.data_dir, "*.*"))),
            )
        )
        self.depth = depth
        self.channels = channels
        self.augment = augment

    def __getitem__(self, item):
        image_ori = self.image_path[item]

        image = Image.open(image_ori)
        if self.channels == 3 and self.depth == 8:
            image = image.convert("RGB")
        elif self.channels == 1:
            image = image.convert("F")

        if self.augment:
            transform = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    np.array,
                    transforms.ToTensor(),
                    # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Dataset(train_data_dir, image_size)
    test_dataset = Dataset(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_train_loader(train_data_dir, image_size, batch_size):
    train_dataset = Dataset(train_data_dir, image_size)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    return train_dataset, train_loader
