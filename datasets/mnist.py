"""Dataset setting and data loader for MNIST."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

import params


def get_provider_mnist(provider_no):
    """Get MNIST dataset loader."""

    provider_i_dir = params.dataset_division_testno + '/provider' + str(provider_no)
    images_dir = provider_i_dir + '/images.npy'
    labels_dir = provider_i_dir + '/labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    # dataset and data loader
    mnist_dataset = MnistDataset(images_dir=images_dir,
                                 labels_dir=labels_dir,
                                 image_transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader


def get_test_mnist():
    """Get MNIST test dataset loader."""
    images_dir = params.dataset_npy_data + '/test_images.npy'
    labels_dir = params.dataset_npy_data + '/test_labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    # dataset and data loader
    mnist_test_dataset = MnistDataset(images_dir=images_dir,
                                      labels_dir=labels_dir,
                                      image_transform=pre_process)

    mnist_test_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_test_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_test_data_loader


def get_test_dataset_mnist():
    images_dir = params.dataset_npy_data + '/test_images.npy'
    labels_dir = params.dataset_npy_data + '/test_labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

    # dataset and data loader
    mnist_test_dataset = MnistDataset(images_dir=images_dir,
                                      labels_dir=labels_dir,
                                      image_transform=pre_process)
    return mnist_test_dataset


class MnistDataset(Dataset):
    def __init__(self,
                 images_dir,
                 labels_dir,
                 image_transform):
        self.images = np.load(images_dir, encoding='bytes', allow_pickle=True)
        self.labels = np.load(labels_dir, encoding='bytes', allow_pickle=True)
        self.images = torch.from_numpy(self.images).clone().detach()
        self.labels = torch.from_numpy(self.labels).clone().detach()
        self.image_transform = image_transform
        # 归一化
        self.images = torch.div(self.images, 255)
        # 标准化
        self.images = self.image_transform(self.images)
        # 转换成适合net输入的格式   image:float32  label:long
        self.images = self.images.reshape(-1, 1, 28, 28)
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = self.labels.long()

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)
