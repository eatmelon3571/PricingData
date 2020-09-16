import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

import params


def get_provider_cifar10(provider_no):
    """Get cifar10 dataset loader."""

    provider_i_dir = params.dataset_division_testno + '/provider' + str(provider_no)
    images_dir = provider_i_dir + '/images.npy'
    labels_dir = provider_i_dir + '/labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize(mean=(0.491372549, 0.482352941,
                                                                 0.446666667),
                                                           std=(0.247058824, 0.243529412,
                                                                0.261568627))])

    # dataset and data loader
    dataset = CIFAR10Dataset(images_dir=images_dir,
                             labels_dir=labels_dir,
                             image_transform=pre_process)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return data_loader


def get_test_cifar10():
    """Get cifar10 test dataset loader."""
    images_dir = params.dataset_npy_data + '/test_images.npy'
    labels_dir = params.dataset_npy_data + '/test_labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize(mean=(0.491372549, 0.482352941,
                                                                 0.446666667),
                                                           std=(0.247058824, 0.243529412,
                                                                0.261568627))])

    # dataset and data loader
    test_dataset = CIFAR10Dataset(images_dir=images_dir,
                                  labels_dir=labels_dir,
                                  image_transform=pre_process)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return test_data_loader


def get_test_dataset_cifar10():
    images_dir = params.dataset_npy_data + '/test_images.npy'
    labels_dir = params.dataset_npy_data + '/test_labels.npy'

    # image pre-processing
    pre_process = transforms.Compose([transforms.Normalize(mean=(0.491372549, 0.482352941,
                                                                 0.446666667),
                                                           std=(0.247058824, 0.243529412,
                                                                0.261568627))])

    # dataset and data loader
    test_dataset = CIFAR10Dataset(images_dir=images_dir,
                                  labels_dir=labels_dir,
                                  image_transform=pre_process)
    return test_dataset


class CIFAR10Dataset(Dataset):
    def __init__(self,
                 images_dir,
                 labels_dir,
                 image_transform):
        self.images = np.load(images_dir, encoding='bytes', allow_pickle=True)
        self.labels = np.load(labels_dir, encoding='bytes', allow_pickle=True)
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.from_numpy(self.labels)
        self.image_transform = image_transform
        #
        self.images = self.images.reshape(-1, 3, 32, 32)
        # 归一化
        self.images = torch.div(self.images, 255)
        # 标准化
        for i in range(len(self.images)):
            self.images[i] = self.image_transform(self.images[i])
        # 转换成适合net输入的格式   image:float32  label:long
        self.labels = self.labels.long()

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)
