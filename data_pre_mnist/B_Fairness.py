
import numpy as np

# 我的
import params
from utils import mkdir


def allocation_provider_Dataset_Y():
    """
    分为5个客户端，客户端0复制0-4数据
    数据集复制0-4数据
    """
    # 读取解码后的数据
    images = np.load(params.dataset_npy_data + '/train_images.npy')
    labels = np.load(params.dataset_npy_data + '/train_labels.npy')

    # 打乱原始数据  并保证打乱后图像和标签数据顺序一致
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)

    # 创建本次实验总目录
    mkdir(params.dataset_division_testno)

    # 创建客户端文件夹
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)
        mkdir(provider_i_dir)

    # 给客户端分配数据      截取数组：y = x[0:3]  从0开始截取
    offset = 0
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)

        start = offset  # 数据截取开始位置
        end = offset + params.data_per_provider_num  # 数据截取结束位置

        if i == 0:
            img = np.array(images[start:end]).copy()
            lab = np.array(labels[start:end]).copy()
            # 复制0-4数据
            images_duplicated, labels_duplicated = duplicated(img, lab)
            write_client_data(provider_i_dir, images_duplicated, labels_duplicated)
        else:
            write_client_data(provider_i_dir, images[start:end], labels[start:end])

        offset += params.data_per_provider_num


def allocation_test_Dataset_Y():
    """
    测试集数据
    数据集复制0-4数据
    """
    # 读取解码后的数据
    images = np.load(params.dataset_npy_data + '/test_images.npy')
    labels = np.load(params.dataset_npy_data + '/test_labels.npy')

    # 创建测试集目录
    mkdir(params.test_division_dir)

    # 复制0-4数据
    images_duplicated, labels_duplicated = duplicated(images, labels)
    np.save(params.test_division_dir + '/test_images.npy', images_duplicated)
    np.save(params.test_division_dir + '/test_labels.npy', labels_duplicated)


def duplicated(images, labels):
    l = len(labels)
    for i in range(l):
        label = labels[i]
        if label < 5:
            images = np.vstack((images, images[i]))
            labels = np.vstack((labels, labels[i]))
    return images, labels


def write_client_data(provider_dir, images, labels):
    """
    将images和labels写入到dir中 文件名分别为images.npy 和 labels.npy
    :param provider_dir: 客户端数据根目录
    :param images: 图片数据
    :param labels: 标签数据
    """
    np.save(provider_dir + '/images.npy', images)
    np.save(provider_dir + '/labels.npy', labels)
