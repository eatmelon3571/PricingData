
import numpy as np

# 我的
import params
from utils import mkdir


def mnist_allocation():
    if params.division == 'iid':
        allocation_iid()
    elif params.division == 'noniid':
        allocation_noniid()
    elif params.division == 'partialnoniid':
        allocation_partialnoniid()


def allocation_iid():
    # 本次实验根目录   params.mnist_division_testno = './data/mnist/iid/test0'

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

        write_client_data(provider_i_dir, images[start:end], labels[start:end])
        offset += params.data_per_provider_num


def allocation_noniid():

    # 本次实验根目录   params.mnist_iid_testno = './data/mnist/iid/test0'

    # 读取解码后的数据
    images = np.load(params.dataset_npy_data + '/train_images.npy')
    labels = np.load(params.dataset_npy_data + '/train_labels.npy')

    # 创建本次实验总目录
    mkdir(params.dataset_division_testno)

    # 创建客户端文件夹
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)
        mkdir(provider_i_dir)
    # 数据排序
    # 存放排好序的数据
    data_images = []
    data_labels = []
    for i in range(10):
        data_images.append([])
        data_labels.append([])

    for i in range(params.mnist_data_total):
        label = int(labels[i])
        data_images[label].append(images[i])
        data_labels[label].append(label)

    images_ = []
    labels_ = []
    for i in range(len(data_labels)):
        images_.append(np.array(data_images[i]))
        labels_.append(np.array(data_labels[i]))


    # 给客户端分配数据      截取数组：y = x[0:3]  从0开始截取
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)
        write_client_data(provider_i_dir, images_[i], labels_[i])


def allocation_partialnoniid():

    # 本次实验根目录   params.mnist_iid_testno = './data/mnist/iid/test0'

    # 读取解码后的数据
    images = np.load(params.dataset_npy_data + '/train_images.npy')
    labels = np.load(params.dataset_npy_data + '/train_labels.npy')


    images_iid = images.copy()
    labels_iid = labels.copy()
    # 打乱原始数据  并保证打乱后图像和标签数据顺序一致
    state = np.random.get_state()
    np.random.shuffle(images_iid)
    np.random.set_state(state)
    np.random.shuffle(labels_iid)


    # 创建本次实验总目录
    mkdir(params.dataset_division_testno)

    # 创建客户端文件夹
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)
        mkdir(provider_i_dir)
    # 数据排序
    # 存放排好序的数据
    data_images = []
    data_labels = []
    for i in range(10):
        data_images.append([])
        data_labels.append([])

    for i in range(params.mnist_data_total):
        label = int(labels[i])
        data_images[label].append(images[i])
        data_labels[label].append(label)

    images_ = []
    labels_ = []
    for i in range(len(data_labels)):
        images_.append(np.array(data_images[i]))
        labels_.append(np.array(data_labels[i]))

    noniid_num = 2000
    iid_num = 2000

    # 给客户端分配数据      截取数组：y = x[0:3]  从0开始截取
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)


        write_client_data(provider_i_dir, images_[i][0:], labels_[i])

    # 给客户端分配数据      截取数组：y = x[0:3]  从0开始截取
    offset = 0
    for i in range(params.provider_num):
        provider_i_dir = params.dataset_division_testno + '/provider' + str(i)


        start = offset  # 数据截取开始位置
        end = offset + params.data_per_provider_num  # 数据截取结束位置

        print(images_[i][0:noniid_num].shape)
        print(images[start:end].shape)
        imgs = np.vstack((images_[i][0:noniid_num], images[start:end]))
        labs = np.hstack((labels_[i][0:noniid_num], labels[start:end]))

        # print('##', imgs.shape)
        # print('##', labs.shape)

        write_client_data(provider_i_dir, imgs, labs)
        offset += iid_num


def write_client_data(provider_dir, images, labels):
    """
    将images和labels写入到dir中 文件名分别为images.npy 和 labels.npy
    :param provider_dir: 客户端数据根目录
    :param images: 图片数据
    :param labels: 标签数据
    """
    np.save(provider_dir + '/images.npy', images)
    np.save(provider_dir + '/labels.npy', labels)

