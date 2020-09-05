import numpy as np

import params


def decode_cifar10_data_to_file():
    # 对原始数据的处理
    data_dir = params.dataset_origin_data
    train_images = []
    train_labels = []
    # 读取数据
    for i in range(1, 6):
        tdir = data_dir + '/data_batch_' + str(i)
        tdata = unpickle(tdir)
        train_images.append(tdata[b'data'])
        train_labels.append(tdata[b'labels'])
    tdir = data_dir + '/test_batch'
    tdata = unpickle(tdir)
    test_images = tdata[b'data']
    test_labels = tdata[b'labels']
    # 数据规整形状
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_images = train_images.reshape(-1, 3 * 32 * 32)
    train_labels = train_labels.reshape(-1)
    # 测试数据另外放一份完整的
    np.save(params.dataset_npy_data + '/train_images.npy', train_images)
    np.save(params.dataset_npy_data + '/train_labels.npy', train_labels)
    np.save(params.dataset_npy_data + '/test_images.npy', test_images)
    np.save(params.dataset_npy_data + '/test_labels.npy', test_labels)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
