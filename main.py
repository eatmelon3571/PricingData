"""Main script for PricingData."""

import params
from data_pre_mnist.mnist_decode import decode_mnist_data_to_file
from data_pre_mnist.provider_data import mnist_allocation
from data_pre_cifar10.provider_data import cifar10_allocation
from data_pre_cifar10.cifar10_decode import decode_cifar10_data_to_file

from utils import create_dir

from core.collaborative_modelling import CollaborativeModelling, Original
from core.collaborative_modelling import ScoreAverage


if __name__ == '__main__':
    # 创建所需文件夹
    create_dir()

    # cifar10数据文件（data_batch_1等）放在./data/cifar10/origin_data下
    # '''
    # params.test_no = 0     # 实验编号0
    if params.dataset == 'mnist':
        # 解码数据集
        # decode_mnist_data_to_file()
        # 分配数据集
        mnist_allocation()
    elif params.dataset == 'cifar10':
        # decode_cifar10_data_to_file()
        # cifar10_allocation()
        a = 0
    # '''

    # cifar10 noniid 本地迭代次数要小一点

    # 先把一个网络初值存下来，然后每次都加载这个
    # creat_model()

    # 或者运行前将模型存下来

    # 原本的聚合方法：直接所有节点算SV  不用聚合树
    tree_list = Original()


    # 协作建模
    ScoreAverage()






