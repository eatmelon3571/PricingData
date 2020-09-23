"""Params for PricingData."""

# 数据根目录
root_dir = './data'

'''
数据集dataset选择
'mnist'
'cifar10'
'''
dataset = 'cifar10'



dataset_dir = root_dir + '/' + dataset
mnist_data_total = 60000               # MNIST数据
cifar10_data_total = 50000             # CIFAR10数据

dataset_origin_data = dataset_dir + '/origin_data'   # 原始数据
dataset_npy_data = dataset_dir + '/npy_data'         # 存放mnist解码后的数据

provider_num = 6
data_per_provider_num = 2000

# 实验数据和存储
'''
数据分布division选择
'iid'
'noniid'
'partialnoniid'
'''
division = 'partialnoniid'

dataset_division = dataset_dir + '/' + division
test_no = 0                                  # 实验编号
dataset_division_testno = dataset_division + '/test' + str(test_no)    # 本次实验目录
dataset_division_testno_save = dataset_division_testno + '/save'       # 存储目录




# DataBuyer参数
miu = 0                   # 阈值μ
B = 100                   # 准备付的金额

# 网络参数


# 训练参数
learning_rate_mnist = 0.01
momentum = 0.9

learning_rate_cifar10 = 0.0001      # 0.001收敛快一点
beta1 = 0.9
beta2 = 0.999


epochnum = 5              # noniid
batch_size = 50           # mnist noniid 设为10


# 训练过程参数
round_start = 0            # 从第round_start轮开始
round_end = 5              # 从第round_end轮结束

round_cur = round_start    # 训练当前轮数，从start开始到end结束   感觉不应该放这里

# K = 4
K = provider_num                      # K个provider做聚合   只做一次聚合

local_time = 20            # 聚合前本地训练次数

fed_train_time = 30        # 联邦学习训练轮数    mnist iid 10次基本不变了
v_S_fed_train_time = 30     # 聚合时联邦学习训练轮数

excel_dir = dataset_division_testno + '/1.xls'

txt_dir = dataset_division_testno + '/11.txt'


npy_original_v_all = dataset_division_testno + '/Original_v_all.npy'
npy_ScoreAverage_v_all = dataset_division_testno + '/ScoreAverage_v_all.npy'

# 记录pa pb pab的值
txt_dir_papb = dataset_division_testno + '/papb.txt'
# 记录   pab聚合-pab估计
txt_dir_pab_reduce = dataset_division_testno + '/pab_reduce.txt'


def change_param(dataset_temp='mnist',
                 division_temp='iid'):
    """在改变对路径有影响的参数时，把相应路径也修改了"""
    dataset = dataset_temp
    dataset_dir = root_dir + '/' + dataset

    dataset_origin_data = dataset_dir + '/origin_data'  # 原始数据
    dataset_npy_data = dataset_dir + '/npy_data'  # 存放mnist解码后的数据

    division = division_temp

    dataset_division = dataset_dir + '/' + division
    dataset_division_testno = dataset_division + '/test' + str(test_no)  # 本次实验目录
    dataset_division_testno_save = dataset_division_testno + '/save'  # 存储目录

    excel_dir = dataset_division_testno + '/1.xls'

    txt_dir = dataset_division_testno + '/11.txt'


    npy_original_v_all = dataset_division_testno + '/Original_v_all.npy'
    npy_ScoreAverage_v_all = dataset_division_testno + '/ScoreAverage_v_all.npy'
