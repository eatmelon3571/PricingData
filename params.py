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

K = 3                      # K个provider做聚合

fed_train_time = 0        # 联邦学习训练轮数    mnist iid 10次基本不变了
v_S_fed_train_time = 30     # 聚合时联邦学习训练轮数

excel_dir = dataset_division_testno + '/1.xls'

txt_dir = dataset_division_testno + '/4.txt'





# 实验B_Fairness ----------------------------

'''
'datasetX'
'datasetY'
'datasetZ'
'''
test_division = 'datasetY'

test_division_dir = dataset_npy_data + '/' + test_division

