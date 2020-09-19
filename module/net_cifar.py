import torch.nn as nn
import torch.nn.functional as F


class NetCIFAR10(nn.Module):

    def __init__(self):
        super(NetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.mp = nn.MaxPool2d(2)
        # 全连接
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        # softmax的输出层是一个全连接层，所以我们使用一个线性模块就可以
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 1)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.fc1.weight, 1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.weight, 1)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.weight, 1)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.mp(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.mp(x)
        x = F.relu(x)

        x = x.view(in_size, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

    def get_w(self):
        """
        根据网络结构返回各层权重
        :return: 类型：字典(Dictionary)  网络权重的深拷贝
        """
        w = {}
        for key, values in self.state_dict().items():
            w[key] = values.clone().detach()
        return w

    def update_w(self, w):
        """
        :param w: 类型：字典(Dictionary)
        """
        self.load_state_dict(w)


class NetCIFAR10_LeNet(nn.Module):

    def __init__(self):
        super(NetCIFAR10_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.mp = nn.MaxPool2d(2)
        # 全连接
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(self.conv1(x))
        x = self.mp(self.conv2(x))

        x = x.view(in_size, -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def get_w(self):
        """
        根据网络结构返回各层权重
        :return: 类型：字典(Dictionary)  网络权重的深拷贝
        """
        w = {}
        for key, values in self.state_dict().items():
            w[key] = values.clone().detach()
        return w

    def update_w(self, w):
        """
        :param w: 类型：字典(Dictionary)
        """
        self.load_state_dict(w)




