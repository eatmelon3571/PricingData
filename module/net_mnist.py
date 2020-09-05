import torch.nn as nn
import torch.nn.functional as F


class NetMNIST(nn.Module):

    def __init__(self):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.mp = nn.MaxPool2d(2)
        # 全连接
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
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