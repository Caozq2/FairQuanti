import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_shape, s_shape=2):  # s_shape默认为2，因为s是one-hot编码的
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape + s_shape, 32)  # 输入维度增加s的维度
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = F.dropout(hidden, 0.1, training=self.training)
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        y = self.fc4(hidden)
        return y

class Net_CENSUS(nn.Module):

    def __init__(self, input_shape, s_shape=2):
        super(Net_CENSUS, self).__init__()
        self.fc1 = nn.Linear(input_shape + s_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, 0.1, training=self.training)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)
        y = self.fc4(hidden)

        return y

class NetPlus_(nn.Module):

    def __init__(self, input_shape, s_shape=2):
        super(NetPlus_, self).__init__()
        self.fc1 = nn.Linear(input_shape + s_shape, 256)  # 增加宽度
        self.bn1 = nn.BatchNorm1d(256)  # 批量归一化
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 1)

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.bn1(hidden)  # 应用批归一化
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.bn2(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc4(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc5(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc6(hidden)
        hidden = F.relu(hidden)

        y = self.fc7(hidden)
        y = self.dropout(y)

        return y