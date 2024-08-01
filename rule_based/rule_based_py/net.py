"""# Net"""

class DQN_Net(nn.Module):
    def __init__(self, state, action_size, conv_units=64):
        super().__init__()
        self.state_size = len(state) * len(state)
        # 합성곱 층 정의
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_units)
        self.bn2 = nn.BatchNorm2d(conv_units)

        self.conv2 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.conv3 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.conv4 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.fc1 = nn.Linear(5184, action_size)
    def forward(self, x):
        # 순전파
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # flatten
        x = x.view(-1, 5184)  # 배치 크기에 맞게 데이터를 평탄화

        # 완전 연결층
        x = self.fc1(x)

        return x