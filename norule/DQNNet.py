class DQN_Net(nn.Module):
    def __init__(self, state_size, action_size, conv_units):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=2)
      
        self.conv2 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_units)

        self.conv3 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_units)

        self.conv4 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.fc_size = conv_units * (state_size[-1]+2) * (state_size[-2]+2)
        self.fc = nn.Linear(self.fc_size, action_size)

    def forward(self, x):
        # 순전파
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # flatten
        x = x.view(-1, self.fc_size)
        # 전연결층
        x = self.fc(x)

        return x
