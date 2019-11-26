from torch import nn


class VessNN(nn.Module):
    def __init__(self):
        super(VessNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 24, (2, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(24, 24, (2, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(24, 24, (2, 3, 3)),
            nn.Tanh(),
            nn.MaxPool3d((1, 2, 2), stride=(1, 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(24, 36, (1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(36, 36, (1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2), stride=(1, 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(36, 48, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(48, 48, (2, 3, 3), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.MaxPool3d((2, 2, 2), stride=(1, 1, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(48, 60, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(60, 60, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(60, 100, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU()
        )
        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(100 * 5 * 62 * 62, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = x.reshape(x.shape[0], 100 * 5 * 62 * 62)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 2, 1, 1, 1)

        return x
