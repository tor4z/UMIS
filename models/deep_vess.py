from torch import nn


class DeepVess(nn.Module):
    def __init__(self):
        super(DeepVess, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), stride=(2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, (1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), stride=(2, 2, 2))
        )

        self.fc1 = nn.Linear(64 * 1 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 2 * 1 * 4 * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], 64 * 1 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], 2, 1, 4, 4)

        return x