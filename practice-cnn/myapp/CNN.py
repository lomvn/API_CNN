import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=1152, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=n_classes)
            # предскажем кол-во классов out_features=n_classes net = CNN(47)
        )

    def forward(self, x):
        return self.model(x)

