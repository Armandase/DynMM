import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(
        self,
        embed_dim=6,
        num_classes=6,
        dropout_rate=0.1,
        norm_layer=nn.BatchNorm2d,
        width=7,
    ):
        super(ClassifierHead, self).__init__()

        self.flatten = nn.Flatten()
        dense_input_dim = embed_dim * width * width

        self.fc1 = nn.Linear(dense_input_dim, dense_input_dim // width)
        self.norm_layer1 = nn.LayerNorm(dense_input_dim // width)

        self.fc2 = nn.Linear(dense_input_dim // width, dense_input_dim // (width * 2))
        self.norm_layer2 = nn.LayerNorm(dense_input_dim // (width * 2))

        self.fc3 = nn.Linear(
            dense_input_dim // (width * 2), dense_input_dim // (width * 4)
        )
        self.norm_layer3 = nn.LayerNorm(dense_input_dim // (width * 4))

        self.fcOutput = nn.Linear(dense_input_dim // (width * 4), num_classes)
        self.drop = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.norm_layer1(x)

        x = F.relu(self.fc2(x))
        x = self.norm_layer2(x)
        x = self.drop(x)

        x = F.relu(self.fc3(x))
        x = self.norm_layer3(x)
        x = self.drop(x)

        x = F.relu(self.fcOutput(x))
        x = self.softmax(x)
        return x
