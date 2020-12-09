from torch import nn

"""
ANNmodel.py  用于定义网络的结构
"""


class ANNmodel(nn.Module):
    def __init__(self):
        super(ANNmodel, self).__init__()
        self.layer1 = nn.Linear(4, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layerOutput = nn.Linear(5, 5)

        self.model = nn.Sequential(
            nn.BatchNorm1d(4, eps=1e-5),
            self.layer1,
            nn.LeakyReLU(),

            nn.BatchNorm1d(5, eps=1e-5),
            self.layer2,
            nn.LeakyReLU(),

            nn.BatchNorm1d(5, eps=1e-5),
            self.layerOutput,
        )

    def forward(self, x):
        return self.model(x)

