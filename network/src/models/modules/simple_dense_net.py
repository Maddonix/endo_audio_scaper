from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        in_size = int(hparams["sample_rate"]*hparams["duration"] / 1000)
        
        self.model = nn.Sequential(
            nn.Linear(in_size, hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin3_size"], hparams["output_size"]),
        )

    def forward(self, x):
        # batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        # x = x.view(batch_size, -1)

        return self.model(x)
