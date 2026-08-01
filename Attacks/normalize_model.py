import torch
import torch.nn as nn


class NormalizeModel(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.4914, 0.4822, 0.4465]
            ).view(1,3,1,1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.2023, 0.1994, 0.2010]
            ).view(1,3,1,1)
        )


    def forward(self, x):

        x = (x - self.mean) / self.std

        return self.model(x)