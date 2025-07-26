import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d
)
from torch_scatter import scatter_mean
import numpy as np

NPARTICLES = 60 #number of particles per event
INPUTS = 19 #how many features [batch size, INPUTS, nparticles]
H1 = 642 
H2 = 32
H3 = 16
C1 = 50
OUTPUTS = 2
class DeepSets(torch.nn.Module):
    def __init__(self):
        super(DeepSets, self).__init__()
        self.phi = Seq(
            Conv1d(INPUTS, H1, 1),
            BatchNorm1d(H1),
            ReLU(),
            Conv1d(H1, H2, 1),
            BatchNorm1d(H2),
            ReLU(),
            Conv1d(H2, H3, 1),
            BatchNorm1d(H3),
            ReLU(),
        )
        self.rho = Seq(
            Lin(H3, C1),
            BatchNorm1d(C1),
            ReLU(),
            Lin(C1, OUTPUTS),
            Sigmoid(),
        )

    def forward(self, x):
        out = self.phi(x)
        out = scatter_mean(out, torch.LongTensor(np.zeros(NPARTICLES)), dim=-1)
        return self.rho(torch.squeeze(out))