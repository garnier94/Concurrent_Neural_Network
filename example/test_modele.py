import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Concurrent_Neural_Network.models import Concurrent_Module


class Feed_forward_model(nn.Module):
    "2 layers neural network used for for testing"

    def __init__(self, n_input, n_hidden):
        super().__init__()
        self.linear_1 = nn.Linear(n_input, n_hidden, bias=False)
        self.linear_2 = nn.Linear(n_hidden, 1, bias=False)
        self.n_input = n_input
        self.n_hidden = n_hidden

    def forward(self, x):
        return F.softplus(self.linear_2(F.relu(self.linear_1(x))), )


X = [[[10, 1], [1, 2], [9, 1], [1, 2]], [[8, 1], [4, 2], [1, 1], [3, 1]], [[0, 1], [1, 2], [2, 1], [1, 4]]]
y = [[4, 3, 2, 1], [3, 1, 4, 2], [0, 3, 4, 3], [4, 0, 3, 3]]
data_loader = [[torch.tensor(np.array(X[i]).astype(np.float32)), torch.tensor(np.array(y[i]).astype(np.float32))] for i
               in range(4)]

n_input = 2
n_hidden = 2
submodel = Feed_forward_model(n_input, n_hidden)
model = Concurrent_Module(submodel,sum_factor=1, loss='L1')
model.eval(data_loader)
model.train(data_loader, max_epochs=100, batch_print = 1)
print('Ok')