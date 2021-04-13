import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_layer_feed_forward_model(nn.Module):
    "layers neural network used for for testing"

    def __init__(self, n_input, n_hidden):
        """

        n_input :
        n_hidden :
        """
        super().__init__()
        self.list_layer = []
        n_h = n_hidden + [1]
        for i in range(len(n_hidden)) :
            self.list_layer.append(nn.Linear(n_h[i], n_h[i+1], bias=False))
        self.n_input = n_input
        self.n_hidden = n_hidden

    def forward(self, x):
        for j,lay in enumerate(self.list_layer):
            if j == len(self.n_hidden)-1:
                act = F.softplus
            else:
                act = F.relu
            x = act(lay(x))
        return x