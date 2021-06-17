<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from Concurrent_Neural_Network.models import  poissonLoss

class Multi_layer_feed_forward_model(nn.Module):
    "layers neural network used for for testing"

    def __init__(self, n_input, n_hidden, loss= 'L1', learning_rate=1):
        """

        n_input :
        n_hidden :
        """
        super().__init__()
        self.list_layer = []
        n_h = [n_input] + n_hidden + [1]
        for i in range(len(n_hidden)+1) :
            self.list_layer.append(nn.Linear(n_h[i], n_h[i+1], bias=False))
        self.n_input = n_input
        self.n_hidden = n_hidden
        if loss == 'L1':
            self.loss_function = nn.L1Loss(reduction='sum')
            self.loss_eval = nn.L1Loss(reduction='sum')
        elif loss == 'poisson':
            self.loss_function = poissonLoss
            self.loss_eval = nn.L1Loss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def shape(self):
        [len(list(layer.parameters())[0]) for layer in self.list_layer]

    def parameters(self):
        return [list(layer.parameters())[0] for layer in self.list_layer]

    def forward(self, x):
        for j,lay in enumerate(self.list_layer):
            if j == len(self.n_hidden)-1:
                act = F.softplus
            else:
                act = F.relu
            x = act(lay(x))
        return x

    def train(self, data_loader, max_epochs=100, eval_dataset=None, batch_print=10,early_stopping=None):
        trace = []

        if early_stopping is not None:
            cur_stopping = 0
            min_epoch_loss = float('inf')
        sum_train = sum([sum(local_labels) for _, local_labels in data_loader])
        for epoch in range(max_epochs):
            # Training
            epoch_loss = 0
            eval_loss = 0
            for local_batch, local_labels in data_loader:
                prediction = self.forward(local_batch)
                loss = self.loss_function(prediction[:, 0], local_labels)
                eval_loss += self.loss_eval(prediction[:, 0], local_labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if epoch % batch_print == 0:
                MAPE = 100 * eval_loss / sum_train
                print('Epoch %s' % (epoch))
                print('Train MAPE: %.4f' % MAPE)
                if not eval_dataset is None:
                    _ = self.eval(eval_dataset)
            if early_stopping is not None:
                if min_epoch_loss > eval_loss:
                    cur_stopping += 1
                    if cur_stopping > early_stopping:
                        break
                else:
                    cur_stopping = 0
                    min_epoch_loss = eval_loss

    def eval(self, data_loader, return_MAPE=False):
        """
        Evaluation of the module on a testing set (for L1 Loss)
        :param data_loader: Collection of [features,samples] batch used for evaluation
        :return:
        """
        sum_label = 0
        sum_error = 0
        for X_test, y_test in data_loader:
            prediction = self.forward(X_test)
            sum_error += self.loss_eval(prediction[:, 0], y_test)
            sum_label += sum(y_test)
        if return_MAPE:
            return (100 * sum_error / sum_label)
        else:
            print('Test MAPE: %.4f \n' % (100 * sum_error / sum_label))
            return prediction
=======
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
        n_h = [n_input] + n_hidden + [1]
        for i in range(len(n_hidden)+1) :
            self.list_layer.append(nn.Linear(n_h[i], n_h[i+1], bias=False))
        self.n_input = n_input
        self.n_hidden = n_hidden

    def parameters(self):
        return [list(layer.parameters())[0] for layer in self.list_layer]

    def forward(self, x):
        for j,lay in enumerate(self.list_layer):
            if j == len(self.n_hidden)-1:
                act = F.softplus
            else:
                act = F.relu
            x = act(lay(x))
        return x
>>>>>>> 2bb6eb0e5aa0fb46bc1567d060e3aa9e96da9ac4
