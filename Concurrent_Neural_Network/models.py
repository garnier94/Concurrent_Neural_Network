import torch
import torch.nn as nn


def poissonLoss(predicted, observed):
    """Custom loss function for Poisson model."""
    loss = torch.mean(predicted - observed * torch.log(predicted))
    return loss


class Concurrent_Module(nn.Module):
    def __init__(self, submodule, sum_factor=1, loss='L1', learning_rate=1):
        """
        Apply the submodule f to the batch of data and returns :
        $$  f(X)/\|f(X)\|_1 $$
        :param submodule: Underlying nn.Module used to compute the value for each component. Its output is supposed to be a seq_len X 1 tensor
        :param sum_factor: The output of this network is rescaled to sum_factor
        :param loss: 'L1" loss for L1 Loss, 'poisson' for Poisson Loss
        :param learning_rate: Learning rate used for the
        """
        super().__init__()
        self.submodule = submodule
        self.sum_factor = sum_factor
        if loss == 'L1':
            self.loss_function = nn.L1Loss(reduction='sum')
        elif loss == 'poisson':
            self.loss_function = poissonLoss
        self.optimizer = torch.optim.Adam(self.submodule.parameters(), lr=learning_rate)

    def forward(self, x):
        fx = self.submodule.forward(x)
        return self.sum_factor * fx / fx.sum()

    def train(self, data_loader, max_epochs=100, eval_dataset=None, keep_trace=False, batch_print = 10):
        """
        Train the underlying submodule
        :param data_loader: Collection of [features,samples] batch used for training
        :param max_epochs: number of epochs used for training
        :param eval_dataset: if None, nothing happens. Otherwise, it could be a data_loader of similar format used for testing
        :param keep_trace: TODO
        :param batch_print: print a message every batch_print epoch
        :return:
        """
        trace = []
        sum_train = sum([sum(local_labels) for _, local_labels in data_loader])
        for epoch in range(max_epochs):
            # Training
            epoch_loss = 0
            for local_batch, local_labels in data_loader:
                prediction = self.forward(local_batch)
                loss = self.loss_function(prediction[:, 0], local_labels)
                epoch_loss += float(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if epoch % batch_print == 0:
                MAPE = 100 * epoch_loss / sum_train
                print('Epoch %s' % (epoch))
                print('Train MAPE: %.4f' % MAPE)
                if not eval_dataset is None:
                    _ = self.eval(eval_dataset)

    def eval(self, data_loader):
        """
        Evaluation of the module on a testing set
        :param data_loader: Collection of [features,samples] batch used for evaluation
        :return:
        """
        sum_label = 0
        sum_error = 0
        for X_test, y_test in data_loader:
            prediction = self.forward(X_test)
            sum_error += self.loss_function(prediction[:, 0], y_test)
            sum_label += sum(y_test)
        print('Test MAPE: %.4f \n' % (100 * sum_error / sum_label))
        return prediction
