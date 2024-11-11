import torch
from torch import nn

from utils import HyperParameters, ProgressBoard
from optim import SGD


class Module(nn.Module, HyperParameters):
    """
    The base class of models.
    """
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    
    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X):
        assert hasattr(self, 'net'), "Neural network is not defined"
        return self.net(X)
    
    def plot(self, key, value, train):
        """
        Plot a point in animation.
        """
        assert hasattr(self, 'trainer'), "Trainer is not inited"
        self.board.xlabel = 'epoch'
        self.board.ylabel = key
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch

        self.board.draw(x, value.to(torch.device('cpu')).detach().numpy(), 
                        ('train_' if train else 'val_') + key, every_n=int(n))
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1]) # self(*batch[:-1]) call Module(), and calculate the prediction
        self.plot('loss', l, train=True)
        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
    

class LinearRegressionScratch(Module):
    """
    The linear regression model implemented from scratch.
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    
    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    