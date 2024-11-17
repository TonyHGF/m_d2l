import torch
import os
import matplotlib.pyplot as plt
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
    
    def plot(self, key, value, train, draw_online=True, img_path=None):
        """
        Plot a point in animation.
        """
        assert hasattr(self, 'trainer'), "Trainer is not inited"
        self.board.xlabel = 'epoch'
        self.board.ylabel = key
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
            phase = 'train'
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
            phase = 'test'

        x_value = x
        y_value = value.to(torch.device('cpu')).detach().numpy()
        label = f"{phase}_{key}"

        if draw_online:
            self.board.draw(x_value, y_value, label, every_n=int(n))
        else:
            assert img_path is not None, "img_path must be specified if draw_online is False"

            epoch = self.trainer.epoch + 1
            file_name = f"{phase}_epoch{epoch}_{key}.png"
            full_path = os.path.join(img_path, file_name)

            os.makedirs(img_path, exist_ok=True)

            self.board.draw(x_value, y_value, label, every_n=int(n), img_path=full_path)
    

    def training_step(self, batch, draw_online=True, img_path=None):
        l = self.loss(self(*batch[:-1]), batch[-1]) # self(*batch[:-1]) call Module(), and calculate the prediction
        self.plot('loss', l, train=True, draw_online=draw_online, img_path=img_path)
        return l
    
    def validation_step(self, batch, draw_online=True, img_path=None):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False, draw_online=draw_online, img_path=img_path)

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
    

class LinearRegression(Module):
    """
    The linear regression model implemented with high-level APIs.
    """
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1) 
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)
