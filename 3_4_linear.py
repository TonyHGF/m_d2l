import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
batch_size = 10
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
num_epochs = 10


def main():
    dataloader = load_array(features, batch_size, is_train=True)
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # loss = nn.MSELoss()
    loss = nn.SmoothL1Loss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # print(features.shape)

    for epoch in range(num_epochs):
        for i, X in enumerate(dataloader):
            X = torch.stack(X)
            res = net(X)
            # print(res.shape, labels.shape)
            l = loss(res, labels)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'Epoch {epoch}, Loss: {l}')


if __name__ == '__main__':
    main()
