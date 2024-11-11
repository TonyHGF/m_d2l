import torch

from dataloader import SyntheticRegressionData
from model import LinearRegressionScratch
from train import Trainer


def main():
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    print('-'*20)
    print('features:', data.X[0],'\nlabel:', data.y[0])

    X, y = next(iter(data.train_dataloader()))
    print('-'*20)
    print('X shape:', X.shape, '\ny shape:', y.shape)

    print('-'*20)
    print(len(data.train_dataloader()))

    model = LinearRegressionScratch(2, lr=0.03)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    


if __name__ == '__main__':
    main()