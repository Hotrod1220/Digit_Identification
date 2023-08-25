import torch 
import pickle

from model import Model
from dataset import MNIST
from trainer import Trainer

if __name__ == '__main__':
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    batch_size = 128
    learning_rate = 0.01

    dataset = MNIST(batch_size=batch_size)

    model = Model()

    loss = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    trainer = Trainer()
    trainer.epochs = 15
    trainer.device = device
    trainer.model = model
    trainer.loss = loss
    trainer.optimizer = optimizer
    trainer.training = dataset.train_loader
    trainer.validating = dataset.test_loader
    history = trainer.train()

    torch.save(
        model.state_dict(),
        'state/model.pth'
    )

    with open('state/trainer.pkl', 'wb') as handle:
        pickle.dump(trainer, handle)

    with open('state/history.pkl', 'wb') as handle:
        pickle.dump(history, handle)