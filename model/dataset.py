from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class MNIST:
    """
    MNIST dataset download and data loader.
    """
    def __init__(self, batch_size = 128):
        """
        Download and load the MNIST dataset. 

        Args:
            batch_size: the size of the batch used for the datasets
        """
        train_data = datasets.MNIST(
            root='data',
            train=True,
            transform=ToTensor(),
            download=True
        )

        test_data = datasets.MNIST(
            root='data',
            train=False,
            transform=ToTensor(),
            download=True
        )

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )

        self.test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )
