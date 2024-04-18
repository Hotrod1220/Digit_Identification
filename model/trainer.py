import torch
from tqdm import tqdm

class Trainer:
    """
    Trains a neural network.
    """
    def __init__(
        self,
        epochs = 0,
        device = 'cpu',
        model = None,
        loss = None,
        optimizer = None,
        training = None,
        testing = None,
    ):
        self.epochs = epochs
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.training = training
        self.testing = testing

    def _training_epoch(self):
        """
        Performs a single training epoch on the training dataset

        Returns:
            tuple, (accuracy, loss) of the training epoch
        """
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0

        for (images, labels) in tqdm(self.training, total=len(self.training)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.loss(output, labels)
            
            total_loss += loss.item()

            _, prediction = torch.max(output, dim=1)
            
            total_accuracy += (prediction == labels).sum().item()

            loss.backward()

            self.optimizer.step()

        return (
            total_loss / len(self.training),
            total_accuracy / len(self.training.dataset)
        )

    def _testing_epoch(self):
        """
        Performs a single training epoch on the testing dataset

        Returns:
            tuple, (accuracy, loss) of the testing epoch
        """
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0

        with torch.no_grad():
            for images, labels in tqdm(self.testing, total=len(self.testing)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                loss = self.loss(output, labels)

                total_loss += loss.item()

                _, prediction = torch.max(output, dim=1)

                total_accuracy += (prediction == labels).sum().item()

        return (
            total_loss / len(self.testing),
            total_accuracy / len(self.testing.dataset)
        )
    
    def train(self):
        """
        Training process of the model, records accuracy and 
        loss of training and testing.

        Returns:
            Dict, training and testing accuracy and loss throughout
            the training process.
        """
        self.model.to(self.device)

        history = {
            'training': {
                'classification_accuracy': [],
                'classification_loss': []
            },
            'test': {
                'classification_accuracy': [],
                'classification_loss': []
            }
        }

        for i in range(self.epochs):
            print(f"[Epoch {i + 1}]")

            loss, accuracy = self._training_epoch()

            accuracy = round(accuracy, 10)
            loss = round(loss, 10)

            history['training']['classification_accuracy'].append(accuracy)
            history['training']['classification_loss'].append(loss)

            print(f"training_accuracy: {accuracy:.4f}")
            print(f"training_loss: {loss:.4f}")

            loss, accuracy = self._testing_epoch()

            accuracy = round(accuracy, 10)
            loss = round(loss, 10)

            history['test']['classification_accuracy'].append(accuracy)
            history['test']['classification_loss'].append(loss)

            print(f"test_accuracy: {accuracy:.4f}")
            print(f"test_loss: {loss:.4f}")

            print()

        print('Training is complete')

        return history