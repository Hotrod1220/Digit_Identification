import torch
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        epochs = 0,
        device = 'cpu',
        model = None,
        loss = None,
        optimizer = None,
        training = None,
        validating = None,
    ):
        self.epochs = epochs
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.training = training
        self.validating = validating

    def _training_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0

        for (images, labels) in tqdm(self.training, total=len(self.training)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)
            loss = self.loss(output, labels)
            
            total_loss += loss.item()

            _, prediction = torch.max(output, dim=1)
            
            total_accuracy += (prediction == labels).sum().item()

            loss.backward()

            self.optimizer.zero_grad()
            self.optimizer.step()

        return (
            total_loss / len(self.training),
            total_accuracy / len(self.training.dataset)
        )

    def _validating_epoch(self):
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0

        for images, labels in tqdm(self.validating, total=len(self.validating)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)
            loss = self.loss(output, labels)
            
            total_loss += loss.item()

            _, prediction = torch.max(output, dim=1)
            
            total_accuracy += (prediction == labels).sum().item()

        return (
            total_loss / len(self.validating),
            total_accuracy / len(self.validating.dataset)
        )
    
    def train(self):
        self.model.to(self.device)

        history = {
            'training': {
                'classification_accuracy': [],
                'classification_loss': []
            },
            'validation': {
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

            loss, accuracy = self._validating_epoch()

            accuracy = round(accuracy, 10)
            loss = round(loss, 10)

            history['validation']['classification_accuracy'].append(accuracy)
            history['validation']['classification_loss'].append(loss)

            print(f"validation_accuracy: {accuracy:.4f}")
            print(f"validation_loss: {loss:.4f}")

            print()

        print('Training is complete')

        return history