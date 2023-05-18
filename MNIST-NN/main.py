import torch 
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import CNN
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
learning_rate = 0.01
num_epochs = 10
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

def download_mnist():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    return train_data, test_data

def data_loader():
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return train_loader, test_loader

def train_model():
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Step: {i + 1}/{total_steps}, Loss: {loss.item():.4f}")
            
def validate_data():
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        num_class_correct = [0 for i in range(10)]
        num_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                prediction = predicted[i]
                if (label == prediction):
                    num_class_correct[label] += 1
                num_class_samples[label] += 1
        
        accuracy = 100.0 * num_correct / num_samples
        print(f"Accuracy of the network: {accuracy} %")

        for i in range(100):
            accuracy = 100.0 * num_class_correct[i] / num_class_samples[i]
            print(f"Accuracy of {classes[i]}: {accuracy} %")

if __name__ == '__main__':
    train_data, test_data = download_mnist()
    train_loader, test_loader = data_loader()

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #train_model()

    file = "model.pth"
    torch.save(model.state_dict(), file)

    #validate_data()
