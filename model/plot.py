import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path

if __name__ == '__main__':
    """
    Plots the accuracy and loss of the CNN created.
    """
    path = Path.cwd()
    path = path.joinpath('state/history.pkl')

    with open(path, 'rb') as handle:
        history = pickle.load(handle)

    training = history.get('training')
    validation = history.get('validation')

    # Accuracy
    training_accuracy = training.get('classification_accuracy')
    validation_accuracy = validation.get('classification_accuracy')

    training_accuracy = [x * 100 for x in training_accuracy]
    validation_accuracy = [x * 100 for x in validation_accuracy]

    plt.figure(figsize=(10, 5))

    plt.plot(
        training_accuracy,
        label='Training Accuracy'
    )

    plt.plot(
        validation_accuracy,
        label='Validation Accuracy'
    )

    plt.title('Classification: Accuracy')
    plt.xticks(
        np.arange(len(training_accuracy)), 
        np.arange(1, len(validation_accuracy) + 1)
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig(
        'plot/accuracy.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()

    # Loss
    training_loss = training.get('classification_loss')
    validation_loss = validation.get('classification_loss')

    plt.figure(figsize=(10, 5))

    plt.plot(
        training_loss,
        label='Training Loss'
    )

    plt.plot(
        validation_loss,
        label='Validation Loss'
    )

    plt.title('Classification: Loss')
    plt.xticks(
        np.arange(len(training_loss)), 
        np.arange(1, len(validation_loss) + 1)
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(
        'plot/loss.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()
