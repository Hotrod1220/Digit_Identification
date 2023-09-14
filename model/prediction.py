import torch
from pathlib import Path

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.model import Model  # noqa: E402

# For inference.py
# from model import Model

class Predictor:
    """
    Used to perform digit predictions from a trained model.
    """
    def __init__(self):
        """
        Loads the trained model weights.
        """
        path = Path.cwd()
        path = path.with_name('model')
        path = path.joinpath('state/model.pth')
        state = torch.load(path)

        self.model = Model()
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image):
        """
        Download and load the MNIST dataset. 

        Args:
            image: tensor, digit image to predict

        Returns:
            int, predicted digit.
        """
        image = image.unsqueeze(0)
        output = self.model(image)
        
        _, prediction = torch.max(output, dim=1)

        return prediction.item()
