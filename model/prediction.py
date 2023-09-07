import torch
from model import Model

class Predictor:
    """
    Used to perform digit predictions from a trained model.
    """
    def __init__(self):
        """
        Loads the trained model weights.
        """
        state = torch.load('state/model.pth')

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
