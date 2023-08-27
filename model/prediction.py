import torch
from model import Model

class Predictor:
    def __init__(self):
        state = torch.load('state/model.pth')

        self.model = Model()
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image):
        image = image.unsqueeze(0)
        output = self.model(image)
        
        _, prediction = torch.max(output, dim=1)

        return prediction.item()
