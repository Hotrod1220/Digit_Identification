import torch
from model import Model

class Predictor:
    def __init__(self):
        self.model = Model()

        state = torch.load('state/model.pth')
        self.model.load_state_dict(state)
        self.model.eval()

        self.mapping = [i for i in range(0,10)]

    def predict(self, image):
        # TODO(Hotrod1220) pass image into model, get label and return it.
        pass