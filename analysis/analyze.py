import ast
import os
import sys
import torch
import numpy as np

from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from PIL import Image

from visualize import Visualize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.prediction import Predictor  # noqa: E402


class Analyze(ABC):
    """
    Abtract class that preprocesses the data for analysis children classes.
    """
    def __init__(self, vary_size):
        """
        Initizes the data and visualization. Selects folder based on task.

        Args:
            vary_size: boolean value indicates if the images should vary in size.
                False - Task A
                True - Task B
        """
        path = Path.cwd()
        self.dataset_path = path.joinpath('dataset')
        
        if vary_size:
            folder = 'nxn'
        else:
            folder = '28x28'        
        
        data_path = str(self.dataset_path.joinpath(f"{folder}/*.png"))

        paths = glob(data_path)
        self.data = self.init_data(paths)

        self.visualize = Visualize()
        self.predictor = Predictor()

    @abstractmethod
    def anaylze(self):
        pass

    def init_data(self, paths):
        """
        Initialized data with images and labels from the task folder.

        Args:
            paths: list containing the paths for the images, labels are from filename.

        Returns:
            List of tuples, (image as a 2D nd.array, labels as a string).
        """
        data = []
        start = 'labels_'
        end = '.png'

        for path in paths:
            start_index = path.find(start) + len(start)
            end_index = path.find(end)
            
            labels = path[start_index:end_index]

            image_png = Image.open(path)
            image = np.array(image_png)
            
            data.append((image, labels))

        return data
    
    def crop(self, image, coord, image_size = 28):
        """
        Crops a 2D nd.array to extract a singular digit.

        Args:
            image: 2D nd.array, large image to crop.
            coord: tuple, (x, y) top-left coordinate to crop.
            image_size: int, used to determine bottom-right coordinate to crop.

        Returns:
            2D nd.array, centered extracted digit
        """
        digit = image[
            coord[1] : coord[1] + image_size,
            coord[0] : coord[0] + image_size
        ]

        digit = self.center_digit(digit, image_size)
        
        return digit
    
    def center_digit(self, digit, image_size = 28):
        """
        Centers the cropped digit.

        Args:
            digit: 2D nd.array, image to center.
            image_size: int, size of the image to center digit in.

        Returns:
            2D nd.array, centered digit image
        """
        indices = np.where(digit != 0)

        start_x = np.min(indices[1])
        end_x = np.max(indices[1])
        start_y = np.min(indices[0])
        end_y = np.max(indices[0])

        shift_h = round(((image_size - end_x) - start_x) / 2)
        shift_v = round(((image_size - end_y) - start_y) / 2)

        centered = np.roll(digit, shift_h)
        centered = np.roll(centered, shift_v, axis = 0)

        return centered

    def predict(self, image):
        """
        Gets the prediction of a single digit image.

        Args:
            image: 2D nd.array, digit image to predict.

        Returns:
            int, 0 - 9 digit prediction
        """
        image = torch.from_numpy(image)
        image = image.float()
        image = image.unsqueeze(0)
        image /= 255

        prediction = self.predictor.predict(image)

        return prediction

    def validate(self, predictions, labels):
        """
        Calculates the accuracy of the predictions from all the digits.
        Note the order of predictions and labels does not always match.

        Args:
            predictions: list, contains all the predictions from the digits.
            labels: str, contains the validation labels for the digits.

        Returns:
            float, accuracy of all the digit predictions.
        """
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        sort_labels = labels.copy()
        sort_labels.sort()

        sort_predictions = predictions.copy()
        sort_predictions.sort()

        accuracy = 0

        for predict in sort_predictions:
            if predict in sort_labels:
                accuracy += 1
                sort_labels.remove(predict)

        accuracy /= float(len(labels))

        return accuracy
        