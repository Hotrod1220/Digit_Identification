import ast
import csv
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
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
    def __init__(
        self,
        vary_size,
        digit_size = 28,
        folder = None
    ):
        """
        Initizes the data and visualization. Selects folder based on task.

        Args:
            vary_size: boolean value indicates if the images should vary in size.
                True - Task B
                False - Task C
            digit_size: int, indicates the size of the Task B digit boundaries.
            folder: str, manually select folder.
        """
        path = Path.cwd()
        
        if vary_size:
            file_folder = 'nxn'
        else:
            file_folder = f'{digit_size}x{digit_size}'

        if folder is not None:
            file_folder = folder
        
        self.dataset_path = path.joinpath(f'dataset/{file_folder}')

        csv_file = self.dataset_path.joinpath('annotations.csv')
        self.data = self.init_data(csv_file)

        visual_path = self.dataset_path.joinpath('visualization')
        self.visualize = Visualize(visual_path)
        
        self.predictor = Predictor()

    @abstractmethod
    def analyze(self):
        pass

    def init_data(self, path):
        """
        Initialized data with images and labels from the task folder.

        Args:
            paths: list containing the paths for the images, labels are from filename.

        Returns:
            list, tuples (image as a 2D nd.array, labels as a string).
        """
        data = []
        with open(path, 'r') as file:
            csvreader = csv.reader(file)
            
            for row in csvreader:
                image_png = Image.open(row[0])
                image = np.array(image_png)
                
                data.append((image, row[1]))

        return data
    
    def crop(self, image, coord, image_size = 28):
        """
        Crops a 2D nd.array to extract a singular digit.

        Args:
            image: 2D nd.array, large image to crop.
            coord: tuple, (x, y) top-left coordinate to crop.
            image_size: int, used to determine bottom-right coordinate to crop.

        Returns:
            2D nd.array, extracted digit
        """
        image = Image.fromarray(image)

        image = image.crop((
            coord[0],
            coord[1],
            coord[0] + image_size,
            coord[1] + image_size
        ))

        if image_size != 28:
            background = Image.new(
                mode='L',
                size=(28, 28),
                color=0
            )

            background.paste(image)
            image = background

        image = np.array(image)
        image = image.astype(float)
        image /= 255

        return image
    
    def center(self, digit, image_size = 28):
        """
        Centers a digit in an image.

        Args:
            digit: 2D nd.array, image to center.
            image_size: int, size of the image to center digit in.

        Returns:
            2D nd.array, centered digit image
        """
        indices = np.where(digit != 0)

        if indices[0].size == 0:
            return digit

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
            image: 2D nd.array | PIL.Image, digit image to predict.

        Returns:
            int, 0 - 9 digit prediction
        """
        if isinstance(image, Image.Image):
            digit = np.array(image)
            digit = Image.fromarray(digit * 255)

            transform = transforms.Compose([transforms.PILToTensor()])
            image = transform(image)
        else:
            digit = Image.fromarray(image * 255)
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
        
        image = image.float()
        
        if image.max() > 1:
            image /= 255

        prediction = self.predictor.predict(image)

        self.visualize.predictions.append((digit, prediction))

        return prediction

    def validate(self, predictions, labels):
        """
        Calculates the accuracy of the predictions and localization accuracy 
        from all the digits. Note the order of predictions and labels does 
        not always match.

        Args:
            predictions: list, contains all the predictions from the digits.
            labels: str, contains the validation labels for the digits.

        Returns:
            tuple[float, float], accuracy and localization accuracy of all 
            the digit predictions.
        """
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        sort_labels = labels.copy()
        sort_labels.sort()

        sort_predictions = predictions.copy()
        sort_predictions.sort()

        len_pred = len(sort_predictions)
        len_labels = len(sort_labels)
        
        if len_pred > len_labels:
            local_accuracy = (len_labels - (len_pred - len_labels)) / len_labels
        else:
            local_accuracy = len_pred / len_labels

        accuracy = 0
        for predict in sort_predictions:
            if predict in sort_labels:
                accuracy += 1
                sort_labels.remove(predict)

        accuracy /= float(len(labels))

        print(f"\nAccuracy: {accuracy:.2f}")
        print(f"Localization Accuracy: {local_accuracy:.2f}")
        print(f"Predictions: {predictions}, Labels: {labels}\n")

        return accuracy, local_accuracy
    
    def slices(self, image, boundaries, horizontal):
        """
        Slices a 2D array into sections according to a list of boundaries
        
        Args:
            image: 2D nd.array, image that will be sliced
            boundaries: list, the values where the image should be sliced
            horizontal: Boolean, if the image is to be sliced horizontally 
                        or vertically

        Returns:
            list, 2D nd.arrays, sliced images
        """
        slices = []
        
        for index in range(len(boundaries)):
            if index == 0:
                first = None
                second = boundaries[index]
            elif index == len(boundaries) - 1:
                first = boundaries[index]
                second = None
            else:
                first = boundaries[index]
                second = boundaries[index + 1]

            if horizontal:
                slices.append(image[:, first:second])
            else:
                slices.append(image[first:second, :])

            if index == 0:
                if len(boundaries) >= 2:
                    first = boundaries[index + 1]
                
                if horizontal:
                    slices.append(image[:, second:first])
                else:
                    slices.append(image[second:first, :])

        return slices
    
    def slice_image(self, image, x_boundary, y_boundary):
        """
        Slices an image into differnt sections according 
        to the boundaries given.

        Args:
            image: 2D nd.array, image to slice
            x_boundary: list, int values for where to slice 
                        the image horiztonally
            y_boundary: list, int values for where to slice 
                        the image vertically

        Returns:
            list, 2D nd.arrays, different slices of the image
            
        """
        if len(x_boundary) > 0:
            slices_x = self.slices(
                image,
                x_boundary,
                horizontal = True
            )

            if len(y_boundary) > 0:
                slices_xy = []

                for x_sliced in slices_x:
                    slices_xy += (
                        self.slices(
                            x_sliced,
                            y_boundary,
                            horizontal = False
                        )
                    )
            else:
                slices_xy = slices_x
                
        else:
            slices_xy = self.slices(
                image,
                y_boundary,
                horizontal = False
            )

        slices_xy = [
            sliced_image
            for sliced_image in slices_xy
            if sliced_image.size != 0 and sliced_image.max() > 0.1
        ]

        return slices_xy
    
    def remove_noise(self, image):
        """
        Detects and removes any pixel noise from a digit image.

        Args:
            image: 2D nd.array, digit image to analyze.

        Returns:
            2D nd.array, image with noise removed.
        """
        indices = np.where(image > 0.1)

        x = indices[1]
        y = indices[0]

        x_sort = x.copy()
        x_sort.sort()
        y_sort = y.copy()
        y_sort.sort()

        boundary_x = []
        boundary_y = []

        for index in range(len(x_sort) - 1):
            if (x_sort[index] - x_sort[index + 1] < -1):
                boundary_x.append(x_sort[index] + 1)
                
            if (y_sort[index] - y_sort[index + 1] < -1):
                boundary_y.append(y_sort[index] + 1)

        if len(boundary_x) == 1:
            slices = self.slices(
                image,
                boundary_x,
                horizontal = True
            )
        else:
            slices = self.slices(
                image,
                boundary_y,
                horizontal = False
            )

        large = (
            slices[0]
            if slices[0].shape > slices[1].shape
            else slices[1]
        )

        large = Image.fromarray(large * 255)

        background = Image.new(
            mode='L',
            size=(28, 28),
            color=0
        )

        background.paste(large)

        image = np.array(background)
        image = image.astype(float)
        image /= 255

        return image
