import csv
import os
import numpy as np

from dataset import Dataset
from random import randrange
from pathlib import Path
from PIL import Image


class DatasetInner(Dataset):
    """
    Generates a dataset of randomly placed MNIST digits on a black background image.
    Digit sizes can be 28x28 pixels or randomly vary in size.
    """
    def __init__(self):
        """
        Preprocesses the MNIST dataset and background image for analysis.
        """
        super().__init__()

    def generate(self, num_images = 5, density = 10):
        """
        Creates a dataset of randomly pasted digits on a black background, 
        checks for intersections before pasting the digit. Allows for digits
        to be contained within other digits, the digits must not intersect.

        Args:
            num_images: int, number of images to generate.
            density: int, measures how dense the image should be with numbers.
                     note: values above 200 slows the program significantly
        """
        path = Path.cwd().joinpath('dataset')        
        dataset_path = path.joinpath('nxn_inner')
        csv_path = dataset_path.joinpath('annotations.csv')
        
        if os.path.exists(csv_path):
            os.remove(csv_path)

        file = 0

        for i in range(num_images):
            output = self.background.copy()
            output = np.array(output)
            output = output.astype(float)
            output /= 255
            
            labels = []
            iterations = 0

            while iterations < density:
                index = randrange(len(self.images))
                
                label = self.labels[index]
                image = self.images[index]

                image = self._preprocess(image)

                random_x = self.background_dim - (image.shape[1] + 1)
                random_y = self.background_dim - (image.shape[0] + 1)
                x = randrange(random_x)
                y = randrange(random_y)

                intersection = self._intersection(image, output, (x, y))

                if not intersection:
                    output = self._paste(image, output, (x, y))
                    labels.append((x, label))
                    iterations = 0

                iterations += 1

            labels.sort()

            for index in range(len(labels)):
                labels[index] = labels[index][1]
            
            output = Image.fromarray(output * 255)
            output = output.convert('L')

            file_path = dataset_path.joinpath(f"images/{str(file)}.png")
            output.save(file_path)

            with open(csv_path, 'a') as f:
                csv_file = csv.writer(f)
                csv_file.writerow((file_path, labels))
            
            file += 1

    def _intersection(self, image, background, coord):
        """
        Checks for intersection between a digit's pixels and all other 
        not black pixels on the background image.

        Args:
            image: 2D nd.array, digit image.
            background: 2D nd.array, background image.
            coord: top-left coordinate of the new digit to be placed.

        Returns:
            True if intersection, False otherwise.
        """
        pixels = np.where(image > 0.1)
        pixels_x = pixels[1]
        pixels_y = pixels[0]

        for x, y in zip(pixels_x, pixels_y):
            x += coord[0]
            y += coord[1]

            bound_left = x - 1 > 0
            bound_right = x + 1 < background.shape[1]
            bound_top = y - 1 > 0
            bound_down = y + 1 < background.shape[0]

            if bound_left:                
                if background[y][x - 1] > 0.1:
                    return True
                
            if bound_right:
                if background[y][x + 1] > 0.1:
                    return True
                
            if bound_top:
                if background[y - 1][x] > 0.1:
                    return True

            if bound_down:
                if background[y + 1][x] > 0.1:
                    return True
                
            if bound_left and bound_top:
                if background[y - 1][x - 1] > 0.1:
                    return True
            
            if bound_left and bound_down:
                if background[y + 1][x - 1] > 0.1:
                    return True
            
            if bound_right and bound_top:
                if background[y - 1][x + 1] > 0.1:
                    return True
            
            if bound_right and bound_down:
                if background[y + 1][x + 1] > 0.1:
                    return True
                
            if background[y][x] > 0.1:
                return True
            
        return False
    
    def _paste(self, image, background, coord):
        """
        Places the pixels of a digit onto a background image.

        Args:
            image: 2D nd.array, digit.
            background: 2D nd.array, digit.
            coord: tuple, (x, y), position to place digit.

        Returns:
            2D nd.array, preprocess digit.
        """
        pixels = np.where(image > 0.1)
        pixels_x = pixels[1]
        pixels_y = pixels[0]

        for x, y in zip(pixels_x, pixels_y):
            x1 = x + coord[0]
            y1 = y + coord[1]
            background[y1][x1] = image[y][x]

        return background
    
    def _preprocess(self, image):
        """
        Converts to image, resizes and crops a digit image.

        Args:
            image: digit to preprocess.

        Returns:
            2D nd.array, background image with digit.
        """
        image = self._list_to_image(image)
        image = self._crop_digit(image)

        height = randrange(12, int(self.background_dim / 4))
        ratio = height / float(image.size[1])
        width = int(float(image.size[0]) * float(ratio))
        
        image = image.resize(
            (width, height),
            Image.LANCZOS
        )
        
        image = np.array(image)
        image = image.astype(float)
        image /= 255

        return image

if __name__ == '__main__':
    dataset = DatasetInner()
    dataset.generate(
        num_images = 50,
        density = 10
    )
