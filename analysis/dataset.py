import numpy as np

from random import randrange
from mnist import MNIST
from pathlib import Path
from PIL import Image


class Dataset:
    """
    Generates a dataset of randomly placed MNIST digits on a black background image.
    Digit sizes can be 28x28 pixels or randomly vary in size.
    """
    def __init__(self):
        """
        Preprocesses the MNIST dataset and background image for analysis.
        """
        path = Path.cwd()

        # data = path.joinpath('data/MNIST/raw')
        # Debugger
        data = path.joinpath('analysis/data/MNIST/raw')

        data = MNIST(data)
        self.images, self.labels = data.load_testing()      
        
        self.background_dim = 28 * 5  
        self.background = Image.new(
            mode='L',
            size=(self.background_dim, self.background_dim),
            color=0
        )

    def _list_to_image(self, image_list):
        """
        Converts a list to a PIL Image.

        Args:
            image_list: The pixel values of the image to be converted.

        Returns:
            PIL.Image, digit image.
        """
        pixels = image_list
        image = Image.new(
            mode='L',
            size=(28, 28)
        )
        image.putdata(pixels)
        
        return image
    
    def _crop_digit(self, image):
        """
        Crops a digit leaving no border.

        Args:
            image: PIL.Image, image to extract digit.

        Returns:
            PIL.Image, digit with no surrounding border.
        """
        digit = np.array(image)
        indices = np.where(digit > 0.15)

        x = np.min(indices[1]) - 1
        y = np.min(indices[0]) - 1
        x2 = np.max(indices[1]) + 2
        y2 = np.max(indices[0]) + 2

        image = image.crop((x, y, x2, y2))
        
        return image

    def _intersection(self, positions, coord, size, vary_size):
        """
        Checks for intersection between two coordinates.

        Args:
            positions: list containing the top-left coordinate of already placed digits.
            coord: coordinate of the new digit to be placed.
            image_size: int, size of the intersection range to detect.

        Returns:
            True if intersection, False otherwise.
        """
        for position in positions:
            if vary_size:
                size2 = position[1]
                position = position[0]
            else:
                size2 = size
            
            rect1_x1 = coord[0]
            rect1_y1 = coord[1]
            rect1_x2 = coord[0] + size
            rect1_y2 = coord[1] + size
            rect2_x1 = position[0]
            rect2_y1 = position[1]
            rect2_x2 = position[0] + size2
            rect2_y2 = position[1] + size2

            inter_left = max(rect1_x1, rect2_x1)
            inter_top = max(rect1_y1, rect2_y1)
            inter_right = min(rect1_x2, rect2_x2)
            inter_bottom = min(rect1_y2, rect2_y2)

            if inter_right > inter_left and inter_bottom > inter_top:
                return True
            
        return False
    
    def generate(self, vary_size, digit_size = 28, num_images = 5):
        """
        Creates a dataset of randomly pasted digits on a black background, 
        checks for intersections before pasting the digit. 

        Args:
            vary_size: boolean value indicates if the images should vary in size.
                False - Task A
                True - Task B
            digit_size: int, size of the digits intersection restriction.
            num_images: int, number of images to generate.
        """
        # path = Path.cwd().joinpath('dataset')
        # Debugger

        path = Path.cwd().joinpath('analysis/dataset')
        
        if vary_size:
            folder = 'nxn'
        else:
            folder = f'{digit_size}x{digit_size}'
        
        dataset_path = path.joinpath(folder)

        file = 0

        for i in range(num_images):
            output = self.background.copy()
            
            positions = []
            labels = []
            iterations = 0

            while iterations < 10:
                index = randrange(len(self.images))
                
                label = self.labels[index]
                image = self.images[index]

                image = self._list_to_image(image)
                
                if digit_size != 28 or vary_size:
                    image = self._crop_digit(image)

                if vary_size:
                    height = randrange(12, 50)
                    
                    ratio = height / float(image.size[1])
                    width = int(float(image.size[0]) * float(ratio))
                    
                    image = image.resize(
                        (width, height),
                        Image.LANCZOS
                    )

                    image_size = max(image.size)
                    image.show()
                else:
                    image_size = digit_size

                random_range = self.background_dim - image_size
                x = randrange(random_range)
                y = randrange(random_range)

                intersection = self._intersection(
                    positions,
                    (x, y),
                    image_size,
                    vary_size
                )

                if not intersection:
                    output.paste(image, (x, y))
                    output.show()

                    if vary_size:
                        positions.append(((x, y), image_size))
                    else:
                        positions.append((x, y))

                    labels.append((x, label))
                    
                    iterations = 0

                iterations += 1

            labels.sort()

            for index in range(len(labels)):
                labels[index] = labels[index][1]
            
            file_path = dataset_path.joinpath(f"{str(file)}_labels_{labels}.png")
            output.save(file_path)
            
            file += 1

    def digit_dimension(self):
        """
        Measures the average, maximum and minimum width and height of the MNIST digits.
        """
        avg_width = 0
        avg_height = 0
        maximum = {
            'width' : 0,
            'height' : 0
        }
        
        for image in self.images:
            image = self._list_to_image(image)
            image = np.array(image)
            indices = np.where(image != 0)

            start_x = np.min(indices[1])
            end_x = np.max(indices[1])
            start_y = np.min(indices[0])
            end_y = np.max(indices[0])

            width = end_x - start_x
            height = end_y - start_y

            avg_width += width
            avg_height += height

            if width > maximum['width']:
                maximum['width'] = width
            if height > maximum['height']:
                maximum['height'] = height
        
        avg_width /= len(self.images)
        avg_height /= len(self.images)
        
        print("MNIST Digit Information")
        print(f"Average Width: {avg_width}")
        print(f"Average Height: {avg_height}")
        print(f"Max Width: {maximum['width']}")
        print(f"Max Height: {maximum['height']}")


if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate(
        vary_size = True,
        digit_size = 28,
        num_images = 5
    )
    # dataset.digit_dimension()