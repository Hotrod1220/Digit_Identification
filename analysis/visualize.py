import numpy as np

from pathlib import Path
from PIL import ImageDraw, Image, ImageFont

class Visualize:
    """
    Used for providing visualization for PIL Images
    """
    def __init__(self, path):
        """
        Initializes the font, folder, multi-digit image, accuracy and 
        list of individual digit with their predictions.
        """
        self._path = path
        self.image = None
        self.accuracy = None
        self.predictions = []
        
        path = Path.cwd()
        font = str(path.joinpath('font/Poppins-Regular.ttf'))
        self._font = ImageFont.truetype(font, 10)

    def visualize(self, file):
        """
        Creates a new image with the multi-digit image, accuracy,
        individual digits and the prediction.

        Args:
            file: int, file name
        """
        predictions = self._image_predictions()

        x = 2
        y = 165

        height = y + (self.height + 6) * int(len(predictions) / 4 + 0.75)
        width = (self.width + 4) * 4

        background = Image.new(
            mode = 'L',
            size = (width, height),
            color = 255
        )
        
        accuracy = f"Accuracy: {round(self.accuracy * 100, 1)}%"
        image = ImageDraw.Draw(background)
        
        image.text(
            (75, 142),
            accuracy,
            font = self._font,
            fill = 0
        )

        for predict in predictions:
            x_image = int((width - 140) / 2)

            background.paste(self.image, (x_image, 0))
            background.paste(predict, (x, y))

            x += self.width + 4
            
            if x >= width:
                x = 0
                y += self.height + 6
        
        file_path = self._path.joinpath(f"{file}.png")
        background.save(file_path)

        self.image = None
        self.accuracy = None
        self.predictions = []

    def boundary_box(self, image, coord, box_size):
        """
        Creates an image with a boundary box indicated.

        Args:
            image: PIL Image that will have the boundary box placed on.
            coord: Tuple, (x, y) values to place boundary box
            box_size: Size of boundary box to place.

        Returns:
            PIL Image, Image with boundary box at coord.
        """
        coord = [
            coord[0],
            coord[1],
            coord[0] + box_size,
            coord[1] + box_size
        ]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        box = ImageDraw.Draw(image)
        box.rectangle(
            coord,
            outline='white'
        )

        return image
    
    def _image_predictions(self):
        """
        Converts list of tuples (image, prediction) to a 
        digit image with the prediction below the image.

        Returns
            list, image with the digit and the prediction displayed below.
        """
        images = []
        for predict in self.predictions:
            image = predict[0]
            prediction = predict[1]
            prediction = f"Predict: {prediction}"
            self.width = 56
            self.height = 45

            background = Image.new(
                mode = 'L',
                size = (self.width, self.height),
                color = 255
            )

            x = int((self.width - 28) / 2)
            background.paste(image, (x, 0))
            
            image = ImageDraw.Draw(background)

            image.text(
                (4, 29),
                prediction,
                font = self._font,
                fill = 0
            )

            images.append(background)

        return images
