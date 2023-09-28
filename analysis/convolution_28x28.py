import numpy as np

from convolution import Convolution


class Convolution28(Convolution):
    """
    Extracts the digits present in an image using a convoluton operation.
    """
    def __init__(self):
        """
        Initizes the task to Task A for the extraction. 
        """
        self.digit_size = 28
        super().__init__(self.digit_size)

    def analyze(self):
        super().analyze()
    
    def predictions(self, centers, image):
        """
        Generates all the predictions for an image given a list of centers.

        Args:
            centers: list, tuples containing all centers it found
            image: 2D nd.array, image that contains digits to be predicted

        Returns:
            list of int predictions
        """
        predictions = []
        visual = image

        for center in centers:
            x = int(center[0] - self.digit_size + self.digit_size / 4)
            y = int(center[1] - self.digit_size + self.digit_size / 4)

            visual = self.visualize.boundary_box(
                visual, 
                (x, y),
                self.digit_size
            )

            digit = self.crop(image, (x, y))
            digit = self.center(digit, image_size = 28)
            
            prediction = self.predict(digit)
            
            predictions.append(prediction)

        return predictions, visual

    def _centers(self, conv):
        """
        Analyzes a convolution image to detect where the centers of the digits are,
        finds areas of same pixel value to discover center.

        Args: 
            conv: nd.array, convolution image.

        Returns:
            list of coordinates where the center of the digits are.
        """
        indices = np.where(conv != 0)

        centers = []

        for index in range(len(indices[0])):
            x = indices[1][index]
            y = indices[0][index]

            if conv[y][x] == 0:
                continue

            if (conv[y][x] == conv[y + 2][x] == conv[y + 4][x] and
                conv[y][x] == conv[y][x + 3] == conv[y][x + 6] and
                conv[y + 2][x] == conv[y + 2][x + 3] == conv[y + 4][x + 6]):

                repeated = self.repeated(
                    centers,
                    (x, y),
                    self.digit_size
                )

                if not repeated:
                    centers.append((x, y))
        
        return centers
