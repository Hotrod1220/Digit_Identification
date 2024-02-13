import scipy
import numpy as np

from abc import abstractmethod
from analyze import Analyze
from PIL import Image

class Convolution(Analyze):
    """
    Extracts the digits present in an image using a convoluton operation.
    """
    def __init__(self, digit_size):
        """
        Initizes the task to Task A for the extraction. 
        """
        vary_size = False
        self.digit_size = digit_size
        super().__init__(vary_size, digit_size)

    @abstractmethod
    def predictions(self):
        pass

    def analyze(self):
        """
        Analyzes the convolution image, extracting any digits
        and predicts which digits are present.
        """        
        avg_accuracy = 0
        avg_local = 0
        
        file = 0
        
        for data in self.data:
            image = data[0]
            labels = data[1]
            
            conv = self.convolution(image, self.digit_size)

            conv_image = Image.fromarray(conv)
            conv_image.show()
            
            centers = self._centers(conv)
            centers.sort()

            predictions, visual = self.predictions(centers, image)            
            
            accuracy, local_accuracy = self.validate(predictions, labels)
            avg_accuracy += accuracy
            avg_local += local_accuracy

            self.visualize.image = visual
            self.visualize.accuracy = accuracy
            self.visualize.visualize(file)
            
            file += 1

        avg_accuracy /= (len(self.data) / 100)
        avg_local /= (len(self.data) / 100)
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Localization Accuracy: {avg_local:.2f}%")

    def convolution(self, image, kernel_dim):
        """
        Performs a 2D convolution on an image.

        Args:
            image: 2D nd.array, image that the convolution will be computed for.
            kernel_dim: int, dimension of the convolution kernel

        Returns:
            2D nd.array, the final convolution image.
        """
        kernel_dim -= 1
        kernel = np.array(kernel_dim * [kernel_dim * [1]])

        conv = scipy.signal.convolve2d(
            image,
            kernel
        )
        
        conv_image = conv / (conv.max() / 255.0)
        conv_image = np.uint8(conv_image)

        return conv_image    
    
    def repeated(self, centers, coord, boundary):
        """
        Checks the surrounding area of a coordinate to detect if it is a repeat.

        Args:
            centers: list, containing coordinates of discovered digit centers.
            coord: tuple, (x, y) coordinate that is being checked if it is repeated.
            boundary: the range around the coordinate.

        Returns:
            True if the coordinate is a repeat, False otherwise
        """
        for center in centers:
            x_diff = coord[0] - center[0]
            y_diff = coord[1] - center[1]

            upper_bound = boundary
            lower_bound = upper_bound * -1
            if (x_diff < upper_bound and x_diff > lower_bound and
                y_diff < upper_bound and y_diff > lower_bound):
                return True

        return False
    
    