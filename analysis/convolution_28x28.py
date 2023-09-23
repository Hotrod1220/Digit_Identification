import scipy
import numpy as np

from analyze import Analyze


class Convolution(Analyze):
    """
    Extracts the digits present in an image using a convoluton operation.
    """
    def __init__(self):
        """
        Initizes the task to Task A for the extraction. 
        """
        vary_size = False
        super().__init__(vary_size)

    def anaylze(self):
        """
        Analyzes the convolution image, extracting any digits present.
        """
        visual_path = self.dataset_path.joinpath('visualization/28x28')
        
        avg_accuracy = 0
        
        file = 0
        
        for data in self.data:
            image = data[0]
            labels = data[1]

            self.kernel_dim = 28
            
            conv = self.convolution(image, self.kernel_dim)
            
            centers = self._centers(conv)
            centers.sort()

            predictions, visual = self.predictions(centers, image)

            file_path = visual_path.joinpath(f"{str(file)}_labels_{labels}.png")
            visual.save(file_path)
            
            file += 1
            
            accuracy = self.validate(predictions, labels)
            avg_accuracy += accuracy

        avg_accuracy /= (len(self.data) / 100)
        print(f"Average Accuracy: {avg_accuracy:.2f}")

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

                if not self._repeated(centers, (x, y)):
                    centers.append((x, y))
        
        return centers
    
    def predictions(self, centers, image):
        """
        Generates all the predictions for an image given a list of centers.

        Args:
            centers: list, tuples containing all centers it found
            image: np.array, image that contains digits to be predicted

        Returns:
            list of int predictions
        """
        predictions = []
        visual = image

        for center in centers:
            x = int(center[0] - self.kernel_dim + self.kernel_dim / 4)
            y = int(center[1] - self.kernel_dim + self.kernel_dim / 4)

            visual = self.visualize.boundary_box(
                visual, 
                (x, y),
                self.kernel_dim
            )

            digit = self.crop(image, (x, y))
            digit = self.center(digit, image_size = 28)
            
            prediction = self.predict(digit)
            
            predictions.append(prediction)

        return predictions, visual

    def _repeated(self, centers, coord):
        """
        Checks the surrounding area of a coordinate to detect if it is a repeat.

        Args:
            centers: list, containing coordinates of discovered digit centers.
            coord: tuple, (x, y) coordinate that is being checked if it is repeated.

        Returns:
            True if the coordinate is a repeat, False otherwise
        """
        for center in centers:
            x_diff = coord[0] - center[0]
            y_diff = coord[1] - center[1]

            upper_bound = self.kernel_dim
            lower_bound = self.kernel_dim * -1
            if (x_diff < upper_bound and x_diff > lower_bound and
                y_diff < upper_bound and y_diff > lower_bound):
                return True

        return False

if __name__ == '__main__':
    convolution = Convolution()
    convolution.anaylze()