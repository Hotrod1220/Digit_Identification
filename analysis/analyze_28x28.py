import scipy
import numpy as np

from PIL import Image

from analyze import Analyze

class Convolution(Analyze):
    """
    Extracts the digits present in an image using a convolution operation.
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
        for data in self.data:
            boxes = []

            image = data[0]
            # 19 is the MNIST digits max and min dimension
            kernel_dim = 19
            kernel = np.array(kernel_dim * [kernel_dim * [1]])
            
            conv = self.convolution(image, kernel)

            indices = np.where(conv != 0)
            
            for index in range(0, len(indices[0]), 1):
                x = indices[1][index]
                y = indices[0][index]

                is_repeat = self._check_repeat((x, y), boxes)

                if not is_repeat:
                    x = x - kernel_dim // 2
                    image = self.visualize.boundary_box(image, (x, y), 22)
                    boxes.append((x, y))

            image.show()


    def convolution(self, image, kernel):
        """
        Performs a 2D convolution on an image.

        Args:
            image: 2D nd.array, image that the convolution will be computed for.
            kernel: 2D nd.array, convolution filter to apply during convolution.

        Returns:
            2D nd.array, the final convolution image.
        """
        conv = scipy.signal.convolve2d(
            image,
            kernel
        )
        
        conv_image = conv / (conv.max() / 255.0)
        conv_image = np.uint8(conv_image)

        return conv_image
    
    def _check_repeat(self, coord, boxes):
        """
        To be done later.
        """
        x = coord[0]
        y = coord[1]
        for box in boxes:
            x_placed = box[0]
            y_placed = box[1]

            x_diff = x - x_placed
            y_diff = y - y_placed

            if (x_diff < 40 and x_diff >= -4) and (y_diff < 40 and y_diff >= -4):
                return True

        return False



if __name__ == '__main__':
    convolution = Convolution()
    convolution.anaylze()
