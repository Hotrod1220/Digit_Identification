import scipy
import numpy as np

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

    def anaylze(self):
        """
        Analyzes the convolution image, extracting any digits
        and predicts which digits are present.
        """
        visual_path = self.dataset_path.joinpath('visualization')
        visual_path = visual_path.joinpath(
            f'{self.digit_size}x{self.digit_size}'
        )
        
        avg_accuracy = 0
        
        file = 0
        
        for data in self.data:
            image = data[0]
            labels = data[1]
            
            conv = self.convolution(image, self.digit_size)
            
            centers = self._centers(conv)
            centers.sort()

            predictions, visual = self.predictions(centers, image)

            file_path = visual_path.joinpath(f"{str(file)}_labels_{labels}.png")
            
            if isinstance(visual, Image.Image):
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
        Analyzes a convolution image to detect the general location of digits.

        Args: 
            conv: 2D nd.array, convolution image.

        Returns:
            list of coordinates where the center of the digits are.
        """
        indices = np.where(conv > 0.1)

        centers = []

        for index in range(len(indices[0])):
            x = indices[1][index]
            y = indices[0][index]
            y2 = y + 1
            y3 = y - 1
            x2 = x + 1
            x3 = x - 1
            idx_max = len(conv) - 1

            if y2 > idx_max:
                y2 = idx_max
            if y3 > idx_max:
                y3 = idx_max
            if x2 > idx_max:
                x2 = idx_max
            if x3 > idx_max:
                x3 = idx_max

            if conv[y][x] <= 0.1:
                continue

            v_gradient = conv[y3][x] < conv[y][x] >= conv[y2][x]
            h_gradient = conv[y][x3] < conv[y][x] >= conv[y][x2]
            h_equal = conv[y][x3] == conv[y][x] == conv[y][x2]
            square = conv[y][x] == conv[y][x2] == conv[y2][x] == conv[y2][x2]

            if (v_gradient and h_gradient or
                v_gradient and h_equal or
                square):
                if not self._repeated(centers, (x, y), boundary = 6):
                    centers.append((x, y))
        
        return centers
    
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
        visualize = []
        visual = image

        for center in centers:
            x = int(center[0] - self.digit_size) + 1
            y = int(center[1] - self.digit_size) + 1

            digit = self.crop(
                image = image,
                coord = (x, y),
                image_size = self.digit_size
            )

            if (self._single_digit(digit)):
                visualize = self._duplicated(
                    visualize,
                    (x, y),
                    boundary = 15,
                    image = image
                )

        visualize = list(dict.fromkeys(visualize))

        for center in visualize:
            x = center[0]
            y = center[1]

            digit = self.crop(
                image = image,
                coord = (x, y),
                image_size = self.digit_size
            )

            visual = self.visualize.boundary_box(
                visual, 
                (x, y),
                self.digit_size
            )
            
            clusters = self.clusters(digit)

            if len(clusters) != 1:
                digit = self.remove_noise(digit)
            
            digit = self.center(digit, image_size = 28)

            prediction = self.predict(digit)
            predictions.append(prediction)
                
        return predictions, visual

    def _single_digit(self, digit):
        """
        Analyzes the clusters discovered on cropped image to determine if 
        the image contains a single digit.

        Args:
            digit: 2D nd.array, cropped digit image

        Returns:
            True if single digit, False otherwise
        """
        clusters = self.clusters(digit)

        if clusters is None:
            return False
        
        if all(
            (cluster['height'] < 9 or cluster['height'] < 2)
            for cluster in clusters
        ):
            return False
        
        if len(clusters) == 1:
            return True
        
        if len(clusters) > 2:
            return False

        if (clusters[0]['width'] > clusters[1]['width'] and 
            clusters[0]['height'] > clusters[1]['height']):
            larger = clusters[0]
            smaller = clusters[1]
        else:
            larger = clusters[1]
            smaller = clusters[0]

        tiny_noise = (larger['width'] > 5 and larger['height'] > 14 and
                      (smaller['width'] < 5 or smaller['height'] < 5))

        if tiny_noise:
            return True
        
        return False
    
    def _repeated(self, centers, coord, boundary):
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
    
    def clusters(self, image):
        """
        Analyzes an image to determine the width and height of all clusters.

        Args:
            image: 2D nd.array, image containing the pixel clusters to detect.

        Returns:
            list, dictionaries containing the clusters information
        """
        indices = np.where(image > 0.1)

        if indices[0].size == 0:
            return None

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

        if len(boundary_x) == 0 and len(boundary_y) == 0:
            cluster = [{
                'width' : (max(x) - min(x)) + 1,
                'height' : (max(y) - min(y)) + 1,
            }]

            return cluster
            
        if len(boundary_x) > 0:
            slices_x = self.slices(
                boundary_x,
                image,
                horizontal = True
            )

            if len(boundary_y) > 0:
                slices_xy = []

                for x_sliced in slices_x:
                    slices_xy += (
                        self.slices(
                            boundary_y,
                            x_sliced,
                            horizontal = False
                        )
                    )
            else:
                slices_xy = slices_x
                
        else:
            slices_xy = self.slices(
                boundary_y,
                image,
                horizontal = False
            )

        slices_xy = [
            sliced_image
            for sliced_image in slices_xy
            if sliced_image.max() > 0.1
        ]

        clusters = []

        for xy in slices_xy:
            clusters += self.clusters(xy)

        return clusters
    
    def _duplicated(self, positions, coord, boundary, image):
        """
        Analyzes a list of digit positions to determine if a coordinate 
        is a duplicate. If it is, it determines which image is higher 
        quality and discards the poor qulaity digit.

        Args:
            positions: list, digit positions.
            coord: tuple, coordinate to check.
            boundary: range to analyze for duplicates.
            image: 2D nd.array, crop image in order to determine best digit.

        Returns:
            list, digit positions with duplicated removed.
        """
        index = 0
        duplicate = False
        
        while index < len(positions):
            center = (positions[index][0], positions[index][1])
            x_diff = coord[0] - center[0]
            y_diff = coord[1] - center[1]

            if center == coord:
                continue

            upper_bound = boundary
            lower_bound = upper_bound * -1
            if (x_diff < upper_bound and x_diff > lower_bound and
                y_diff < upper_bound and y_diff > lower_bound):
                
                digit_coord = self.crop(
                    image = image,
                    coord = (coord[0], coord[1]),
                    image_size = self.digit_size
                )
               
                digit_center = self.crop(
                    image = image,
                    coord = (center[0], center[1]),
                    image_size = self.digit_size
                )

                indices_coord = np.where(digit_coord > 0.1)
                indices_center = np.where(digit_center > 0.1)

                if len(indices_coord[0]) > len(indices_center[0]):
                    positions[index] = (coord[0], coord[1])
                
                duplicate = True
            index += 1
        
        if not duplicate:
            positions.append(coord)
                    
        return positions

if __name__ == '__main__':
    convolution = Convolution(digit_size = 20)
    convolution.anaylze()