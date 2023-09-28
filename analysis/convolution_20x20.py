import numpy as np

from convolution import Convolution


class Convolution20(Convolution):
    """
    Extracts the digits present in an image using a convoluton operation.
    """
    def __init__(self):
        """
        Initizes the task to Task A for the extraction. 
        """
        self.digit_size = 20
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

            if self._single_digit(digit):
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
            
            clusters = self._clusters(digit)

            if len(clusters) != 1:
                digit = self.remove_noise(digit)
            
            digit = self.center(digit, image_size = 28)

            prediction = self.predict(digit)
            predictions.append(prediction)
                
        return predictions, visual
    
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

                repeated = self.repeated(
                    centers,
                    (x, y),
                    boundary = 6
                )

                if not repeated:
                    centers.append((x, y))
        
        return centers

    def _single_digit(self, digit):
        """
        Analyzes the clusters discovered on cropped image to determine if 
        the image contains a single digit.

        Args:
            digit: 2D nd.array, cropped digit image

        Returns:
            True if single digit, False otherwise
        """
        clusters = self._clusters(digit)

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
    
    def _clusters(self, image):
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
            clusters += self._clusters(xy)

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
