import cv2
import copy
import numpy as np

from analyze import Analyze
from itertools import combinations
from PIL import Image


class DivideConquer(Analyze):
    """
    Uses divide and conquer technique to identify handwritten 
    digits on an image.
    """

    def __init__(self):
        super().__init__(vary_size=True)

    def analyze(self):
        """
        Analyze images with digits of varying sizes to predict 
        handwritten digits that are present.
        """
        avg_accuracy = 0
        file = 0

        for data in self.data:
            image = data[0]
            labels = data[1]

            image = image.astype("float")
            image /= 255.0

            predictions = self.predictions(image)
            accuracy = self.validate(predictions, labels)
            avg_accuracy += accuracy

            self.visualize.image = Image.fromarray(image * 255)
            self.visualize.accuracy = accuracy
            self.visualize.visualize(file)

            file += 1

        avg_accuracy /= len(self.data) / 100
        print(f"Average Accuracy: {avg_accuracy:.2f}%")

    def predictions(self, image):
        """
        Acquires and returns all the predictions for the 
        handwritten digits on an image.

        Args:
            image: 2D nd.array, image to analyze.

        Returns:
            list, 2D nd.array, all digits on an image.
        """
        predictions = []
        digits = self.divide(image)
        digits = [digit for digit in digits if digit.max() > 0.5]

        for digit in digits:
            if digit.shape[0] < 5 and digit.shape[1] < 5:
                continue

            digit = self.resize(digit)
            digit = self.center(digit, image_size=28)

            prediction = self.predict(digit)
            predictions.append(prediction)

        return predictions
    
    def divide(self, image):
        """
        Extracts digits by spliting image where black rows or columns
        are detected. If no rows or columns are detected, uses 
        algorithm inspired from DBScan to extract a digit. It then 
        covers the digit in the original image, then continues.

        Args:
            image: 2D nd.array, image to split.

        Returns:
            list, 2D nd.array, all digits present on an image.
        """
        digits = []
        x_zeros, y_zeros = self._zeros(image)

        if len(x_zeros) <= 1 and len(y_zeros) <= 1:
            count = self._num_clusters(image)

            if count > 1:
                row, img_idx = self._smallest_sum(image)
                pixels = self._pixels_slice(image, row, img_idx)
                
                image, extract = self._extract_digits(image, row, pixels, img_idx)

                if isinstance(image, list):
                    return image

                digits = digits + extract + self.divide(image)
            else:
                if image.shape[0] > 8 and image.shape[1] > 2:
                    return [image]
                else:
                    return []
        else:
            sections = self.slice_image(image, x_zeros, y_zeros)

            for xy in sections:
                digits += self.divide(xy)

        return digits
    
    def _extract_digits(self, image, row, pixels, img_idx):
        """
        Extracts all digits from an image using pixels from a 
        row or column. Detects the dimensions of a digit, extracts 
        the digit and covers the digit in the original image.

        Args:
            image: 2D nd.array, image to scan.
            row: bool, if row or column.
            pixels: pixels to apply algorithm to.
            img_idx: row or column index.

        Returns:
            2D nd.array, originial image with extracted digits covered.
            list, 2D nd.array, extracted digits.
        """
        dimensions = []
        ext_digits = []

        for idx in pixels:
            if row:
                coord = (idx, img_idx)
            else:
                coord = (img_idx, idx)
            
            dimensions.append(self._cluster_dim(
                image,
                coord,
                [],
                (
                    float("inf"),
                    float("-inf"),
                    float("inf"),
                    float("-inf"),
                )
            ))

        for d in combinations(dimensions, 2):
            if d[0] == d[1]:
                dimensions.remove(d[0])

        for dimension in dimensions:                    
            width = dimension[1] - dimension[0]
            height = dimension[3] - dimension[2]

            if ((width < 3 and height < 3 and 
                 len(dimensions) != 1)
            ):
                continue

            if image.shape[1] - width < 5 and image.shape[0] - height < 5:
                return [image], []

            image, digit = self._crop(image, dimension)
            ext_digits.append(digit)

        return image, ext_digits

    def resize(self, digit):
        """
        Resizes a digit back to the original 28x28 size.

        Args:
            digit: 2D nd.array, digit to resize.

        Returns:
            2D nd.array, digit resized to 28x28.
        """
        digit = Image.fromarray(digit * 255)

        height = 19

        ratio = height / float(digit.size[1])
        width = int(float(digit.size[0]) * float(ratio))

        digit = digit.resize((width, height), Image.LANCZOS)

        background = Image.new(
            mode="L",
            size=(28, 28),
            color=0
        )

        background.paste(digit)
        digit = background

        digit = np.array(digit)
        digit = digit.astype("float")
        digit /= 255

        return digit
    
    def _crop(self, image, coord):
        """
        Crops a 2D nd.array to extract a singular digit. Pastes a 
        black square where the digit was extracted. 

        Args:
            image: 2D nd.array, large image to crop.
            coord: tuple, (x1, x2, y1, y2) coordinate to crop.

        Returns:
            2D nd.array, image with black square over digit.
            2D nd.array, extracted digit.
        """
        image = Image.fromarray(image)

        digit = image.crop((
            coord[0],
            coord[2],
            coord[1] + 1,
            coord[3] + 1
        ))
        digit = np.array(digit)

        width = coord[1] + 1 - coord[0]
        height = coord[3] + 1 - coord[2]

        black = Image.new(
            mode='L',
            size=(width, height),
            color=0
        )
        image.paste(black, (coord[0], coord[2]))
        image = np.array(image)

        return image, digit

    def _zeros(self, image):
        """
        Scans an image to detect which rows and columns are black.

        Args:
            image: 2D nd.array, image to scan.

        Returns:
            list, int, index of black rows.
            list, int, index of black columns.
        """
        x = []
        y = []

        for index in range(image.shape[0]):
            y_zero = np.all(image[index, :] < 0.15)
            if y_zero:
                y.append(index)

        for idx in range(image.shape[1]):
            x_zero = np.all(image[:, idx] < 0.15)

            if x_zero:
                x.append(idx)

        return x, y

    def _num_clusters(self, image):
        """
        Detects the number of pixel clusters present in an image.

        Args:
            image: 2D nd.array, image to scan.

        Returns:
            int, number of clusters.
        """
        img = copy.deepcopy(image * 255)
        img = img.astype(np.uint8)

        _, thresh = cv2.threshold(
            img, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
            img
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        return len(contours)

    def _smallest_sum(self, image):
        """
        Detects the row or column with the least amount of pixels.

        Args:
            image: 2D nd.array, image to scan.

        Returns:
            bool, True if row, False if column
            int, row or column index.
        """
        x_min = float("inf")
        y_min = float("inf")

        for index in range(image.shape[0]):
            if index < 3 or image.shape[0] - index < 3:
                continue

            y = np.sum(image[index, :])

            if y < y_min:
                row = index
                y_min = y

        for idx in range(image.shape[1]):
            if idx < 3 or image.shape[1] - idx < 3:
                continue

            x = np.sum(image[:, idx])

            if x < x_min:
                col = idx
                x_min = x

        if y_min <= x_min:
            return True, row

        return False, col

    def _pixels_slice(self, image, row, img_idx):
        """
        Given a row or column, it detects the pixels present.
        If there are mulitple adjacent pixels, they are from 
        the same cluster, only saves one pixel per cluster.

        Args:
            image: 2D nd.array, image to scan.
            row: bool, if row or column.
            img_idx: index of row or column.

        Returns:
            list, int, a pixel from each cluster in row or column.
        """
        remove = []

        if row:
            pixels = np.where(image[img_idx, :] > 0.15)
        else:
            pixels = np.where(image[:, img_idx] > 0.15)

        pixels = list(pixels[0])

        for index in range(len(pixels) - 1):
            if pixels[index + 1] - pixels[index] == 1:
                remove.append(pixels[index])

        for x in remove:
            pixels.remove(x)

        return pixels
    
    def _cluster_dim(self, image, coord, past_coords, dimension):
        """
        Given a coordinate for a pixel, analyzes surrounding pixels
        that are not black to find digit dimensions.

        Args:
            image: 2D nd.array, image to scan.
            coord: tuple, (x, y), current pixel.
            past_coords: list, tuples, previous pixels.
            dimension: tuple, (x_min, x_max, y_min, y_max),
                       dimensions of digit detected.

        Returns:
            tuple, (x_min, x_max, y_min, y_max), dimensions of 
            digit detected.
        """
        if image.shape[0] < 4 and image.shape[1] < 4:
            return (0, 0, 0, 0)

        x = coord[0]
        y = coord[1]
        x_min = dimension[0]
        x_max = dimension[1]
        y_min = dimension[2]
        y_max = dimension[3]

        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

        past_coords.append(coord)
        first = len(past_coords) < 3

        left = (x - 1, y)
        top = (x, y - 1)
        right = (x + 1, y)
        down = (x, y + 1)

        directions = [left, top, right, down]

        for direction in directions:
            if direction in past_coords:
                continue
            
            def boundary_limit(d):
                return (
                    d[0] < 0 or 
                    d[0] > image.shape[1] - 1 or 
                    d[1] < 0 or
                    d[1] > image.shape[0] - 1
                )

            if not boundary_limit(direction):
                pixel = image[direction[1]][direction[0]]
            else:
                pixel = 0.0

            if pixel < 0.1:
                for direct in directions:
                    if (direct != direction and 
                        not boundary_limit(direct)
                    ):
                        next_pixel = image[direct[1]][direct[0]]
                        
                        if next_pixel < 0.1 and not first:
                            return (x_min, x_max, y_min, y_max)
                continue

            next_dim = self._cluster_dim(
                image = image,
                coord = (direction[0], direction[1]),
                past_coords = past_coords,
                dimension = (x_min, x_max, y_min, y_max)
            )

            if next_dim[0] < x_min:
                x_min = next_dim[0]
            if next_dim[1] > x_max:
                x_max = next_dim[1]
            if next_dim[2] < y_min:
                y_min = next_dim[2]
            if next_dim[3] > y_max:
                y_max = next_dim[3]

        return (x_min, x_max, y_min, y_max)
    