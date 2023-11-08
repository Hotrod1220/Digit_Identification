import numpy as np

from divide import DivideConquer


class DivideConquerInner(DivideConquer):
    """
    Uses divide and conquer technique to identify handwritten 
    digits on an image. Digits can be contained inside another digit.
    """

    def __init__(self, scanned = False, folder = None):
        """
        Initizes the data and visualization. Selects folder based on task.

        Args:
            scanned: bool, if the images were scanned from paper.
            folder: str, manually select folder.
        """
        self._scanned = scanned
        super().__init__(scanned = scanned, folder = folder)
    
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
            2D nd.array, original image with extracted digits covered.
            list, 2D nd.array, extracted digits.
        """
        digits_info = []
        ext_digits = []

        for idx in pixels:
            if row:
                coord = (idx, img_idx)
            else:
                coord = (img_idx, idx)
            
            digits_info.append(self._cluster_dim(
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
        
        index = 0
        while index < len(digits_info) - 1:
            index2 = 1
            while index2 < len(digits_info):
                if digits_info[index][0] == digits_info[index2][0]:
                    digits_info.remove(digits_info[index])
                else:
                    index2 += 1
            index += 1

        for info in digits_info:
            dimension = info[0]         
            coords = info[1]         
            width = dimension[1] - dimension[0]
            height = dimension[3] - dimension[2]

            if ((width < 3 and height < 3 and 
                 len(digits_info) != 1)
            ):
                continue

            image, digit = self._crop(image, dimension, coords)
            ext_digits.append(digit)

        return image, ext_digits
    
    def _crop(self, image, dimension, coords):
        """
        Crops a 2D nd.array to extract a singular digit. Pastes a 
        black square where the digit was extracted. 

        Args:
            image: 2D nd.array, large image to crop.
            dimension: tuple, (x1, x2, y1, y2) coordinate to crop.

        Returns:
            2D nd.array, image with black square over digit.
            2D nd.array, extracted digit.
        """
        width = dimension[1] + 1 - dimension[0]
        height = dimension[3] + 1 - dimension[2]

        digit = np.zeros([height, width])

        for coord in coords:
            x = coord[0] - dimension[0]
            y = coord[1] - dimension[2]

            digit[y][x] = image[coord[1]][coord[0]]
            image[coord[1]][coord[0]] = 0.0

        return image, digit
