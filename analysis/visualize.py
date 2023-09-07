import numpy as np
from PIL import ImageDraw, Image

class Visualize:
    """
    Used for providing visualization for PIL Images
    """
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