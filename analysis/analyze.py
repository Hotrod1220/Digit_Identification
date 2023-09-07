import numpy as np

from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from PIL import Image

from visualize import Visualize

class Analyze(ABC):
    """
    Abtract class that preprocesses the data for analysis children classes.
    """
    def __init__(self, vary_size):
        """
        Initizes the data and visualization. Selects folder based on task.

        Args:
            vary_size: boolean value indicates if the images should vary in size.
                False - Task A
                True - Task B
        """
        path = Path.cwd()
        # dataset_path = path.joinpath('dataset')
        dataset_path = path.joinpath('analysis/dataset')
        
        if vary_size:
            folder = 'nxn'
        else:
            folder = '28x28'
        
        data_path = str(dataset_path.joinpath(f"{folder}/*.png"))

        paths = glob(data_path)
        self.data = self.init_data(paths)

        self.visualize = Visualize()

    @abstractmethod
    def anaylze(self):
        pass

    def init_data(self, paths):
        """
        Initialized data with images and labels from the task folder.

        Args:
            paths: list containing the paths for the images, labels are from filename.

        Returns:
            List of tuples, (image as a 2D nd.array, labels as a string).
        """
        data = []
        start = 'labels_'
        end = '.png'

        for path in paths:
            start_index = path.find(start) + len(start)
            end_index = path.find(end)
            
            labels = path[start_index:end_index]

            image_png = Image.open(path)
            image = np.array(image_png)
            
            data.append((image, labels))
        return data
