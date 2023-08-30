import random
from random import randrange
from mnist import MNIST
from pathlib import Path
from PIL import Image


class Dataset:
    def __init__(self):
        path = Path.cwd()
        # data = path.joinpath('analysis/data/MNIST/raw')
        data = path.joinpath('data/MNIST/raw')

        data = MNIST(data)
        self.images, self.labels = data.load_testing()      
        
        self.background_dim = 28 * 5  
        self.background = Image.new(
            mode='L',
            size=(self.background_dim, self.background_dim),
            color=0
        )

    def listToImage(self, image_list):
        pixels = image_list
        image = Image.new(mode='L', size=(28, 28))
        image.putdata(pixels)
        return image
    
    def intersection(self, positions, coord, image_size):
        for position in positions:
            rect1_x1 = coord[0]
            rect1_y1 = coord[1]
            rect1_x2 = coord[0] + image_size
            rect1_y2 = coord[1] + image_size
            rect2_x1 = position[0]
            rect2_y1 = position[1]
            rect2_x2 = position[0] + image_size
            rect2_y2 = position[1] + image_size

            inter_left = max(rect1_x1, rect2_x1)
            inter_top = max(rect1_y1, rect2_y1)
            inter_right = min(rect1_x2, rect2_x2)
            inter_bottom = min(rect1_y2, rect2_y2)

            if inter_right > inter_left and inter_bottom > inter_top:
                return True
            
        return False
    
    def generate(self, vary_size):
        path = Path.cwd().joinpath('dataset')
        # path = Path.cwd().joinpath('analysis/dataset')

        if vary_size:
            folder = 'nxn'
        else:
            folder = '28x28'
        file = 0

        dataset_path = path.joinpath(folder)
        
        for i in range(5):
            output = self.background.copy()
            
            positions = []
            labels = []
            iterations = 0

            while iterations < 10:
                index = randrange(len(self.images))
                
                label = self.labels[index]

                image = self.images[index]
                image = self.listToImage(image)

                if vary_size:
                    # Image scaling
                    # Determine a way to get image size
                    image_size = 40
                else:
                    image_size = 28

                random_range = self.background_dim - image_size
                x = randrange(random_range)
                y = randrange(random_range)

                if not self.intersection(positions, (x, y), image_size):
                    output.paste(image, (x, y))
            
                    positions.append((x, y))
                    labels.append(label)
                    
                    iterations = 0

                iterations += 1

            file_path = dataset_path.joinpath(f"{str(file)}_labels_{labels}.png")
            output.save(file_path)
            
            file += 1


if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate(vary_size=False)