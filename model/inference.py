from pathlib import Path
from PIL import Image
from torchvision import transforms

from prediction import Predictor

if __name__ == '__main__':
    current = Path.cwd()
    current = current.joinpath('images')

    predictions = []

    for i in range(1, 351):
        image_path = current.joinpath(f"img_{i}.jpg")
        image = Image.open(image_path)

        transform = transforms.Compose([transforms.PILToTensor()])
        image = transform(image)
        image = image.float()

        predictor = Predictor()
        prediction = predictor.predict(image)

        predictions.append(prediction)

    print(predictions)
