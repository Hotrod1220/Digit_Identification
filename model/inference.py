from pathlib import Path
from PIL import Image
from torchvision import transforms

from prediction import Predictor

if __name__ == '__main__':
    current = Path.cwd()
    current = current.joinpath('model/images_test')

    predictions = []
    predictor = Predictor()

    # for i in range(0, 351):
    #     image_path = current.joinpath(f"img_{i}.jpg")
    for i in range(0, 40):
        image_path = current.joinpath(f"pasted_{i}.jpg")
        # image_path = current.joinpath(f"cropped_{i}.jpg")
        image = Image.open(image_path)

        transform = transforms.Compose([transforms.PILToTensor()])
        image = transform(image)
        image = image.float()
        image /= 255

        prediction = predictor.predict(image)

        predictions.append(prediction)

    print(predictions)
