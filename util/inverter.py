import numpy as np
import cv2
import configparser
import pathlib

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get('training-parameters', 'root')

mean = np.array([0.20577213, 0.20577213, 0.20577213])
std = np.array([0.17678809, 0.17678809, 0.17678809])
threshold = 255/2

path_to_data = f'{DATA_PATH}/'
paths = list(pathlib.Path(path_to_data).rglob('*.png'))
print(len(paths))

def invert(image):

    color = np.array([image[0:50, 0:50].mean(), image[-50:, -50:].mean(),
                      image[:50, -50:].mean(), image[-50:, :50].mean()]).mean()

    if image.mean() > threshold or color > threshold:
        image = 255 - image
    return image

print(len(paths))
for path in paths:
    print(f'Converting image: {path.as_posix()}')
    image = cv2.imread(path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = invert(image)
    cv2.imwrite(path.as_posix(), image)
