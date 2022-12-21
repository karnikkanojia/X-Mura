import tensorflow as tf
import pandas as pd
import configparser
from util.preprocessing import padding
import matplotlib.pyplot as plt
import numpy as np

config = configparser.ConfigParser(allow_no_value=True)
config.read('config.ini')


target_h = config.getint('training-parameters', 'target_size_h')
target_w = config.getint('training-parameters', 'target_size_w')
ROOT = config.get('training-parameters', 'root')
BATCH_SIZE = config.getint('training-parameters', 'batch_size')
SEED = config.getint('training-parameters', 'seed')
TARGET_SIZE = (target_h, target_w)
CLASS_MODE = 'binary'
SHUFFLE = config.getboolean('training-parameters', 'shuffle')


def generate_df(dataset_root, type='train'):
    if type == 'valid':
        csv_name = 'valid_image_paths.csv'
    elif type == 'train':
        csv_name = 'train_image_paths.csv'
    else:
        print('Invalid Argument for type -> Use only *train* and *valid*')
    df = pd.read_csv(f'{dataset_root}/{csv_name}',
                     header=None, names=['Image'])
    df['Label'] = (df.Image.str.extract('study.*_(positive|negative)'))
    return df


def preprocessing_fn(image):
    image = padding(image)
    return image


data_gen_args = dict(
    rescale=1./255.,
    preprocessing_function=preprocessing_fn
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

#! datagen.fit()
# Run datagen.fit() if the datagen args
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening are set to True


def get_train_gen():
    global datagen
    return datagen.flow_from_dataframe(generate_df(ROOT, 'train'),
                                       directory=None,
                                       target_size=TARGET_SIZE,
                                       class_mode=CLASS_MODE,
                                       batch_size=BATCH_SIZE,
                                       x_col='Image',
                                       y_col='Label')

def get_valid_gen():
    global datagen
    return datagen.flow_from_dataframe(generate_df(ROOT, 'valid'),
                                       directory=None,
                                       target_size=TARGET_SIZE,
                                       class_mode=CLASS_MODE,
                                       batch_size=BATCH_SIZE,
                                       x_col='Image',
                                       y_col='Label')

if __name__ == "__main__":
    # print(generate_df(ROOT, 'valid'))
    get_train_gen()
    get_valid_gen()

