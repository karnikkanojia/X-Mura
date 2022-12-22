import tensorflow as tf
import configparser

config = configparser.ConfigParser(allow_no_value=True)
config.read('config.ini')
target_h = config.getint('training-parameters', 'target_size_h')
target_w = config.getint('training-parameters', 'target_size_w')
ROOT = config.get('training-parameters', 'seed')
TARGET_SIZE = (target_h, target_w, 3)


def get_densenet121() -> tf.keras.models.Model:
    model = tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=TARGET_SIZE,
        pooling=None
    )

    return model