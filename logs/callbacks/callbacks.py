import tensorflow as tf
import configparser
from datetime import datetime

config = configparser.ConfigParser()
config.read('config.ini')

LOG_DIR = config.get('logs', 'path') + datetime.now().strftime("%Y%m%d-%H%M%S")
DECAY = config.getfloat('hyperparameters', 'decay')
OUTPUT = config.get('logs', 'outputs')


def get_callbacks():
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=DECAY, patience=2, min_lr=1e6, verbose=1)
    model_cp = tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT, monitor='val_loss', verbose=1)

    return [tensorboard_callback, reduce_lr, model_cp]