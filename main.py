import tensorflow as tf
from util.loader import get_train_gen, get_valid_gen
from models.densenet169 import get_densenet
from util.loss import WeightedBinaryCrossEntropy
from util.metric import get_metrics
from logs.callbacks.callbacks import get_callbacks
import pandas as pd
import configparser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = configparser.ConfigParser()
config.read('config.ini')
SEED = config.getint('training-parameters', 'seed')
LR = config.getfloat('hyperparameters', 'learningrate')
BETA1 = config.getfloat('hyperparameters', 'beta1')
BETA2 = config.getfloat('hyperparameters', 'beta2')
ROOT = config.get('training-parameters', 'root')
EPOCHS = config.getint('training-parameters', 'epochs')
BATCH_SIZE = config.getint('training-parameters', 'batch_size')
target_h = config.getint('training-parameters', 'target_size_h')
target_w = config.getint('training-parameters', 'target_size_w')
INPUT_SHAPE = (target_h, target_w, 3)

df = pd.read_csv(f'{ROOT}/train_labeled_studies.csv',
                 names=['Image', 'Label'], header=None)

train_gen = get_train_gen()
valid_gen = get_valid_gen()
model = get_densenet()
initializer = tf.keras.initializers.he_normal(seed=SEED)

model.trainable = True
for layer in model.layers:
    if 'conv5' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False


inputs = tf.keras.Input(shape=INPUT_SHAPE)
out = model(inputs=inputs)
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(256, activation='relu',
                            kernel_initializer=initializer)(out)
out = tf.keras.layers.Dropout(0.4)(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.4)(out)
out = tf.keras.layers.Dense(units=1,
                            activation='sigmoid',
                            kernel_initializer=initializer
                            )(out)
model = tf.keras.models.Model(inputs=inputs, outputs=out)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    beta_1=BETA1,
    beta_2=BETA2
)

loss = WeightedBinaryCrossEntropy(df)
callbacks = get_callbacks()
metrics = get_metrics()

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_gen, epochs=EPOCHS, verbose=1, validation_data=valid_gen,callbacks=callbacks)
