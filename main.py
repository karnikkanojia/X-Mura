import tensorflow as tf
from util.loader import get_train_gen, get_valid_gen
from models.densenet169 import get_densenet
from util.loss import WeightedBinaryCrossEntropy
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

df = pd.read_csv(f'{ROOT}/train_labeled_studies.csv', names=['Image', 'Label'], header=None)

loader = get_valid_gen()
model = get_densenet()
initializer = tf.keras.initializers.he_normal(seed=SEED)

model.trainable = True
for layer in model.layers:
    if 'conv5' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False


inputs = tf.keras.Input(shape=(512, 512, 3))
out = model(inputs=inputs)
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
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

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history = model.fit(loader, epochs=20, verbose=1)