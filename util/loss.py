from tensorflow.keras.losses import Loss
from tensorflow.keras.backend import binary_crossentropy, mean
import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser(allow_no_value=True)
config.read('config.ini')
ROOT = config.get('training-parameters', 'root')


class WeightedBinaryCrossEntropy(Loss):
    def __init__(self, df):
        super().__init__()
        self.labels = df['Label'].values.tolist()
        self.classes = np.unique(self.labels)
        self.weights = self.balance_class_weights()

    def call(self, y_true, y_pred):
        bin_crossentropy = binary_crossentropy(y_true, y_pred)
        weight_0, weight_1 = self.weights
        weights = y_true*weight_1 + (1.-y_true)*weight_0
        weighted_binary_crossentropy = weights*bin_crossentropy

        return mean(weighted_binary_crossentropy)


    def balance_class_weights(self):
        count = [0] * len(self.classes)
        for item in self.labels:
            count[item] += 1
        weight_per_class = [0.] * len(self.classes)
        N = float(sum(count))
        for i in range(len(self.classes)):
            weight_per_class[i] = N / float(count[i])
        return weight_per_class


if __name__ == "__main__":
    df = pd.read_csv(f'{ROOT}/train_labeled_studies.csv', names=['Image', 'Label'], header=None)
    loss = WeightedBinaryCrossEntropy(df)