from tensorflow.image import resize_with_crop_or_pad
import tensorflow as tf
import configparser


config = configparser.ConfigParser(allow_no_value=True)
config.read('config.ini')
target_h = config.getint('training-parameters', 'target_size_h')
target_w = config.getint('training-parameters', 'target_size_w')
ROOT = config.get('training-parameters', 'seed')

def padding(image):
    return resize_with_crop_or_pad(image, target_h, target_w)

if __name__ == "__main__":
    image = tf.keras.utils.load_img('MURA-v1.1/valid/XR_ELBOW/patient11186/study1_positive/image1.png') # Random image import
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, axis=0)
    print(padding(input_arr).shape)

