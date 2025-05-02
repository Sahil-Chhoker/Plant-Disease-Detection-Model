import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_visible_devices(gpus)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))