# Customr L1 Distance layer


# Importing Dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance layer 
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        # input_embedding = tf.convert_to_tensor(input_embedding)
        # validation_embedding = tf.convert_to_tensor(validation_embedding)

        return tf.math.abs(input_embedding - validation_embedding)