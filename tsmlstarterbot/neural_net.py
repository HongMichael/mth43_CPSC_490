import os
import tensorflow as tf
import numpy as np

from tsmlstarterbot.common import PLANET_MAX_NUM, MAP_MAX_HEIGHT, MAP_MAX_WIDTH, SCALE_FACTOR, NUM_IMAGE_LAYERS
from tensorflow.keras import datasets, layers, models

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.

if type(tf.contrib) != type(tf): tf.contrib._warning = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize features within each frame.
def normalize_input(input_data):

    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 4 and shape[1] == MAP_MAX_WIDTH//SCALE_FACTOR and shape[2] == MAP_MAX_HEIGHT//SCALE_FACTOR and shape[3] == NUM_IMAGE_LAYERS

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


class NeuralNet(object):
    FIRST_LAYER_SIZE = 12
    SECOND_LAYER_SIZE = 6

    def __init__(self, cached_model=None, seed=None):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, MAP_MAX_WIDTH//SCALE_FACTOR, MAP_MAX_HEIGHT//SCALE_FACTOR, NUM_IMAGE_LAYERS))

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))


            first_conv_layer = tf.contrib.layers.conv2d(self._features, 4, (3,3)) 
            first_pool_layer = tf.contrib.layers.max_pool2d(first_conv_layer, (2,2))
            second_conv_layer = tf.contrib.layers.conv2d(first_pool_layer, 8, (3,3))
            second_pool_layer = tf.contrib.layers.max_pool2d(second_conv_layer, (2,2))
            third_conv_layer = tf.contrib.layers.conv2d(second_pool_layer, 8, (3,3))
            flattened_layer = tf.contrib.layers.flatten(third_conv_layer)
            first_dense_layer = tf.contrib.layers.fully_connected(flattened_layer, 64)
            dropout_layer = tf.contrib.layers.dropout(first_dense_layer)
            logits = tf.contrib.layers.fully_connected(dropout_layer, PLANET_MAX_NUM)

            self._prediction_normalized = tf.nn.softmax(logits)

            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, MAP_MAX_WIDTH//SCALE_FACTOR, MAP_MAX_HEIGHT//SCALE_FACTOR, NUM_IMAGE_LAYERS)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: normalize_input(input_data),
                                               self._target_distribution: expected_output_data})
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (MAP_MAX_WIDTH//SCALE_FACTOR, MAP_MAX_HEIGHT//SCALE_FACTOR, NUM_IMAGE_LAYERS)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, MAP_MAX_WIDTH//SCALE_FACTOR, MAP_MAX_HEIGHT//SCALE_FACTOR, NUM_IMAGE_LAYERS)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: normalize_input(input_data),
                                            self._target_distribution: expected_output_data})

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)

