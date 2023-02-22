from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K
import math
from tensorflow.keras import regularizers
from pandas.core.computation.ops import Constant
from tensorflow.python.keras.utils import tf_utils
# implementation of VGG and Adacos with tf datasets
weight_decay = 1e-1


def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return tf_utils.constant_value(training)


# making VGG blocks
def vgg_block(layer_in, n_filters, n_conv):
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
    return layer_in


class AdaCos(Layer):

    def __init__(self,
                 num_classes,
                 is_dynamic=True,
                 regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._init_s = math.sqrt(2) * math.log(num_classes - 1)
        self._is_dynamic = is_dynamic
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)
        if self._is_dynamic:
            self._s = self.add_weight(shape=(),
                                      initializer=Constant(self._init_s, env=None),
                                      trainable=False,
                                      aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.squeeze(label)

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1)
        w = tf.nn.l2_normalize(self._w, axis=0)
        logits = tf.matmul(x, w)

        # Fixed AdaCos
        is_dynamic = tf_utils.constant_value(self._is_dynamic)
        if not is_dynamic:
            # _s is not created since we are not in dynamic mode
            output = tf.multiply(self._init_s, logits)
            return output

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels to update _s if we're not in training mode
            return self._s * logits
        else:
            theta = tf.math.acos(
                K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            one_hot = tf.one_hot(label, depth=self._n_classes)
            b_avg = tf.where(one_hot < 1.0,
                             tf.exp(self._s * logits),
                             tf.zeros_like(logits))
            b_avg = tf.reduce_mean(tf.reduce_sum(b_avg, axis=1))
            theta_class = tf.gather_nd(
                theta,
                tf.stack([
                    tf.range(tf.shape(label)[0]),
                    tf.cast(label, tf.int32)
                ], axis=1))
            mid_index = tf.shape(theta_class)[0] // 2 + 1
            theta_med = tf.nn.top_k(theta_class, mid_index).values[-1]

            # Since _s is not trainable, this assignment is safe. Also,
            # tf.function ensures that this will run in the right order.
            self._s.assign(
                tf.math.log(b_avg) /
                tf.math.cos(tf.minimum(math.pi / 4, theta_med)))

            return self._s * logits


def create_model():
    # starting VGG16 model
    visible = Input(shape=(28, 28, 1))
    label = Input(shape=(10,), dtype=tf.int32)
    layer = vgg_block(visible, 64, 2)
    layer = vgg_block(layer, 128, 2)
    layer = vgg_block(layer, 256, 3)
    layer = vgg_block(layer, 512, 3)
    # layer = vgg_block(layer, 512, 3)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output = AdaCos(10, regularizer=regularizers.l2(weight_decay))([layer, label])
    model = Model(inputs=[visible, label], outputs=output)

    return model
