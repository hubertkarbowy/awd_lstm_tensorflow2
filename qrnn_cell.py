import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTMCell
from tensorflow.python.keras import backend as K

class QRNNCell(object):
    def init(self, units, num_filters=2, k=2, use_bias=True, fo_dropout_rate=0.5):
        self.state_size = None
        self.output_size = None
        pass

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        pass

    def call(self, inputs, states, training=None):
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        pass
