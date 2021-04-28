import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K

@tf.keras.utils.register_keras_serializable()
class WeightDropLSTMCell(tf.keras.layers.LSTMCell):
  """ Weight-dropped Long short-term memory unit (AWD-LSTM) recurrent network cell.
      Adapted from Tensorflow 2.4.1 source code for LSTMCell available here:

      https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/recurrent_v2.py
  """
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               weight_dropout=None, # zeros and ones
               **kwargs):
        self.weight_dropout = weight_dropout
        self.per_batch_mask = None
        super(WeightDropLSTMCell, self).__init__(
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=recurrent_regularizer,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels.
       Uses weight dropout on recurrent (U) matrices if requested
    """
    x_i, x_f, x_c, x_o = x
    training = K.learning_phase()
    if training:
        dropped_recurrent_kernel = self.recurrent_kernel * self.per_batch_mask
    else:
        dropped_recurrent_kernel = self.recurrent_kernel
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, dropped_recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, dropped_recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, dropped_recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, dropped_recurrent_kernel[:, self.units * 3:]))
    return c, o

  def build(self, input_shape):
      super().build(input_shape)
      self.per_batch_mask = tf.ones(self.recurrent_kernel.shape, dtype=tf.float32)

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state
  
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)
  
    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)
  
      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      # tf.print("Implementacja druga")
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      z = K.dot(inputs, self.kernel)
      if training is True:
        # z += K.dot(h_tm1, self.recurrent_kernel)
        # ;STAD - cieknie z linijki ponizej, choc juz mniej niz poprzednio.
        # Moze trzeba jakos znalezc
        # sposob, zeby self.per_batch_mask tu nie siedzialo?
        # Tylko jak to kurcze podac do *warstwy* RNN, a nie do *komorki*?
        # Bo ten kod (w call) odpala sie za kazdym obrotem komorki.
        # Po kerasowej warstwie tf.keras.LSTM zdaje sie nie da sie dziedziczyc.
        # Nie wiem czy nie trzeba bedzie robic wielu wejsc tylko dla maski...
        z += K.dot(h_tm1, self.per_batch_mask * self.recurrent_kernel)
      else:
        z += K.dot(h_tm1, self.recurrent_kernel)
      #z += K.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)
  
      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)
  
    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    cfg = super().get_config()
    cfg.update({'weight_dropout': self.weight_dropout}) 
    return cfg

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
      training = tf.keras.backend.learning_phase()
      if training:
          tf.print("~~~~~~ Setting the initial state (czemu tego nie widac??) ~~~~~")
          # ;STAD CHYBA TEZ CIEKNIE
          # self.per_batch_mask = tf.nn.dropout(tf.fill(self.recurrent_kernel.shape, 0.5), rate=self.weight_dropout)
      return super().get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
      # return original_states # + [per_batch_mask]


  @classmethod
  def from_config(cls, config):
    return cls(**config)




@tf.keras.utils.register_keras_serializable()
class WeightDropLSTMCell_v2(tf.keras.layers.LSTMCell):
  """ Weight-dropped Long short-term memory unit (AWD-LSTM) recurrent network cell.
      Adapted from Tensorflow 2.4.1 source code for LSTMCell available here:

      https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/recurrent_v2.py
  """
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               weight_dropout=None, # zeros and ones
               **kwargs):
        self.weight_dropout = weight_dropout
        self.per_batch_mask = None
        super(WeightDropLSTMCell_v2, self).__init__(
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=recurrent_regularizer,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels.
       Uses weight dropout on recurrent (U) matrices if requested
    """
    x_i, x_f, x_c, x_o = x
    training = K.learning_phase()
    if training:
        dropped_recurrent_kernel = self.recurrent_kernel * self.per_batch_mask
    else:
        dropped_recurrent_kernel = self.recurrent_kernel
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, dropped_recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, dropped_recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, dropped_recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, dropped_recurrent_kernel[:, self.units * 3:]))
    return c, o

  def build(self, input_shape):
      super().build(input_shape[1])
      self.per_batch_mask = tf.ones(self.recurrent_kernel.shape, dtype=tf.float32)

  def call(self, inputz, statez, training=None):
    inputs, carried_mask = tf.nest.flatten(inputz)
    states, carried_state = statez
    h_tm1 = states[0][0]  # previous memory state
    c_tm1 = states[0][1]  # previous carry state
  
    dp_mask = self.get_dropout_mask_for_cell(inputs[0], training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)
  
    # tf.print("Implementacja druga")
    if 0. < self.dropout < 1.:
      inputs = inputs * dp_mask[0]
    z = K.dot(inputs, self.kernel)
    if training is True:
      # z += K.dot(h_tm1, self.recurrent_kernel)
      # ;STAD - cieknie z linijki ponizej, choc juz mniej niz poprzednio.
      # Moze trzeba jakos znalezc
      # sposob, zeby self.per_batch_mask tu nie siedzialo?
      # Tylko jak to kurcze podac do *warstwy* RNN, a nie do *komorki*?
      # Bo ten kod (w call) odpala sie za kazdym obrotem komorki.
      # Po kerasowej warstwie tf.keras.LSTM zdaje sie nie da sie dziedziczyc.
      # Nie wiem czy nie trzeba bedzie robic wielu wejsc tylko dla maski...
      z += K.dot(h_tm1, self.per_batch_mask * self.recurrent_kernel)
    else:
      z += K.dot(h_tm1, self.recurrent_kernel)
    #z += K.dot(h_tm1, self.recurrent_kernel)
    if self.use_bias:
      z = K.bias_add(z, self.bias)
  
    z = array_ops.split(z, num_or_size_splits=4, axis=1)
    c, o = self._compute_carry_and_output_fused(z, c_tm1)
  
    h = o * self.activation(c)
    return (h, carried_mask), ([h, c], carried_state)

  def get_config(self):
    cfg = super().get_config()
    cfg.update({'weight_dropout': self.weight_dropout}) 
    return cfg

  # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
  #     training = tf.keras.backend.learning_phase()
  #     if training:
  #         tf.print("~~~~~~ Setting the initial state (czemu tego nie widac??) ~~~~~")
  #         # ;STAD CHYBA TEZ CIEKNIE
  #         # self.per_batch_mask = tf.nn.dropout(tf.fill(self.recurrent_kernel.shape, 0.5), rate=self.weight_dropout)
  #     return super().get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
  #     # return original_states # + [per_batch_mask]


  @classmethod
  def from_config(cls, config):
    return cls(**config)

