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
               weight_dropout=None, # AWD rate
               verbose=False,
               **kwargs):
        self.weight_dropout = weight_dropout
        self.per_batch_mask = None
        self.awd_recurrent_kernel = None
        self.modified = False # a hack - this will lose the last batch
        self.verbose = False
        if recurrent_dropout > 0 and weight_dropout is not None:
            tf.print("WARNING: applying both AWD and recurrent dropout does not make sense - only the TF default version will be applied")
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

  def build(self, input_shape):
      super().build(input_shape)
      # we set up a separate recurrent kernel for use during the training phase:
      self.awd_recurrent_kernel = tf.Variable(self.recurrent_kernel,
                                              name="awd_recurrent_kernel",
                                              trainable=True)
      self.per_batch_mask = tf.Variable(lambda: tf.ones(self.recurrent_kernel.shape, dtype=tf.float32),
                                        name="awd_recurrent_mask",
                                        trainable=False)

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
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      z = K.dot(inputs, self.kernel)
      #z += K.dot(h_tm1, self.recurrent_kernel)
      if training is True:
        # z += K.dot(h_tm1, self.recurrent_kernel)
        z += K.dot(h_tm1, self.awd_recurrent_kernel)
        self.modified= True
      else:
        z += K.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)
  
      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)
  
    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    cfg = super().get_config()
    cfg.update({'weight_dropout': self.weight_dropout, 'verbose': self.verbose}) 
    return cfg

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    # We apply a workaround here. If we apply AWD dropout in `call`, this will create
    # lots of new tensors (one per timestep). Instead, we apply it to recurrent kernel
    # here because this function is called OUTSIDE the training phase.
    if self.modified: # FIXME: this is assigned inside the training loop AND WILL ALSO GET CALLED ON FIRST INFERENCE
      self.recurrent_kernel.assign(self.awd_recurrent_kernel) # copy weights after gradient updates
      self.awd_recurrent_kernel.assign(self.per_batch_mask * self.recurrent_kernel)
      self.modified = False
      self.per_batch_mask.assign(tf.nn.dropout(tf.fill(self.recurrent_kernel.shape, 0.5), rate=self.weight_dropout))
      if self.verbose:
        tf.print("~~~~~~  Updating the mask ~~~~~\n") # FIXME: Proper enable TF logging
        tf.print(self.per_batch_mask)
    else:
      self.per_batch_mask.assign(tf.ones(self.recurrent_kernel.shape, dtype=tf.float32))
    return super().get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    return cls(**config)
