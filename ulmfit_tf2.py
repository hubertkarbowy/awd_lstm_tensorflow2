import tensorflow as tf
import tensorflow_text as text
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTMCell
from .awdlstm_tf2 import WeightDropLSTMCell

VOCAB_SIZE=35000
MAX_SEQ_LEN=200

# TODO:
# 1) [DONE] Weight tying
# 2) [DONE] use 1, not 0 for mask/pad
# 2b) [DONE] Include trainable into default signature when exporting a fixed-length Keras model
# 3) One-cycle policy,
# 4) [DONE] Optionally the tokenizer as module,
# 5) [DONE] Heads for finetuning LM + Head for text classification
# 6) [WON'T DO] Cross-batch statefulness

def tf2_ulmfit_encoder(*, fixed_seq_len=None, spm_model_file=None):
    """ This function reconstructs an ULMFiT as a model trainable in Keras. If `fixed_seq_len` is None,
        it uses RaggedTensors. Otherwise it sets a fixed sequence length on inputs and uses 1 (not zero!)
        for padding. As of TF 2.4.1 only the fixed-length version is serializable into a SavedModel.

        Returns four instances of tf.keras.Model:
        lm_model_num - encoder with a language modelling head on top (and weights tied to embeddings).
                       This version accepts already numericalized text.
                       * Example call (fixed length):

                       dziendobry = tf.constant([[11406,  7465, 34951,   218, 34992, 34967, 12545, 34986] + [1]*92])
                       lm_num(dziendobry)

                       Note that the final 92 padding tokens are masked throughout the model - this is taken care of
                       by the `compute_output_mask` in successive layers.

                       * Example call (variable length):

                       dziendobry = tf.ragged.constant([[11406,  7465, 34951,   218, 34992, 34967, 12545, 34986]])
                       lm_num(dziendobry)

        encoder_num - returns only the outputs of the last RNN layer (dim 400 as per ULMFiT paper). Accepts
                      already numericalized text. Calling convention is same as for lm_model_num.
                      Again, note the presence of _keras_mask in the output on padding tokens.

        outmask_num - returns explicit mask for an input sequence. Not used in the model itself, but might be useful
                      for working with some signatures in the serialized version.

  spm_encoder_model - the numericalizer. Accepts a string and outputs its sentencepiece representation. The SPM model
                      must be trained externally and a path needs to be provided in `spm_model_file` (it is also serialized
                      as a tf.saved_model.Asset). In a fixed length setting, this layer TRUNCATES the text
                      if it's longer than fixed_seq_len tokens and AUTOMATICALLY ADDS PADDING WITH A VALUE OF 1 if needed.
    """

    ##### STAGE 1 - FROM INPUT STRING TO A NUMERICALIZED REPRESENTATION TO EMBEDDINGS #####
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None) # initializer for embeddings
    if fixed_seq_len is None:
        string_input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.string, ragged=True, name="ragged_string_input")
        numericalized_layer = SPMNumericalizer(spm_path=spm_model_file,
                                               fixedlen=None,
                                               name="ragged_spm_numericalizer")(string_input_layer)
        print(f"Building an ULMFiT model with RaggedTensors. THIS IS NOT SERIALIZABLE TO a SavedModel as of TF 2.4.x - some promises " \
              f"to fix this were made as of version 2.5.0, so please check in the future. Save weights to a checkpoint " \
              f"instead and restore them with model.load_weights using Python code. If you need proper serialization, build" \
              f"a model with a fixed sequence length (if {fixed_seq_len} != {MAX_SEQ_LEN}), please adjust this either in function call " \
              f"or in code).")
        embedz = CustomMaskableEmbedding(VOCAB_SIZE,
                                         400,
                                         embeddings_initializer=uniform_initializer,
                                         mask_zero=False,
                                         name="ulmfit_embeds")
        encoder_dropout = RaggedEmbeddingDropout(encoder_dp_rate=0.4, name="ragged_emb_dropout")
        SpatialDrop1DLayer = RaggedSpatialDropout1D
        layer_name_prefix="ragged_"
    else:
        string_input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="fixedlen_string_input")
        numericalized_layer = SPMNumericalizer(spm_path=spm_model_file,
                                                   fixedlen=fixed_seq_len,
                                                   name="spm_numericalizer")(string_input_layer)
        print(f"Building an ULMFiT model with a fixed sequence length of {fixed_seq_len}.")
        if fixed_seq_len != MAX_SEQ_LEN:
            print(">>>>>>>>> WARNING, WARNING, WARNING! <<<<<<<<<<<<< ")
            print(f"Please make sure fixed_seq_len parameter ({fixed_seq_len}) and the constant MAX_SEQ_LEN declared in the code " \
                  f"are equal. If you are happy with just saving checkpoints or model.save('...'), you can probably ignore this, " \
                  f"but you won't be able to build the version exportable to TFHub / SavedModel")
        embedz = CustomMaskableEmbedding(VOCAB_SIZE,
                                         400,
                                         embeddings_initializer=uniform_initializer,
                                         mask_zero=False,
                                         mask_value=1,
                                         name="ulmfit_embeds")
        encoder_dropout = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")
        SpatialDrop1DLayer = tf.keras.layers.SpatialDropout1D
        layer_name_prefix=""

    input_dropout = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}inp_dropout")


    ###### STAGE 2 - RECURRENT LAYERS ######

    # AWD LSTM cells as per the said paper
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    rnn1 = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")
    rnn1_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop1") # yeah, this is quirky, but that's what ULMFit authors propose

    AWD_LSTM_Cell2 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    rnn2 = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")
    rnn2_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop2")

    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    rnn3 = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")
    rnn3_drop = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}rnn_drop3")


    ###### BASIC MODEL: From a numericalized input to RNN encoder vectors ######
    middle_input = tf.keras.layers.Input(shape=(fixed_seq_len,), dtype=tf.int32,
                                         name=f"{layer_name_prefix}numericalized_input",
                                         ragged=True if fixed_seq_len is None else False)
    explicit_mask = ExplicitMaskGenerator(mask_value=1)(middle_input)
    m = embedz(middle_input)
    m = encoder_dropout(m)
    m = input_dropout(m)
    m = rnn1(m)
    m = rnn1_drop(m)
    m = rnn2(m)
    m = rnn2_drop(m)
    m = rnn3(m)
    rnn_encoder = rnn3_drop(m)

    ###### OPTIONAL LANGUAGE MODELLING HEAD FOR FINETUNING #######
    fc_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer=embedz, activation='softmax'), name='lm_head_tied')
    fc_head_dp = tf.keras.layers.Dropout(0.05)
    lm = fc_head(rnn_encoder)
    lm = fc_head_dp(lm)

    ##### ALL MODELS ZUSAMMEN DO KUPY TOGETHER #####
    spm_encoder_model = tf.keras.Model(inputs=string_input_layer, outputs=numericalized_layer)
    lm_model_num = tf.keras.Model(inputs=middle_input, outputs=lm)
    encoder_num = tf.keras.Model(inputs=middle_input, outputs=rnn_encoder)
    outmask_num = tf.keras.Model(inputs=middle_input, outputs=explicit_mask)

    return lm_model_num, encoder_num, outmask_num, spm_encoder_model

class ExportableULMFiT(tf.keras.Model):
    """
    This class encapsulates a TF2 SavedModel serializable version of ULMFiT with a couple of useful
    signatures for flexibility.

    !!!!!!    Do not forget to call tf.keras.backend.set_learning_phase(0) before saving !!!!!!!

    Serialization procedure:

    lm_num, enc_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(spm_model_file='plwiki100-sp35k.model', fixed_seq_len=100)
    tf.keras.backend.set_learning_phase(0) # very important!!! do not skip!
    exportable = ExportableULMFiT(lm_model_num, encoder_num, outmask_num, spm_encoder_model)
    convenience_signatures={'numericalized_fixed_len': exported.fixed_len_numericalized,
                            'string_fixed_len': exported.fixed_len_string,
                            'numericalizer': exported.numericalizer}
    tf.saved_model.save(exported, 'ulmfit_tf2', signatures=convenience_signatures)


    Deserialization (you don't need any Python code and it really works with all the custom tweaks!):

    import tensorflow_text # must do it explicitly!
    import tensorflow_hub as hub
    import tensorflow as tf

    restored_hub = hub.load('ulmfit_tf2')   # now you can work with functions listed in signatures:
    hello_vectors = restored_hub.fixed_len_string(tf.constant([["Dzień dobry, ULMFiT!"]]))
    hello_vectors signatures['string_fixed_len'](tf.constant([["Dzień dobry, ULMFiT!"]]))

    Note the above examples return dictionaries with the encoder, LM head and mask outputs.
    If you want to use RNN vectors as a Keras layer you can access the serialized model
    directly like this:

    rnn_encoder = hub.KerasLayer(restored_hub.lm_model_str, trainable=True) # or .lm_model_num for numericalized inputs
    hello_vectors_padded = rnn_encoder(tf.constant([['Dzień dobry, ULMFiT']]))

    If you want, you can also manually verify that all the fancy dropouts from the ULMFiT paper are there:

    tf.keras.backend.set_learning_phase(1)
    Now call `rnn_encoder(tf.constant([['Dzień dobry, ULMFiT']]))` a couple of times - you will see
    values changing all the time (due to WeightDrop in the RNN layers) and some zeros (due to regular
    dropout on the output).
    """

    def __init__(self, lm_model_num, encoder_num, outmask_num, spm_encoder_model):
        super().__init__(self)
        self.lm_model_num = lm_model_num
        self.encoder_num = encoder_num
        self.masker_num = outmask_num
        self.spm_encoder_model = spm_encoder_model

        self.lm_model_str = tf.keras.Model(inputs=self.spm_encoder_model.inputs, outputs=self.lm_model_num(self.spm_encoder_model.outputs))
        self.encoder_str = tf.keras.Model(inputs=self.spm_encoder_model.inputs, outputs=self.encoder_num(self.spm_encoder_model.outputs))
        self.masker_str = tf.keras.Model(inputs=self.spm_encoder_model.inputs, outputs=self.masker_num(self.spm_encoder_model.outputs))

    @tf.function(input_signature=[tf.TensorSpec([None, MAX_SEQ_LEN], dtype=tf.int32)])
    def fixed_len_numericalized(self, numericalized):
        # return {'lm_head': self.lm_model(numericalized)}
        return {'lm_head': self.lm_model_num(numericalized),
                 'encoder': self.encoder_num(numericalized),
                 'explicit_mask': self.masker_num(numericalized)}

    @tf.function(input_signature=[tf.TensorSpec([None, 1], dtype=tf.string)])
    def fixed_len_string(self, string_inputs):
        return {'lm_head': self.lm_model_str(string_inputs),
                 'encoder': self.encoder_str(string_inputs),
                 'explicit_mask': self.masker_str(string_inputs)}

    @tf.function(input_signature=[tf.TensorSpec((None, 1), dtype=tf.string)])
    def numericalizer(self, string_inputs):
        return {'numericalized': self.spm_encoder_model(string_inputs)}

@tf.keras.utils.register_keras_serializable()
class SPMNumericalizer(tf.keras.layers.Layer):
    def __init__(self, name=None, spm_path=None, fixedlen=None,
                 pad_value=1, add_bos=False, add_eos=False, **kwargs):
        self.spm_path = spm_path
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.spm_asset = tf.saved_model.Asset(self.spm_path)
        self.spm_proto = tf.io.read_file(self.spm_asset).numpy()
        self.spmproc = text.SentencepieceTokenizer(self.spm_proto, add_bos=self.add_bos, add_eos=self.add_eos)
        self.fixedlen = fixedlen
        self.pad_value = pad_value
        super().__init__(name=name, **kwargs)
        self.trainable = False

    def build(self, input_shape):
        print(f">>>> INSIDE BUILD / SPMTOK <<<< {input_shape} ")
        super().build(input_shape)

    def call(self, inputs, training=None):
        ret = self.spmproc.tokenize(inputs)
        if self.fixedlen is not None:
            ret = tf.squeeze(ret, axis=1)
            ret_padded = ret.to_tensor(self.pad_value)
            ret_padded = tf.pad(ret_padded, tf.constant([[0,0,], [0,self.fixedlen,]]), constant_values=self.pad_value)
            ret_padded = ret_padded[:, :self.fixedlen]
            return ret_padded
        else:
            return tf.squeeze(ret, axis=1)

    def compute_output_shape(self, input_shape):
        tf.print(f"INPUT SHAPE IS {input_shape}")
        if self.fixedlen is None:
            # return tf.TensorShape(input_shape[0], None)
            return (input_shape[0], None)
        else:
            # return tf.TensorShape([input_shape[0], self.fixedlen])
            return (input_shape[0], self.fixedlen)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'spm_path': self.spm_path,
                    'fixedlen': self.fixedlen,
                    'pad_value': self.pad_value,
                    'add_bos': self.add_bos,
                    'add_eos': self.add_eos})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

class RaggedSparseCategoricalCrossEntropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, reduction='auto'):
        super().__init__(from_logits=from_logits, reduction=reduction)

    def call(self, y_true, y_pred):
        return super().call(y_true.flat_values, y_pred.flat_values)

@tf.keras.utils.register_keras_serializable()
class RaggedEmbeddingDropout(tf.keras.layers.Layer):
    def __init__(self, encoder_dp_rate, **kwargs):
        super(RaggedEmbeddingDropout, self).__init__(**kwargs)
        self.trainable = False
        self.encoder_dp_rate = encoder_dp_rate
        self.supports_masking = True
        self.bsize = None
        self._supports_ragged_inputs = True # for compatibility with TF 2.2

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD <<<< ")

    def call(self, inputs, training=None): # inputs is a ragged tensor now
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_embedding():
            """ Drops whole words. Almost, but not 100% the same as dropping them inside the encoder """
            flattened_batch = inputs.flat_values # inputs is a ragged tensor
            row_starts = inputs.row_starts() # size = batch size
            # row_length = input.row.lengths() 
            # bsize = tf.shape(inputs)[0]
            # seq_len = tf.shape(inputs)[1]
            tf.print(f"{tf.shape(flattened_batch)}")
            ones = tf.ones((tf.shape(flattened_batch)[0],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=1)) # axis is 1 because we still haven't restored the number of train examples in a batch
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_embedding,
                      lambda: array_ops.identity(inputs))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'encoder_dp_rate': self.encoder_dp_rate})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class EmbeddingDropout(tf.keras.layers.Layer):
    def __init__(self, encoder_dp_rate, **kwargs):
        super(EmbeddingDropout, self).__init__(**kwargs)
        self.trainable = False
        self.encoder_dp_rate = encoder_dp_rate
        self.supports_masking = True
        self.bsize = None

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD <<<< ")

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_embedding():
            """ Drops whole words. Almost, but not 100% the same as dropping them inside the encoder """
            bsize = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            ones = tf.ones((bsize, seq_len), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped = inputs * tf.expand_dims(dp_mask, axis=2)
            return dropped

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_embedding,
                      lambda: array_ops.identity(inputs))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'encoder_dp_rate': self.encoder_dp_rate})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class TiedDense(tf.keras.layers.Layer):
    def __init__(self, reference_layer, activation, **kwargs):
        self.ref_layer = reference_layer
        self.biases = None
        self.activation_fn = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
        self._supports_masking = self.supports_masking = True

    def build(self, input_shape):
        #self.biases = self.add_weight(name='tied_bias',
        #                              shape=[self.ref_layer.weights[0].shape[0]],
        #                              initializer='zeros')
        self.biases = self.add_weight(name='tied_bias',
                                      shape=[self.ref_layer.input_dim],
                                      initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        try:
            wx = tf.matmul(inputs, self.ref_layer.weights[0], transpose_b=True)
            z = self.activation_fn(wx + self.biases)
        except:
            tf.print("Warning, warning... - FORWARD PASS GOES TO NULL!")
            z = tf.matmul(inputs, tf.zeros((self.ref_layer.input_dim, self.ref_layer.output_dim)), transpose_b=True)
        return z

    def compute_output_shape(self, input_shape):
        tf.print(f"For TIED DENSE the input shape is {input_shape}")
        print(f"For TIED DENSE the input shape is {input_shape}")
        # return (input_shape[0], tf.shape(self.ref_layer.weights[0])[0])
        return (input_shape[0], self.ref_layer.weights[0].shape[0])
        # return (input_shape[0], 35000)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reference_layer': self.ref_layer, 'activation': self.activation_fn})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class ExplicitMaskGenerator(tf.keras.layers.Layer):
    """ Enhancement of TF's embedding layer where you can set the custom
        value for the mask token, not just zero. SentencePiece uses 1 for <pad>
        and 0 for <unk> and ULMFiT has adopted this convention too.
    """
    def __init__(self, mask_value=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = False

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        explicit_mask = tf.where(inputs == self.mask_value, False, True)
        explicit_mask = tf.cast(explicit_mask, dtype=tf.bool)
        return explicit_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mask_value': self.mask_value})
        return cfg

    @classmethod
    def from_config(cls, config):
        clazz = cls(**config)
        return clazz


@tf.keras.utils.register_keras_serializable()
class CustomMaskableEmbedding(tf.keras.layers.Embedding):
    """ Enhancement of TF's embedding layer where you can set the custom
        value for the mask token, not just zero. SentencePiece uses 1 for <pad>
        and 0 for <unk> and ULMFiT has adopted this convention too.
    """
    def __init__(self,
            input_dim,
            output_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_value=None,
            input_length=None,
            **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         embeddings_initializer=embeddings_initializer,
                         embeddings_regularizer=embeddings_regularizer,
                         activity_regularizer=activity_regularizer,
                         embeddings_constraint=embeddings_constraint,
                         input_length=input_length, **kwargs)
        self.mask_value = mask_value
        if self.mask_value is not None:
            self._supports_masking = True
            self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_value:
            return None
        return math_ops.not_equal(inputs, self.mask_value)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mask_value': self.mask_value})
        return cfg

    @classmethod
    def from_config(cls, config):
        clazz = cls(**config)
        return clazz

@tf.keras.utils.register_keras_serializable()
class RaggedSpatialDropout1D(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(RaggedSpatialDropout1D, self).__init__(**kwargs)
        self.trainable = False
        self.rate = rate
        self.supports_masking = True
        self.bsize = None
        self._supports_ragged_inputs = True # for compatibility with TF 2.2

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD / RSD<<<< ")

    def call(self, inputs, training=None): # inputs is a ragged tensor now
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_1d():
            """ Spatial 1D dropout which operates on ragged tensors """
            flattened_batch = inputs.flat_values # inputs is a ragged tensor
            row_starts = inputs.row_starts() # size = batch size
            # row_length = input.row.lengths() 
            # bsize = tf.shape(inputs)[0]
            # seq_len = tf.shape(inputs)[1]
            tf.print(f"{tf.shape(flattened_batch)}")
            ones = tf.ones((tf.shape(flattened_batch)[1],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=0)) # axis is 0 this time
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_1d,
                      lambda: array_ops.identity(inputs))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'rate': self.rate})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

