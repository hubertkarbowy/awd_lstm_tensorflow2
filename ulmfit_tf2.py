import tensorflow as tf
import tensorflow_text as text
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTMCell
# from .awdlstm_tf2 import WeightDropLSTMCell

# TODO:
# 1) [DONE] Weight tying
# 2) [DONE] use 1, not 0 for mask/pad
# 2b) [DONE] Include trainable into default signature when exporting a fixed-length Keras model
# 3) One-cycle policy,
# 4) [DONE] Optionally the tokenizer as module,
# 5) [DONE] Heads for finetuning LM + Head for text classification
# 6) [WON'T DO] Cross-batch statefulness

def tf2_ulmfit_encoder(*, fixed_seq_len=None, flatten_ragged_outputs=True, spm_args={}):
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
                      must be trained externally and a path needs to be provided in `spm_args{'spm_model_file'}` (it is also serialized
                      as a tf.saved_model.Asset). In a fixed length setting, this layer TRUNCATES the text
                      if it's longer than fixed_seq_len tokens and AUTOMATICALLY ADDS PADDING WITH A VALUE OF 1 if needed.
    """

    ##### STAGE 1 - FROM INPUT STRING TO A NUMERICALIZED REPRESENTATION TO EMBEDDINGS #####
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None) # initializer for embeddings
    if fixed_seq_len is None:
        string_input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="ragged_string_input")
        spm_layer = SPMNumericalizer(spm_path=spm_args['spm_model_file'],
                                     add_bos=spm_args.get('add_bos') or False,
                                     add_eos=spm_args.get('add_eos') or False,
                                     fixedlen=None,
                                     lumped_sents_separator=spm_args.get('lumped_sents_separator') or False,
                                     name="ragged_spm_numericalizer")
        vocab_size = spm_layer.spmproc.vocab_size().numpy()
        numericalized_layer = spm_layer(string_input_layer)
        print(f"Building an ULMFiT model with: \n1) a variable sequence and RaggedTensors\n2) a vocabulary size of {vocab_size}.")
        print("=================================================================================================================")
        print(f"NOTE: THIS MODEL IS ONLY EXPERIMENTALLY SERIALIZABLE TO a SavedModel as of TF 2.4.x because of RaggedTensors." \
                "\nIt will output flat values and row splits, which you must then combine back into a RaggedTensor yourself:\n " \
                "\nret = model(tf.constant(['Hello, world', 'of RaggedTensors'])) " \
                "\nret_tensor = tf.RaggedTensor.from_row_splits(ret['output'][0], ret['output'][1]) " \
                "\n\nThere are plans to fix this around future TF version 2.5.0, until then we have to live with this" \
                "\nserialization workaround.")
        print("=================================================================================================================")
        embedz = CustomMaskableEmbedding(vocab_size,
                                         400,
                                         embeddings_initializer=uniform_initializer,
                                         mask_zero=False,
                                         name="ulmfit_embeds")
        encoder_dropout = RaggedEmbeddingDropout(encoder_dp_rate=0.4, name="ragged_emb_dropout")
        SpatialDrop1DLayer = RaggedSpatialDropout1D
        layer_name_prefix="ragged_"
    else:
        string_input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="fixedlen_string_input")
        spm_layer = SPMNumericalizer(spm_path=spm_args['spm_model_file'],
                                     fixedlen=fixed_seq_len,
                                     add_bos=spm_args.get('add_bos') or False,
                                     add_eos=spm_args.get('add_eos') or False,
                                     lumped_sents_separator=spm_args.get('lumped_sents_separator') or False,
                                     name="spm_numericalizer")
        vocab_size = spm_layer.spmproc.vocab_size().numpy()
        numericalized_layer = spm_layer(string_input_layer)
        print(f"Building an ULMFiT model with: \n1) a fixed sequence length of {fixed_seq_len}\n2) a vocabulary size of {vocab_size}.")
        embedz = CustomMaskableEmbedding(vocab_size,
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

    # Plain (TF optimized) LSTM cells - we will apply AWD manually in the training loop
    rnn1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform'), return_sequences=True, name="AWD_RNN1")
    rnn1_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop1") # yeah, this is quirky, but that's what ULMFit authors propose

    rnn2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform'), return_sequences=True, name="AWD_RNN2")
    rnn2_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop2")

    rnn3 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(400, kernel_initializer='glorot_uniform'), return_sequences=True, name="AWD_RNN3")
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
    if fixed_seq_len is None and flatten_ragged_outputs is True: # RaggedTensors as outputs are not serializable when using signatures. This *may* be fixed in TF 2.5.1
        encoder_num = tf.keras.Model(inputs=middle_input, outputs=[rnn_encoder.flat_values, rnn_encoder.row_splits])
    else:
        encoder_num = tf.keras.Model(inputs=middle_input, outputs=rnn_encoder)

    outmask_num = tf.keras.Model(inputs=middle_input, outputs=explicit_mask)
    return lm_model_num, encoder_num, outmask_num, spm_encoder_model

class ExportableULMFiT(tf.keras.Model):
    """
    This class encapsulates a TF2 SavedModel serializable version of ULMFiT with a couple of useful
    signatures for flexibility.

    !!!!!!    Do not forget to call tf.keras.backend.set_learning_phase(0) before saving !!!!!!!

    Serialization procedure:

    spm_args = {'spm_model_file': '/tmp/plwiki100-sp35k.model', 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    lm_num, enc_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=200, spm_args=spm_args)
    tf.keras.backend.set_learning_phase(0) # very important!!! do not skip!
    exportable = ExportableULMFiT(encoder_num, outmask_num, spm_encoder_model)
    convenience_signatures={'numericalized_encoder': exportable.numericalized_encoder,
                            'string_encoder': exportable.string_encoder,
                            'spm_processor': exportable.string_numericalizer}
    tf.saved_model.save(exportable, 'ulmfit_tf2', signatures=convenience_signatures)


    Deserialization (you don't need any Python code and it really works with all the custom tweaks!):

    import tensorflow_text # must do it explicitly!
    import tensorflow_hub as hub
    import tensorflow as tf

    restored_hub = hub.load('ulmfit_tf2')   # now you can work with functions listed in signatures:
    hello_vectors = restored_hub(tf.constant(["Dzień dobry, ULMFiT!"]))

    Note the above examples return dictionaries with the encoder, numericalized tokens and mask outputs.
    If you want to use RNN vectors as a Keras layer you can access the serialized model
    directly like this:

    rnn_encoder = hub.KerasLayer(restored_hub.encoder_str, trainable=True) # or .encoder_num for numericalized inputs
    hello_vectors = rnn_encoder(tf.constant(['Dzień dobry, ULMFiT']))

    If you want, you can also manually verify that all the fancy dropouts from the ULMFiT paper are there:

    tf.keras.backend.set_learning_phase(1)
    Now call `rnn_encoder(tf.constant([['Dzień dobry, ULMFiT']]))` a couple of times - you will see
    values changing all the time (due to WeightDrop in the RNN layers) and some zeros (due to regular
    dropout on the output).
    """

    def __init__(self, encoder_num, outmask_num, spm_encoder_model, lm_head_biases=None):
        super().__init__()
        self.encoder_num = encoder_num
        self.masker_num = outmask_num
        self.spm_encoder_model = spm_encoder_model
        self.lm_head_biases=tf.Variable(initial_value=lm_head_biases) if lm_head_biases is not None else None

        self.encoder_str = tf.keras.Model(inputs=self.spm_encoder_model.inputs, outputs=self.encoder_num(self.spm_encoder_model.outputs))

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def __call__(self, x):
        tf.print("WARNING: to obtain a trainable model, please wrap the `string_encoder` " \
                 "or `numericalized_encoder` signature into a hub.KerasLayer(..., trainable=True) object. \n")
        return self.string_encoder(x)

    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32)])
    def numericalized_encoder(self, numericalized):
        mask = self.masker_num(numericalized)
        return {'output': self.encoder_num(numericalized),
                'mask': mask}

    @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.float32)])
    def apply_awd(self, awd_rate):
        tf.print("Applying AWD in graph mode")
        rnn1_w = self.encoder_num.get_layer("AWD_RNN1").variables
        rnn2_w = self.encoder_num.get_layer("AWD_RNN2").variables
        rnn3_w = self.encoder_num.get_layer("AWD_RNN3").variables

        w1_mask = tf.nn.dropout(tf.fill(rnn1_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn1_w[1].assign(w1_mask * rnn1_w[1])

        w2_mask = tf.nn.dropout(tf.fill(rnn2_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn2_w[1].assign(w2_mask * rnn2_w[2])

        w3_mask = tf.nn.dropout(tf.fill(rnn3_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn3_w[1].assign(w3_mask * rnn3_w[2])

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def string_encoder(self, string_inputs):
        numerical_representation = self.string_numericalizer(string_inputs)
        hidden_states = self.numericalized_encoder(numerical_representation['numericalized'])['output']
        return {'output': hidden_states,
                'numericalized': numerical_representation['numericalized'],
                'mask': numerical_representation['mask']}

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def string_numericalizer(self, string_inputs):
        numerical_representation = self.spm_encoder_model(string_inputs)
        mask = self.masker_num(numerical_representation)
        return {'numericalized': numerical_representation,
                'mask': mask}

    ################## UNSUPPORTED / EXPERIMENTAL #################

    #@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32)])
    #def numericalized_lm_head(self, numericalized):
    #    # return {'lm_head': self.lm_model(numericalized)}
    #    return self.lm_model_num(numericalized)
    #
    #@tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    #def string_lm_head(self, string_inputs):
    #    return self.lm_model_str(string_inputs)
    #
    #
    #@tf.function(input_signature=[tf.TensorSpec((), dtype=tf.string)])
    #def spm_set_sentence_separator(self, sep):
    #    """ Insert additional <s> and </s> tokens between sentences in a single training example.

    #        This can be useful if working with short documents on which
    #        for some reason the model performs better if they are sentence-tokenized.
    #        
    #        The `sep` symbol is a separator by which each input example is split and
    #        surrounded by <s>...</s> tokens (only if `add_bos` and `add_eos` options
    #        for the SPMNumericalizer were set to True). For example, this input:

    #        The cat sat on a mat. [SEP] And spat.

    #        can be encoded as:
    #        <s> The cat sat on a mat. </s> <s> And spat. </s>
    #    """
    #    self.spm_encoder_model.layers[1].lumped_sents_separator.assign(sep)



class ExportableULMFiTRagged(tf.keras.Model):
    """ Same as ExportableULMFiT but supports RaggedTensors with a workaround """
    def __init__(self, encoder_num, outmask_num, spm_encoder_model, lm_head_biases=None):
        super().__init__()
        self.encoder_num = encoder_num
        self.masker_num = outmask_num
        self.spm_encoder_model = spm_encoder_model
        self.lm_head_biases=tf.Variable(initial_value=lm_head_biases) if lm_head_biases is not None else None

    # def __call__(self, x):
    #     rag_num = self.string_numericalizer(x)['numericalized']
    #     return self.numericalized_encoder(rag_num)

    # I have no clue how to wrap this around a hub.KerasLayer - please help!
    #@tf.function(input_signature=[tf.RaggedTensorSpec([None, None], dtype=tf.int32)])
    @tf.function(input_signature=[[tf.TensorSpec([None,], dtype=tf.int32),
                                   tf.TensorSpec([None,], dtype=tf.int64)]])
    def numericalized_encoder(self, x):
        flat_values=x[0]
        row_splits=x[1]
        ret = self.encoder_num(tf.RaggedTensor.from_row_splits(flat_values, row_splits))
        return {'output_flat': ret[0],
                'output_rows': ret[1]
        }

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def string_numericalizer(self, string_inputs):
        numerical_representation = self.spm_encoder_model(string_inputs)
        mask = self.masker_num(numerical_representation)
        return {'numericalized': numerical_representation,
                'mask': mask}

    @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.float32)])
    def apply_awd(self, awd_rate):
        tf.print("Applying AWD in graph mode and ragged tensors")
        rnn1_w = self.encoder_num.get_layer("AWD_RNN1").variables
        rnn2_w = self.encoder_num.get_layer("AWD_RNN2").variables
        rnn3_w = self.encoder_num.get_layer("AWD_RNN3").variables

        w1_mask = tf.nn.dropout(tf.fill(rnn1_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn1_w[1].assign(w1_mask * rnn1_w[1])

        w2_mask = tf.nn.dropout(tf.fill(rnn2_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn2_w[1].assign(w2_mask * rnn2_w[2])

        w3_mask = tf.nn.dropout(tf.fill(rnn3_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn3_w[1].assign(w3_mask * rnn3_w[2])
 

@tf.keras.utils.register_keras_serializable()
class SPMNumericalizer(tf.keras.layers.Layer):
    def __init__(self, name=None, spm_path=None, fixedlen=None,
                 pad_value=1, add_bos=False, add_eos=False, lumped_sents_separator="", **kwargs):
        self.spm_path = spm_path
        self.add_bos = add_bos
        self.add_eos = add_eos
        if isinstance(spm_path, tf.saved_model.Asset):
            self.spm_asset = spm_path
        else:
            self.spm_asset = tf.saved_model.Asset(self.spm_path)
        self.spm_proto = tf.io.read_file(self.spm_asset).numpy()
        self.spmproc = text.SentencepieceTokenizer(self.spm_proto, add_bos=self.add_bos, add_eos=self.add_eos)
        self.fixedlen = fixedlen
        self.pad_value = pad_value
        self.lumped_sents_separator = lumped_sents_separator
        #self.lumped_sents_separator = tf.Variable(initial_value=tf.keras.initializers.Constant(""),
        #                                          dtype=tf.string,
        #                                          name="spm_sent_sep",
        #                                          trainable=False)
        super().__init__(name=name, **kwargs)
        self.trainable = False

    def build(self, input_shape):
        print(f">>>> INSIDE BUILD / SPMTOK <<<< {input_shape} ")
        super().build(input_shape)
        #self.lumped_sents_separator = tf.Variable(initial_value=tf.keras.initializers.Constant(""),
        #                                          dtype=tf.string,
        #                                          name="spm_sent_sep",
        #                                          trainable=False)

    @tf.function
    def call(self, inputs, training=None):
        if tf.not_equal(self.lumped_sents_separator, ""):
            #splitted = text.regex_split(inputs, self.lumped_sents_separator.numpy().decode())
            splitted = tf.strings.split(inputs, self.lumped_sents_separator)
            ret = self.spmproc.tokenize(splitted)
            ret = ret.merge_dims(1, 2)
        else:
            ret = self.spmproc.tokenize(inputs)
        # ret = self.spmproc.tokenize(inputs)
        if self.fixedlen is not None:
            ret_padded = ret.to_tensor(self.pad_value)
            ret_padded = tf.squeeze(ret_padded, axis=1)
            ret_padded = tf.pad(ret_padded, tf.constant([[0,0,], [0,self.fixedlen,]]), constant_values=self.pad_value)
            ret_padded = ret_padded[:, :self.fixedlen]
            return ret_padded
        else:
            # ret = tf.squeeze(ret, axis=1)
            return ret
    
    # @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.string)])
    def set_sentence_separator(self, sep):
        """ Insert additional <s> and </s> tokens between sentences in a single training example.

            This can be useful if working with short documents on which
            for some reason the model performs better if they are sentence-tokenized.
            
            The `sep` symbol is a separator by which each input example is split and
            surrounded by <s>...</s> tokens (only if `add_bos` and `add_eos` options
            for the SPMNumericalizer were set to True). For example, this input:

            The cat sat on a mat. [SEP] And spat.

            can be encoded as:
            <s> The cat sat on a mat. </s> <s> And spat. </s>
        """
        self.lumped_sents_separator = sep

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
                    'add_eos': self.add_eos,
                    'lumped_sents_separator': ""})
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
            ones = tf.ones((tf.shape(flattened_batch)[0],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=1)) # axis is 1 because we still haven't restored the number of train examples in a batch
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        #ret = tf.cond(tf.convert_to_tensor(training),
        #              dropped_embedding,
        #              lambda: array_ops.identity(inputs))
        if training:
            ret = dropped_embedding()
        else:
            ret = array_ops.identity(inputs)
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

        # ret = tf.cond(tf.convert_to_tensor(training),
        #               dropped_embedding,
        #               lambda: array_ops.identity(inputs))
        if training:
            ret = dropped_embedding()
        else:
            ret = array_ops.identity(inputs)
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
        self.input_dim = None
        self.output_dim = None
        self.activation_fn = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
        self._supports_masking = self.supports_masking = True

    def build(self, input_shape):
        #self.biases = self.add_weight(name='tied_bias',
        #                              shape=[self.ref_layer.weights[0].shape[0]],
        #                              initializer='zeros')
        self.input_dim = self.ref_layer.variables[0].shape[0]
        self.output_dim = self.ref_layer.variables[0].shape[1]
        self.biases = self.add_weight(name='tied_bias',
                                      shape=[self.input_dim],
                                      initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        try:
            wx = tf.matmul(inputs, self.ref_layer.variables[0], transpose_b=True)
            z = self.activation_fn(wx + self.biases)
        except:
            tf.print("Warning, warning... - FORWARD PASS GOES TO NULL!")
            # z = tf.matmul(inputs, tf.zeros((self.ref_layer.input_dim, self.ref_layer.output_dim)), transpose_b=True)
            z = tf.matmul(inputs, tf.zeros((self.input_dim, self.output_dim)), transpose_b=True)
        return z

    def compute_output_shape(self, input_shape):
        # tf.print(f"For TIED DENSE the input shape is {input_shape}")
        # return (input_shape[0], tf.shape(self.ref_layer.weights[0])[0])
        return (input_shape[0], self.ref_layer.variables[0].shape[0])
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
    """ Explicitly return the propagated mask.

        This is useful after serialization where the original _keras_mask object is no longer available.
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
            ones = tf.ones((tf.shape(flattened_batch)[1],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=0)) # axis is 0 this time
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        # ret = tf.cond(tf.convert_to_tensor(training),
        #               dropped_1d,
        #               lambda: array_ops.identity(inputs))
        if training:
            ret = dropped_1d()
        else:
            ret = array_ops.identity(inputs)
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

@tf.keras.utils.register_keras_serializable()
class ConcatPooler(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self._supports_ragged_inputs = False # for compatibility with TF 2.2
    
    def build(self, input_shape):
        print(">>>> INSIDE BUILD / RaggedConcatPooler <<<< ")
    
    def call(self, inputs, training=None): # inputs is a fixed-length tensor
        last_hidden_state = inputs[:, -1, :] # nevermind padding - Keras mask ensures the last meaningful value is repeated until the sequence's end
        max_pooled = tf.math.reduce_max(inputs, axis=1)
        mean_pooled = tf.math.reduce_mean(inputs, axis=1)
        ret = tf.concat([last_hidden_state, max_pooled, mean_pooled], axis=1)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]*3)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class RaggedConcatPooler(tf.keras.layers.Layer):
    def __init__(self, inputs_are_flattened=False, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self._supports_ragged_inputs = True # for compatibility with TF 2.2
        self.inputs_are_flattened = inputs_are_flattened # set this to True if using the TFHub version

    def build(self, input_shape):
        print(">>>> INSIDE BUILD / ConcatPooler <<<< ")

    def call(self, inputs, training=None): # inputs is a ragged tensor
        if self.inputs_are_flattened:
            flat_vals = inputs[0]
            row_limits = inputs[1][1:] - 1 # this is row splits from first index minus 1
            last_hidden_states = tf.gather(flat_vals, row_limits)
            ragged_tensor = tf.RaggedTensor.from_row_splits(inputs[0], inputs[1])
        else:
            flat_vals = inputs.flat_values
            row_limits = inputs.row_limits() - 1
            ragged_tensor = inputs
            last_hidden_states = tf.gather(flat_vals, row_limits)

        max_pooled = tf.math.reduce_max(ragged_tensor, axis=1)
        mean_pooled = tf.math.reduce_mean(ragged_tensor, axis=1)
        ret = tf.concat([last_hidden_states, max_pooled, mean_pooled], axis=1)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]*3)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)
