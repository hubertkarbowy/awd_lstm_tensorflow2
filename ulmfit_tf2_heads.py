import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras.engine import keras_tensor
# from .awdlstm_tf2 import *
from .ulmfit_tf2 import *

def ulmfit_rnn_encoder_native(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args,
                              also_return_spm_encoder=False):
    """ Returns an ULMFiT encoder from Python code """
    print("Building model from Python code (not tf.saved_model)...")
    lm_num, enc_num, _, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, spm_args=spm_model_args,
                                                               flatten_ragged_outputs=False)
    if pretrained_weights is not None:
        print("Restoring weights from file....")
        lm_num.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! Make sure to restore them from file.")
    if also_return_spm_encoder is True:
        return enc_num, spm_encoder_model
    else:
        return enc_num

def ulmfit_rnn_encoder_hub(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args=None,
                              also_return_spm_encoder=False):
    """ Returns an ULMFiT encoder from a serialized SavedModel  """
    if also_return_spm_encoder:
        print(f"Info: The SPM layer is baked into the SavedModel. It will not be returned separately.")
    if spm_model_args is not None:
        print(f"Info: When restoring the ULMFiT encoder from a SavedModel, `spm_model_args` has no effect`")
    restored_hub = hub.load(pretrained_weights)

    if fixed_seq_len == None: # TODO: check tensorspec and print a pretty info if ragged parameters here and in the SavedModel are incompatible
        il = tf.keras.layers.Input(shape=(None,), ragged=True, name="numericalized_input", dtype=tf.int32)
        #il_rows = tf.keras.layers.Input(shape=(None,), name="rowsplits", dtype=tf.int64)
        # kl = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="ulmfit_encoder")
        kl = hub.KerasLayer(restored_hub.encoder_num)(il)
        # model = tf.keras.models.Model(inputs=il, outputs=rec_ragged_tensor)
    else:
        il = tf.keras.layers.Input(shape=(fixed_seq_len,), name="numericalized_input", dtype=tf.int32)
        kl = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="ulmfit_encoder")(il)['output']
        #model = tf.keras.models.Model(inputs=il, outputs=kl)
        #model = tf.keras.models.Model(inputs=il, outputs=kl)
    return il, kl, restored_hub

def ulmfit_sequence_tagger(*, enc_num, model_type='from_cp', num_classes=3, fixed_seq_len=None, **kwargs):
    """ Convenience method to create a sequence-tagging head on top of the ULMFiT encoder.
    
        `enc_num`       - the ULMFiT encoder
        `model_type`    - 'from_cp' if encoder is built from Python code or `from_hub` if it's restored from a SavedModel
        `num_classes`   - how many classes there are
        `fixed_seq_len` - (for SavedModel only) Fixed sequence length to which all training examples will be padded
                                                if necessary. `None` means the SavedModel will use RaggedTensors and
                                                variable sequence length. You must set this parameter exactly in the
                                                same way as you did when exporting the encoder.
        `kwargs`        - (for SavedModel only) contains two keras objects:
                          'input_layer' - an instance of tf.keras.layers.Input which can accept ragged
                                          or fixed-length inputs, depending on how the encoder was exported.
                          'keras_layer' - the encoder output either as a fixed-sequence length batch, or
                                          an array of [flat_values, row_splits] restored back into a RaggedTensor.
    """
    if model_type == 'from_hub':
        print(f"Adding sequence tagging head with n_classes={num_classes} (TF Hub)")
        if fixed_seq_len is None:
            il = kwargs['input_layer']
            kl = kwargs['keras_layer']
            ragged_restored = tf.RaggedTensor.from_row_splits(kl[0], kl[1])
            tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes))(ragged_restored)
            tagger_model = tf.keras.models.Model(inputs=il, outputs=tagger_head)
        else:
            il = kwargs['input_layer']
            kl = kwargs['keras_layer']
            tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes))(kl)
            tagger_model = tf.keras.models.Model(inputs=il, outputs=tagger_head)
    else:
        print(f"Adding sequence tagging head with n_classes={num_classes}")
        tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(enc_num.output)
        tagger_model = tf.keras.models.Model(inputs=enc_num.inputs, outputs=tagger_head)
    return tagger_model

############# THE CODE BELOW THIS LINE IS FOR DEBUGGING / UNSTABLE / EXPERIMENTAL ###########

def ulmfit_tagger_functional(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None):
    print("Building a regular LSTM model using only standard Keras blocks...")
    AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell2 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell3 = tf.keras.layers.LSTMCell(400, kernel_initializer='glorot_uniform')
    il = tf.keras.layers.Input((fixed_seq_len,), ragged=True if fixed_seq_len is None else False)
    l = tf.keras.layers.Masking(mask_value=1)(il)
    l = tf.keras.layers.Embedding(35000, 400)(l)
    l = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")(l)
    l = tf.keras.layers.Dropout(0.3)(l)
    l = tf.keras.layers.SpatialDropout1D(0.3)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(l)
    fake_model = tf.keras.models.Model(inputs=il, outputs=l)
    if pretrained_weights is not None:
        print("Restoring weights from file... (observe the warnings!)")
        fake_model.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!!")
    return fake_model
