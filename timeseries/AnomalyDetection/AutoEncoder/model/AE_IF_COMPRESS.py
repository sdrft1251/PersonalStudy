import tensorflow as tf
import numpy as np
import math

############################################################################################################################################################
################# Utils
############################################################################################################################################################
class EncoderCompress(tf.keras.layers.Layer):
    def __init__(self, name="EncoderCompress"):
        super(EncoderCompress, self).__init__(name=name)

    def call(self, x):
        outputs = tf.math.reduce_mean(x, axis=1)
        return outputs

############################################################################################################################################################
################# Embedding
############################################################################################################################################################
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        return self.pe[:, :inputs.shape[1], :]

class PositionalEmbeddingDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbeddingDecoder, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        outputs = self.pe[:, :inputs.shape[1], :] + inputs - inputs
        return outputs

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model,
                                    kernel_size=3, padding='causal', activation='linear')
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.tokenConv(inputs)
        x = self.activation(x)
        return x

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, d_model, name="time2vec"):
        super(Time2Vec, self).__init__(name=name)
        self.w0 = tf.keras.layers.Dense(1)
        self.wi = tf.keras.layers.Dense(d_model-1)

    def call(self, x):
        v0 = self.w0(x)
        v1 = self.wi(x)
        v1 = tf.math.sign(v1)
        return tf.concat([v0, v1], axis=-1)

class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.cycle_embdding = Time2Vec(d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, **kwargs):
        x = self.value_embedding(x) + self.position_embedding(x) + self.cycle_embdding(x)
        return self.dropout(x)

############################################################################################################################################################
################# Attention
############################################################################################################################################################
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        # 정확히 분배되는지 확인
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)
        return outputs

def scaled_dot_product_attention(query, key, value):
    # Get Q * K
    matmul_qk = tf.linalg.matmul(query, key, transpose_b=True)
    # For Scaling
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    # Get Attention Weights
    attention_weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.linalg.matmul(attention_weights, value)
    return outputs, attention_weights

############################################################################################################################################################
################# Encoder
############################################################################################################################################################
def encoder_layer(time_len, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    # Self-Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs})
    # Add & Norm
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # FFNN
    outputs = tf.keras.layers.Conv1D(filters=d_model*4, kernel_size=1)(attention)
    outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def compress_layer(time_len, d_model, dropout, name="compress_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    outputs = tf.keras.layers.Conv1D(filters=int(d_model/2), kernel_size=3, padding='causal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ELU()(outputs)
    outputs = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def encoder(time_len, num_layers, d_model, num_heads, dropout, name="encoder"):
    # Model Part
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")
    # Time2Vec
    embeddings = DataEmbedding(d_model=d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    # Encoder layers
    for i in range(num_layers-1):
        outputs = encoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))(outputs)
        outputs = compress_layer(time_len=time_len, d_model=d_model, dropout=dropout, name="compress_layer_{}".format(i))(outputs)
        d_model = int(d_model/2)
        time_len = int(time_len/2)
    outputs = encoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(num_layers-1))(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

############################################################################################################################################################
################# Decoder
############################################################################################################################################################
def decoder_layer(time_len, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(d_model), name="encoder_outputs")
    # Self-Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention_1")({'query': inputs, 'key': inputs, 'value': inputs})
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # Attention with Encoder
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")({'query': attention, 'key': enc_outputs, 'value': enc_outputs})
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention)
    # FFNN
    outputs = tf.keras.layers.Dense(units=int(d_model/2), activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + outputs)
    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)

def decoder(time_len, num_layers, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")
    enc_outputs = tf.keras.Input(shape=(d_model), name="decoder_outputs")
    dec_input = PositionalEmbeddingDecoder(d_model=d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(dec_input)
    # Decoder layers
    for i in range(num_layers):
        outputs = decoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs])
    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)

############################################################################################################################################################
################# Total Model
############################################################################################################################################################
def IFEMBD(time_len, d_model, enc_layer_num, num_heads, dec_layer_num, dropout, name="IFEMBD"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")

    # Encoder Part
    enc_outputs = encoder(time_len=time_len, num_layers=enc_layer_num, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=inputs)

    # Compress Layer
    enc_outputs = EncoderCompress()(enc_outputs)

    # Decoder Part
    dec_outputs = decoder(time_len=time_len, num_layers=dec_layer_num, d_model=enc_outputs.shape[-1], num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_outputs])
    
    # For Output
    outputs = tf.keras.layers.Dense(1, name="OutPut_Dense")(dec_outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)