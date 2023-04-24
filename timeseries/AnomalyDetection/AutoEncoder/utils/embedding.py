import tensorflow as tf
import math

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