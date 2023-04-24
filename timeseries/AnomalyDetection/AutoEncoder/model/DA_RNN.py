import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from datetime import datetime
import pytz
import numpy as np

###### Dual-Attention ######

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

class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = tf.keras.layers.LSTM(m, return_state=True)
        self.initial_state = None

    def call(self, x, training=True):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,hidden_size)
        c_s : cell_state (shape = batch,hidden_size)
        x : time series encoder inputs (shape = batch,T,n)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, hidden_size*2
        query = RepeatVector(x.shape[2])(query)  # batch, n, hidden_size*2
        x_perm = Permute((2, 1))(x)  # batch, n, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T
        score = self.v(score)  # batch, n, 1
        score = Permute((2, 1))(score)  # batch,1,n
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights


class Encoder(Layer):
    def __init__(self, T, hidden_size):
        super(Encoder, self).__init__(name="encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(hidden_size)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, n, training=True):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        self.lstm.reset_state(h0=h0, c0=c0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  # (batch,1,n)
            h_s, c_s = self.lstm(x) # (batch, hidden_size)
            self.alpha_t = self.input_att(h_s, c_s, data)  # batch,1,n
            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n
        output = tf.multiply(data, alpha_seq)  # batch, T, n

        return output

class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = tf.keras.layers.LSTM(p, return_state=True)
        self.initial_state = None

    def call(self, x, training=True):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class TemporalAttention(Layer):
    def __init__(self, latent_dims):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(latent_dims)
        self.w2 = Dense(latent_dims)
        self.v = Dense(1)

    def call(self, h_s, c_s, latent):
        """
        h_s : hidden_state (shape = batch, hidden_dims)
        c_s : cell_state (shape = batch, hidden_dims)
        latent : time series encoder inputs (shape = batch, latent_dims)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, hidden_dims*2
        score = tf.nn.tanh(self.w1(latent) + self.w2(query))  # batch, latent_dims
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # Latent Space 안에의 중요성 # batch, latent_dims
        return attention_weights

class Decoder(Layer):
    def __init__(self, hidden_size, T, latent_dims):
        super(Decoder, self).__init__(name="decoder")
        self.T = T
        self.temp_att = TemporalAttention(latent_dims)
        self.lstm = Decoderlstm(hidden_size)
        self.decoder_output_mu_dense = tf.keras.layers.Dense(1)
        self.reinput = tf.keras.layers.Dense(1)
        self.context_v = None
        self.beta_t = None

    def call(self, latent, h0=None, c0=None, training=True):
        """
        latent : Latent Space state (shape = batch, latent_dims)
        """
        out_collect = []
        self.lstm.reset_state(h0=h0, c0=c0)
        self.beta_t = self.temp_att(h0, c0, latent)  # batch, latent_dims
        self.context_v = tf.math.multiply(self.beta_t, latent)  # batch, latent_dims
        for t in range(self.T):
            # Concat Attention values
            x = tf.concat([latent, self.context_v], axis=-1) # batch, latent_dims*2
            # Get states
            h_s, c_s = self.lstm(x)
            # Attention
            self.beta_t = self.temp_att(h_s, c_s, latent)
            self.context_v = tf.math.multiply(self.beta_t, latent)
            # Output Collect
            out = self.decoder_output_mu_dense(h_s)
            out_collect.append(out)
        # Stack
        out_collect = tf.stack(out_collect)
        out_collect = tf.transpose(out_collect, [1, 0, 2])
        return out_collect

def encoder(T, d_model, hidden_size, h0=None, c0=None, name="encoder"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    embeddings = Time2Vec(output_dims=d_model)(inputs) # batch, T, d_model
    outputs = Encoder(T,hidden_size)(embeddings,h0,c0,d_model)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def decoder(latent_dims, hidden_size, T, h0=None, c0=None, name="decoder"):
    inputs = tf.keras.Input(shape=(latent_dims), name="inputs")
    outputs = Decoder(hidden_size, T, latent_dims)(inputs,h0,c0)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def DARNN(T, d_model, batch_size, hidden_size, latent_dims, name="DARNN"):
    # Init
    latent_lstm = tf.keras.layers.LSTM(latent_dims, return_state=True)
    h0 = tf.zeros((batch_size, hidden_size))
    c0 = tf.zeros((batch_size, hidden_size))
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    enc_output = encoder(T=T, d_model=d_model, hidden_size=hidden_size, h0=h0, c0=c0)(inputs) # batch, T, d_model
    _, latent = latent_lstm(enc_output) # batch, latent_dims
    dec_output = decoder(latent_dims=latent_dims, hidden_size=hidden_size, T=T, h0=h0, c0=c0)(latent) # batch, T, 1
    return tf.keras.Model(inputs=inputs, outputs=[dec_output, latent], name=name)