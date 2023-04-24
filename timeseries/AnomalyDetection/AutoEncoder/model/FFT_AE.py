import tensorflow as tf

class FFTtransform(tf.keras.layers.Layer):
    def __init__(self, frame_length, frame_step, fft_length):
        """
        ...
        """
        super(FFTtransform, self).__init__(name="FFTtransform")
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, x):
        """
        x : Time series data (shape = batch,T)
        """
        transformed = tf.signal.stft(x, self.frame_length, self.frame_step, self.fft_length)
        return tf.abs(transformed)

class SignalGenerator(tf.keras.layers.Layer):
    def __init__(self, frame_length, frame_step, fft_length, T):
        super(SignalGenerator, self).__init__(name="SignalGenerator")
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.T = T

    def call(self, x):
        outputs = tf.signal.inverse_stft(x, self.frame_length, self.frame_step, self.fft_length)
        outputs = outputs[:,self.frame_length:self.frame_length+self.T]
        return outputs

class DecoderGRU(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        """
        p : feature dimension
        h0 : initial hidden state
        """
        super(DecoderGRU, self).__init__(name="Decoder_GRU")
        self.grucell = tf.keras.layers.GRUCell(hidden_size)
        self.initial_state = None

    def call(self, x):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        _, h_s = self.grucell(x, states=self.initial_state)
        self.initial_state = h_s
        return h_s

    def reset_state(self, h0):
        self.initial_state = h0

class Decoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, T):
        super(Decoder, self).__init__(name="Decoder")
        self.T = T
        self.gru = DecoderGRU(hidden_size)
        self.decoder_first_state_dense = tf.keras.layers.Dense(hidden_size, activation='tanh')

    def call(self, latent):
        """
        latent : Latent Space state (shape = batch, latent_dims)
        """
        h_s_collect = []
        h_s = self.decoder_first_state_dense(latent)
        self.gru.reset_state(h0=h_s)
        for t in range(self.T):
            # Input to RNN
            h_s = self.gru(latent)
            h_s_collect.append(h_s)
        # Stack
        h_s_collect = tf.stack(h_s_collect)
        h_s_collect = tf.transpose(h_s_collect, [1, 0, 2])
        return h_s_collect

def Encoder_Module(T, frame_length, frame_step, fft_length, hidden_size, name="Encoder_Module"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    reshape_time = tf.squeeze(inputs, axis=-1)
    transformed = FFTtransform(frame_length, frame_step, fft_length)(reshape_time)
    _, h_state = tf.keras.layers.RNN(tf.keras.layers.GRUCell(hidden_size), return_state=True)(transformed)
    return tf.keras.Model(inputs=inputs, outputs=h_state, name=name)

def Decoder_Module(latent_dims, hidden_size, frame_length, frame_step, fft_length, T, name="Decoder_Module"):
    inputs = tf.keras.Input(shape=(latent_dims,), name="inputs")
    rnn_out = Decoder(hidden_size, T+(frame_length*2))(inputs) # batch, T, hidden
    real_vec = tf.keras.layers.Dense(hidden_size)(rnn_out)
    imag_vec = tf.keras.layers.Dense(hidden_size)(rnn_out)
    complex_vec = tf.complex(real_vec, imag_vec)
    gen_out = SignalGenerator(frame_length, frame_step, fft_length, T)(complex_vec)
    outputs = tf.reshape(gen_out, (-1,T,1))
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def FFT_AE(T, frame_length, frame_step, fft_length, hidden_size, latent_dims, name="FFT_AE"):
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    enc_out = Encoder_Module(T, frame_length, frame_step, fft_length, hidden_size)(inputs)
    latent = tf.keras.layers.Dense(latent_dims)(enc_out)
    dec_output = Decoder_Module(latent_dims, fft_length//2+1, frame_length, frame_step, fft_length, T)(latent)
    return tf.keras.Model(inputs=inputs, outputs=[dec_output, latent], name=name)