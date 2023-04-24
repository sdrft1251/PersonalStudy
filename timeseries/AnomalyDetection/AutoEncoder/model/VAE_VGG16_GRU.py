import tensorflow as tf

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, output_dims, name="time2vec"):
        super(Time2Vec, self).__init__(name=name)
        self.output_dims = output_dims

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        # i=0
        self.w0 = tf.Variable(initial_value=w_init(shape=(input_shape[-1], 1),dtype=tf.float32), name="Time2Vec_w0", trainable=True)
        self.b0 = tf.Variable(initial_value=b_init(shape=(1),dtype=tf.float32), name="Time2Vec_b0", trainable=True)
        # i!=0
        self.wi = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.output_dims-1),dtype=tf.float32), name="Time2Vec_wi", trainable=True)
        self.bi = tf.Variable(initial_value=b_init(shape=(self.output_dims-1),dtype=tf.float32), name="Time2Vec_bi", trainable=True)

    def call(self, input_tensor):
        v0 = tf.linalg.matmul(input_tensor, self.w0) + self.b0
        v1 = tf.math.sign(tf.linalg.matmul(input_tensor, self.wi) + self.bi)
        return tf.concat([v0, v1], axis=-1)

class VGG16(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG16, self).__init__(name="VGG16")
        self.layer_1_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')
        self.layer_1_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')
        self.layer_1_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_2_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_2_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_2_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_3_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_4_1 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_2 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_3 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x):
        """
        x : input data (shape = batch,T,d)
        """
        x = self.layer_1_1(x)
        x = self.layer_1_2(x)
        x = self.layer_1_3(x)
        x = self.layer_2_1(x)
        x = self.layer_2_2(x)
        x = self.layer_2_3(x)
        x = self.layer_3_1(x)
        x = self.layer_3_2(x)
        x = self.layer_3_3(x)
        x = self.layer_3_4(x)
        x = self.layer_4_1(x)
        x = self.layer_4_2(x)
        x = self.layer_4_3(x)
        x = self.layer_4_4(x)
        return x

class VGG16_Reverse(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG16_Reverse, self).__init__(name="VGG16_Reverse")
        self.layer_1_1 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_1_2 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_1_3 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_1_4 = tf.keras.layers.UpSampling1D(size=2)
        self.layer_2_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_2_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_2_3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_2_4 = tf.keras.layers.UpSampling1D(size=2)
        self.layer_3_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_3_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_3_3 = tf.keras.layers.UpSampling1D(size=2)
        self.layer_4_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')
        self.layer_4_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')

    def call(self, x):
        """
        x : input data (shape = batch,latent_dims)
        """
        x = self.layer_1_1(x)
        x = self.layer_1_2(x)
        x = self.layer_1_3(x)
        x = self.layer_1_4(x)
        x = self.layer_2_1(x)
        x = self.layer_2_2(x)
        x = self.layer_2_3(x)
        x = self.layer_2_4(x)
        x = self.layer_3_1(x)
        x = self.layer_3_2(x)
        x = self.layer_3_3(x)
        x = self.layer_4_1(x)
        x = self.layer_4_2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout=0.0, recurrent_dropout=0.0, name="EncoderLayer"):
        super(EncoderLayer, self).__init__(name=name)
        self.gru_cell = tf.keras.layers.GRUCell(units=hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)
        self.rnn_wrapper = tf.keras.layers.RNN(self.gru_cell, return_sequences=True, return_state=True)

    def call(self, x):
        outputs, h_state = self.rnn_wrapper(x)
        return outputs, h_state

class GRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name="GRUCell"):
        super(GRUCell, self).__init__(name=name)
        self.grucell = tf.keras.layers.GRUCell(hidden_size)
        self.initial_state = None

    def call(self, x):
        _, state = self.grucell(x, states=self.initial_state)
        self.initial_state = state
        return state

    def reset_state(self, h0):
        self.initial_state = h0

class DecoderFirstLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, T, name="DecoderFirstLayer"):
        super(DecoderFirstLayer, self).__init__(name=name)
        self.T = T
        self.gru = GRUCell(hidden_size)
        self.first_state_dense = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.output_dense = tf.keras.layers.Dense(hidden_size, activation='relu')

    def call(self, x):
        out_collect = []
        h_s = self.first_state_dense(x)
        self.gru.reset_state(h0=h_s)
        for t in range(self.T):
            # Output collect
            out = self.output_dense(h_s)
            out_collect.append(out)
            # Input to RNN
            h_s = self.gru(out)
        # Last Output
        out = self.output_dense(h_s)
        out_collect.append(out)
        # Stack
        out_collect = tf.stack(out_collect[1:])
        out_collect = tf.transpose(out_collect, [1, 0, 2])
        return out_collect
        
class EncoderTransLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dims, name="EncoderTransLayer"):
        super(EncoderTransLayer, self).__init__(name=name)
        self.mu_dense = tf.keras.layers.Dense(latent_dims, name="MU_Dense")
        self.sigma_dense = tf.keras.layers.Dense(latent_dims, name="SIGMA_Dense")
    
    def call(self, x):
        mu = self.mu_dense(x)
        sigma = self.sigma_dense(x)
        sigma = tf.math.softplus(sigma, name="SIGMA_softplus")
        return mu, sigma

class LatentLayer(tf.keras.layers.Layer):
    def __init__(self, reparam=True, name="LatentLayer"):
        super(LatentLayer, self).__init__(name=name)
        self.reparam = reparam

    def call(self, mu, sigma):
        if self.reparam:
            return mu + sigma * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        else:
            return mu

####################################################################################
# Model Part
####################################################################################
def Encoder_Module(T, hidden_size, dropout, recurrent_dropout, name="Encoder_Module"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    t2v = Time2Vec(output_dims=256)(inputs)
    # Compress
    comp = VGG16()(t2v)# batch, compressT, d_model
    # Encoder
    _, outputs = EncoderLayer(hidden_size=hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)(comp)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def Decoder_Module(latent_dims, hidden_size, T, name="Decoder_Module"):
    inputs = tf.keras.Input(shape=(latent_dims,), name="inputs")
    outputs = DecoderFirstLayer(hidden_size=hidden_size, T=int(640/8))(inputs)
    outputs = VGG16_Reverse()(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def VAE_CNNGRU(T, hidden_size, latent_dims, dropout, recurrent_dropout, reparam=True, name="DARNN"):
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    enc_output = Encoder_Module(T=T, hidden_size=hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)(inputs)
    mu, sigma = EncoderTransLayer(latent_dims=latent_dims)(enc_output)
    latent = LatentLayer(reparam=reparam)(mu, sigma)
    dec_output = Decoder_Module(latent_dims=latent_dims, hidden_size=hidden_size, T=T)(latent)
    return tf.keras.Model(inputs=inputs, outputs=[dec_output, mu, sigma], name=name)