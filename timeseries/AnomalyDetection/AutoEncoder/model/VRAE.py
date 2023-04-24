import tensorflow as tf

class VRAE(tf.keras.Model):

    def __init__(self,\
                hidden_size,\
                latent_length,\
                output_depth,\
                time_size,\
                dropout=0.0,\
                recurrent_dropout=0.0
                ):
        super(VRAE, self).__init__(name='')

        self.time_size = time_size
        self.hidden_size = hidden_size

        # For Ecoder
        self.encode_cell = tf.keras.layers.SimpleRNNCell(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Ecoding_RNN_Cell")
        self.encode_rnn = tf.keras.layers.RNN(self.encode_cell, return_state=True, name="RNN_Wrapper")
        self.encode_mu_dense = tf.keras.layers.Dense(latent_length, name="Encoding_MU_Dense")
        self.encode_std_dense = tf.keras.layers.Dense(latent_length, name="Encoding_STD_Dense")

        # For Decoder
        self.decoder_first_state_dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name="Decoding_first_state_Dense")
        self.decode_cell = tf.keras.layers.SimpleRNNCell(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Decoding_RNN_Cell")
        self.decoder_output_mu_dense = tf.keras.layers.Dense(output_depth, activation="sigmoid", name="Decoding_mu_Output")
            
    @tf.function
    def encoder(self, input_tensor):
        # encode layer (Simple lstm)
        x, last_states = self.encode_rnn(input_tensor)
        # get mean & std
        mu_enc = self.encode_mu_dense(last_states)
        stddev_enc = self.encode_std_dense(last_states)
        # Always Positive
        stddev_enc = tf.math.exp(stddev_enc)
        return mu_enc, stddev_enc

    @tf.function
    def decoder(self, z_sample):
        out_collect = []
        # Get First State
        h_state = self.decoder_first_state_dense(z_sample)
        # Loop Start
        for t in range(self.time_size):
            # Get output
            x_out = self.decoder_output_mu_dense(h_state)
            # Next Cell
            rnn_out, h_state = self.decode_cell(inputs=x_out, states=h_state)
            # Collect
            out_collect.append(x_out)
        # Stack
        out_collect = tf.stack(out_collect)
        out_collect = tf.transpose(out_collect, [1, 0, 2])

        return out_collect

    @tf.function
    def latent(self, mu, std, reparam=True):
        if reparam:
            return mu + std * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        else:
            return mu

    @tf.function
    def call(self, input_tensor, reparam=True):
        # From Encoder
        mu_enc, stddev_enc = self.encoder(input_tensor)

        # From Latent
        z = self.latent(mu_enc, stddev_enc, reparam=reparam)

        # From Decoder
        y_mu = self.decoder(z)
        
        return y_mu, mu_enc, stddev_enc