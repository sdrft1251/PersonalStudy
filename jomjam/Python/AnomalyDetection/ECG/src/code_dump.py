import tensorflow as tf

class VRAE(tf.keras.Model):

    def __init__(self,\
                encoder_dims,\
                decoder_dims,\
                z_dims,\
                batch_size,\
                time_size,\
                time_feature_dims):
        super(VRAE, self).__init__(name='')
        self.z_dims = z_dims

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.z_dims = z_dims

        self.batch_size = batch_size
        self.time_size = time_size
        self.time_feature_dims = time_feature_dims

        # 값 초기화
        with tf.init_scope():
            w_enc_init = tf.keras.initializers.VarianceScaling()
            b_enc_init = tf.constant_initializer(0.)
            ##### Encoder
            # For RNN
            self.w_enc_h = tf.Variable(w_enc_init(shape=(self.encoder_dims, self.encoder_dims), dtype=tf.float32), trainable=True)
            self.w_enc_i = tf.Variable(w_enc_init(shape=(self.encoder_dims, self.time_feature_dims), dtype=tf.float32), trainable=True)
            self.b_enc = tf.Variable(b_enc_init(shape=(self.encoder_dims, 1), dtype=tf.float32), trainable=True)
            # For Dense
            self.w_enc_dense = tf.Variable(w_enc_init(shape=(self.z_dims*2, self.encoder_dims), dtype=tf.float32), trainable=True)
            self.b_enc_dense = tf.Variable(b_enc_init(shape=(self.z_dims*2, 1), dtype=tf.float32), trainable=True)
            ##### Decoder
            # For In Dense
            self.w_dec_in_dense = tf.Variable(w_enc_init(shape=(self.decoder_dims, self.z_dims), dtype=tf.float32), trainable=True)
            self.b_dec_in_dense = tf.Variable(b_enc_init(shape=(self.decoder_dims, 1), dtype=tf.float32), trainable=True)
            # For RNN
            self.w_dec_h = tf.Variable(w_enc_init(shape=(self.decoder_dims, self.decoder_dims), dtype=tf.float32), trainable=True)
            self.w_dec_i = tf.Variable(w_enc_init(shape=(self.decoder_dims, self.time_feature_dims), dtype=tf.float32), trainable=True)
            self.b_dec = tf.Variable(b_enc_init(shape=(self.decoder_dims, 1), dtype=tf.float32), trainable=True)
            # For Out Dense
            self.w_dec_dense = tf.Variable(w_enc_init(shape=(self.time_feature_dims, self.decoder_dims), dtype=tf.float32), trainable=True)
            self.b_dec_dense = tf.Variable(b_enc_init(shape=(self.time_feature_dims, 1), dtype=tf.float32), trainable=True)

            
    @tf.function
    def encoder(self, input_tensor):
        x_in = self.time_size * [None]
        h_enc = self.time_size * [None]

        # Make input_shape
        for t in range(self.time_size):
            #x_in[t] = tf.squeeze(tf.slice(input_tensor,begin=[0,t,0],size=[-1,1,-1]),axis=2)
            x_in[t] = tf.slice(input_tensor,begin=[0,t,0],size=[-1,1,-1])

        # init first state
        h_enc[0] = tf.zeros([self.batch_size, self.encoder_dims, 1], dtype=tf.float32)
        # Recurrent
        for t in range(self.time_size-1):
            # Reshaping
            w_enc_h_reshaped = tf.repeat(tf.expand_dims(self.w_enc_h, axis=0), repeats=self.batch_size, axis=0)
            w_enc_i_reshaped = tf.repeat(tf.expand_dims(self.w_enc_i, axis=0), repeats=self.batch_size, axis=0)
            b_enc_reshaped = tf.repeat(tf.expand_dims(self.b_enc, axis=0), repeats=self.batch_size, axis=0)
            # Get Next State
            h_enc[t+1] = tf.math.tanh(tf.matmul(w_enc_h_reshaped, h_enc[t]) + tf.matmul(w_enc_i_reshaped, x_in[t+1]) + b_enc_reshaped)

        # Reshaping
        w_enc_dense_reshaped = tf.repeat(tf.expand_dims(self.w_enc_dense, axis=0), repeats=self.batch_size, axis=0)
        b_enc_dense_reshaped = tf.repeat(tf.expand_dims(self.b_enc_dense, axis=0), repeats=self.batch_size, axis=0)
        # For Latent Space
        gaussian_enc = tf.squeeze(tf.matmul(w_enc_dense_reshaped, h_enc[-1]) + b_enc_dense_reshaped, axis=-1)
        mu_enc = gaussian_enc[:, :self.z_dims]
        stddev_enc = gaussian_enc[:, self.z_dims:]
        stddev_enc = 1e-6 + tf.math.softplus(stddev_enc)
        return mu_enc, stddev_enc

    @tf.function
    def decoder(self, z_sample):
        h_dec = (self.time_size + 1) * [None]
        x_out = self.time_size * [None]
        # init first state
        w_dec_in_dense_reshaped = tf.repeat(tf.expand_dims(self.w_dec_in_dense, axis=0), repeats=self.batch_size, axis=0)
        b_dec_in_dense_reshaped = tf.repeat(tf.expand_dims(self.b_dec_in_dense, axis=0), repeats=self.batch_size, axis=0)
        z_sample_reshaped = tf.expand_dims(z_sample, axis=-1)
        h_dec[0] = tf.math.tanh(tf.matmul(w_dec_in_dense_reshaped, z_sample_reshaped) + b_dec_in_dense_reshaped)
        # Recurrent
        for t in range(self.time_size):
            # Reshaping
            w_dec_dense_reshape = tf.repeat(tf.expand_dims(self.w_dec_dense, axis=0), repeats=self.batch_size, axis=0)
            b_dec_dense_reshape = tf.repeat(tf.expand_dims(self.b_dec_dense, axis=0), repeats=self.batch_size, axis=0)
            # Output val
            x_out[t] = tf.math.sigmoid(tf.matmul(w_dec_dense_reshape, h_dec[t]) + b_dec_dense_reshape)
            if t < self.time_size-1:
                # Reshaping
                w_dec_h_reshaped = tf.repeat(tf.expand_dims(self.w_dec_h, axis=0), repeats=self.batch_size, axis=0)
                w_dec_i_reshaped = tf.repeat(tf.expand_dims(self.w_dec_i, axis=0), repeats=self.batch_size, axis=0)
                b_dec_reshaped = tf.repeat(tf.expand_dims(self.b_dec, axis=0), repeats=self.batch_size, axis=0)
                # next state
                h_dec[t+1] = tf.math.tanh(tf.matmul(w_dec_h_reshaped, h_dec[t]) + tf.matmul(w_dec_i_reshaped, x_out[t]) + b_dec_reshaped)
        x_out = tf.concat(x_out, axis=1)
        x_out = tf.clip_by_value(x_out, 1e-8, 1 - 1e-8)
        return x_out

    @tf.function
    def call(self, input_tensor):
        # From Encoder
        mu_enc, stddev_enc = self.encoder(input_tensor)

        # Re-parameterization
        z = mu_enc + stddev_enc * tf.random.normal(tf.shape(mu_enc), 0, 1, dtype=tf.float32)

        # From Decoder
        y_hat = self.decoder(z)

        return y_hat, mu_enc, stddev_enc





############################################# Use LSTM ########################################
class VRAE(tf.keras.Model):

    def __init__(self,\
                hidden_size,\
                latent_length,\
                output_depth,\
                time_size,\
                ):
        super(VRAE, self).__init__(name='')

        self.time_size = time_size
        self.hidden_size = hidden_size

        # For Ecoder
        self.encode_lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
        self.encode_lstm_rnn = tf.keras.layers.RNN(self.encode_lstm_cell, return_state=True)
        self.encode_mu_dense = tf.keras.layers.Dense(latent_length)
        self.encode_std_dense = tf.keras.layers.Dense(latent_length)

        # For Decoder
        self.decoder_first_state_dense = tf.keras.layers.Dense(hidden_size*2, activation='tanh')
        self.decoder_lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
        self.decoder_output_dense = tf.keras.layers.Dense(output_depth, activation='sigmoid')
            
    @tf.function
    def encoder(self, input_tensor):
        
        # encode layer (Simple lstm)
        x, last_states, carry_states = self.encode_lstm_rnn(input_tensor)
        # get mean & std
        mu_enc = self.encode_mu_dense(last_states)
        stddev_enc = self.encode_std_dense(last_states)
        # Always Positive
        stddev_enc = 1e-6 + tf.math.softplus(stddev_enc)
        return mu_enc, stddev_enc

    @tf.function
    def decoder(self, z_sample):
        out_collect = []
        # Get First State
        h_state = self.decoder_first_state_dense(z_sample)
        c_state = h_state[:, self.hidden_size:]
        h_state = h_state[:, :self.hidden_size]
        # Loop Start
        for t in range(self.time_size):
            # Get output
            x_out = self.decoder_output_dense(h_state)
            # Next Cell
            lstm_out, state_dum = self.decoder_lstm_cell(inputs=x_out, states=(h_state, c_state))
            h_state = state_dum[0]
            c_state = state_dum[1]
            # Collect
            out_collect.append(x_out)
        # Stack
        out_collect = tf.stack(out_collect)
        out_collect = tf.transpose(out_collect, [1, 0, 2])

        # Clip both side
        out_collect = tf.clip_by_value(out_collect, 1e-8, 1 - 1e-8)
        return out_collect

    @tf.function
    def call(self, input_tensor):
        # From Encoder
        mu_enc, stddev_enc = self.encoder(input_tensor)

        # Re-parameterization
        z = mu_enc + stddev_enc * tf.random.normal(tf.shape(mu_enc), 0, 1, dtype=tf.float32)

        # From Decoder
        y_hat = self.decoder(z)

        return y_hat, mu_enc, stddev_enc