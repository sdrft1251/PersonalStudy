import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from datetime import datetime
import pytz
import numpy as np

class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = LSTM(m, return_state=True)
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
        h_s : hidden_state (shape = batch,hidden_dims)
        c_s : cell_state (shape = batch,hidden_dims)
        x : time series encoder inputs (shape = batch,T,1)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, hidden_dims*2
        query = RepeatVector(x.shape[2])(query)  # batch, 1, hidden_dims*2
        x_perm = Permute((2, 1))(x)  # batch, 1, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, 1, T
        score = tf.squeeze(score)  # batch, T
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 time step 별 중요도
        return attention_weights


class Encoder(Layer):
    def __init__(self, T, hidden_size):
        super(Encoder, self).__init__(name="encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(hidden_size)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, training=True):
        """
        data : encoder data (shape = batch, T, 1)
        n : data feature num
        """
        self.lstm.reset_state(h0=h0, c0=c0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  # (batch, 1, 1)
            h_s, c_s = self.lstm(x) # batch, 1, 1 -> batch, hidden_dims
            self.alpha_t = self.input_att(h_s, c_s, data)  # batch, T
            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, self.T))  # batch, T, T
        output = tf.linalg.matmul(alpha_seq, data)  # batch, T, 1
        return output

class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = LSTM(p, return_state=True)
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
        self.decoder_output_mu_dense = tf.keras.layers.Dense(1, activation="sigmoid")
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
            x = tf.concat([latent, self.context_v], axis=-1) # batch, latent_dims*2
            x = self.decoder_output_mu_dense(x) # batch, 1
            out_collect.append(x)
            h_s, c_s = self.lstm(x[:, tf.newaxis, :])  # batch, 1, 1 -> batch, hidden_dims
            self.beta_t = self.temp_att(h_s, c_s, latent)  # batch, latent_dims
            self.context_v = tf.math.multiply(self.beta_t, latent)  # batch, latent_dims
        x = tf.concat([latent, self.context_v], axis=-1) # batch, latent_dims*2
        x = self.decoder_output_mu_dense(x) # batch, 1
        out_collect.append(x)
        # Stack
        out_collect = tf.stack(out_collect[1:])
        out_collect = tf.transpose(out_collect, [1, 0, 2])
        return out_collect

def encoder(T, hidden_size, h0=None, c0=None, name="encoder"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    outputs = Encoder(T,hidden_size)(inputs,h0,c0)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def decoder(latent_dims, hidden_size, T, h0=None, c0=None, name="decoder"):
    inputs = tf.keras.Input(shape=(latent_dims), name="inputs")
    outputs = Decoder(hidden_size, T, latent_dims)(inputs,h0,c0)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def DARNN(T, batch_size, hidden_size, latent_dims, name="DARNN"):
    # Init
    latent_lstm = LSTM(latent_dims)
    h0 = tf.zeros((batch_size, hidden_size))
    c0 = tf.zeros((batch_size, hidden_size))
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    enc_output = encoder(T=T, hidden_size=hidden_size, h0=h0, c0=c0)(inputs) # batch, T, 1
    latent = latent_lstm(enc_output) # batch, latent_dims
    dec_output = decoder(latent_dims=latent_dims, hidden_size=hidden_size, T=T, h0=h0, c0=c0)(latent) # batch, T, 1
    return tf.keras.Model(inputs=inputs, outputs=[dec_output, latent], name=name)

############################################################################################################################
# Train Part
############################################################################################################################
# Cal ELBO Loss
def elbo_loss(model, inputs, beta):
    # From model
    mu_dec, mu_enc = model(inputs)
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    mu_dec = tf.squeeze(mu_dec, axis=-1)
    #sigma_dec = tf.squeeze(sigma_dec, axis=-1)
    # Latent loss: -KL[q(z|x)|p(z)]
    # KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(sigma_enc) - tf.math.log(1e-8 + tf.math.square(sigma_enc)) - 1, 1)
    # KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(tf.math.square(inputs-mu_dec), 1)
    marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Reconstruction Loss: log(p(x|z)) - 2
    # marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(mu_dec) + (1 - inputs) * tf.math.log(1 - mu_dec), 1)
    # marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # Reconstruction Loss: log(p(x|z)) - 3
    # marginal_likelihood = tf.reduce_sum(0.5*tf.math.log(tf.math.square(sigma_dec))+0.5*tf.math.square(inputs-mu_dec)/tf.math.square(sigma_dec), 1)
    # marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    # ELBO = 10*marginal_likelihood - (beta*KL_divergence)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(mu_dec-inputs))
    #print("ELBO : {} Marginal : {} KLD : {}".format(-ELBO.numpy(), -marginal_likelihood.numpy(), KL_divergence.numpy()))
    return -marginal_likelihood, MSE

# Gradient
def grad(model, inputs, beta, reparam=True):
    with tf.GradientTape() as tape:
        reconstruct_er, mse = elbo_loss(model, inputs, beta)
    return reconstruct_er, mse, tape.gradient(reconstruct_er, model.trainable_variables)

def train(model, train_set, epochs, batch_size, beta_cycle, beta_rate, learning_rate, summary_dir, add_name, cp_dir):
    train_loss_results = []
    train_metric_results = []
    # Set Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    # For File Save Name
    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)

    # Train Loop
    for ep_ in range(epochs):
        #epoch_elbo_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        #epoch_kld_avg = tf.keras.metrics.Mean()
        # Data Resampling
        train_dataset = tensorset(arr=train_set, shape=(-1, train_set.shape[1], 1), batch_size=batch_size)
        # Cal Beta
        beta = cal_beta_basic(ep_, beta_cycle) * beta_rate
        # In Batch
        for x in train_dataset:
            # Get Grad
            reconstruct_er, mse, grads = grad(model, x, beta)
            # Apply Gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # For Monitoring
            #epoch_elbo_avg(elbo)
            epoch_reconstruct_avg(reconstruct_er)
            #epoch_kld_avg(kld)
            epoch_mse_avg(mse)
        train_loss_results.append(epoch_reconstruct_avg.result())
        train_metric_results.append(epoch_mse_avg.result())
        
        # Printing Model result
        if ep_ % 1 == 0:
            print("EPOCH : {:05d} | Reconstruct : {:.6f} | MSE : {:.6f} | Beta : {} | TrainSet Size : {}".format(\
            ep_, epoch_reconstruct_avg.result(), epoch_mse_avg.result(), beta, train_set.shape))
        # Save Model
        if len(cp_dir) != 0:
            if ep_ % 3 == 0:
                model.save_weights(cp_dir+"/"+log_file_name+"/save")
        if len(summary_dir) != 0 :
            with writer.as_default():
                #tf.summary.scalar("ELBO Loss", epoch_elbo_avg.result(), step=ep_)
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                #tf.summary.scalar("KLD Loss", epoch_kld_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)
            writer.flush()
    return train_loss_results

# Return Tensor dataset
def tensorset(arr, shape, batch_size, drop_remainder=True):
    # type casting & reshaping
    data = arr.astype(np.float32)
    data = np.reshape(data, shape)
    # make to tensor
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=data.shape[0]*3)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds

# For KLD Rate
def cal_beta_basic(ep_, cycle):
    if cycle == 0:
        return 1
    while ep_>cycle:
        ep_ -= cycle
    beta = ep_*2 / cycle
    if beta >= 1:
        beta = 1
    return beta