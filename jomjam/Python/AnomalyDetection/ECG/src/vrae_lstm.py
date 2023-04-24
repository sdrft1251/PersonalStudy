import tensorflow as tf
from datetime import datetime
import pytz
from src import utils
import numpy as np

class VRAE_LSTM(tf.keras.Model):
    def __init__(self, time_size, enc_hidden_size, latent_length, dec_hidden_size, dropout, recurrent_dropout):
        super(VRAE_LSTM, self).__init__(name='')
        
        self.time_size = time_size

        # For Encoder
        self.encoder_lstm_cell = tf.keras.layers.LSTMCell(units=enc_hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Ecoding_LSTM_Cell")
        self.encoder_rnn_wrapper = tf.keras.layers.RNN(self.encoder_lstm_cell, return_state=True, name="Encoder_RNN_Wrapper")
        self.encode_mu_dense = tf.keras.layers.Dense(latent_length, name="Encoding_MU_Dense")
        self.encode_std_dense = tf.keras.layers.Dense(latent_length, name="Encoding_STD_Dense")
        
        # For Decoder
        self.decoder_first_state_dense = tf.keras.layers.Dense(dec_hidden_size, activation='tanh', name="Decoding_first_state_Dense")
        self.decoder_lstm_cell = tf.keras.layers.LSTMCell(units=dec_hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Decoding_LSTM_Cell")
        self.decode_mu_dense = tf.keras.layers.Dense(1, name="Decoding_MU_Dense")
        self.decode_std_dense = tf.keras.layers.Dense(1, name="Decoding_STD_Dense")


    @tf.function
    def corruption(self, input_tensor):
        return input_tensor + tf.random.normal(tf.shape(input_tensor), 0, 1, dtype=tf.float32)

    @tf.function
    def encoder(self, input_tensor):
        # Get State from Bi-Directional LSTM
        _, h_state, _ = self.encoder_rnn_wrapper(input_tensor)
        # Get Mean & STD
        mu = self.encode_mu_dense(h_state)
        std = self.encode_std_dense(h_state)
        std = tf.math.softplus(std, name="Encoder_std_softplus")
        return mu, std

    @tf.function
    def latent(self, mu, std, reparam=True):
        if reparam:
            return mu + std * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        else:
            return mu

    @tf.function
    def decoder(self, input_tensor):
        mu_collect = []
        sigma_collect = []
        # Get First State
        h_state = self.decoder_first_state_dense(input_tensor)
        c_state = tf.zeros(tf.shape(h_state), dtype=tf.float32)
        for t in range(self.time_size):
            # Next Cell
            _, states = self.decoder_lstm_cell(inputs=input_tensor, states=(h_state, c_state))
            h_state = states[0]
            c_state = states[1]
            # Get Output
            mu_out = self.decode_mu_dense(h_state)
            sigma_out = self.decode_std_dense(h_state)
            # Collect
            mu_collect.append(mu_out)
            sigma_collect.append(sigma_out)
        # Stack
        mu_collect = tf.stack(mu_collect)
        mu_collect = tf.transpose(mu_collect, [1, 0, 2])
        sigma_collect = tf.stack(sigma_collect)
        sigma_collect = tf.transpose(sigma_collect, [1, 0, 2])
        sigma_collect = tf.math.softplus(sigma_collect, name="Decoder_std_softplus")
        return mu_collect, sigma_collect

    @tf.function
    def call(self, input_tensor, reparam=False):
        # Encdoer
        mu_enc, sigma_enc = self.encoder(input_tensor)
        # Latent Space
        z = self.latent(mu_enc, sigma_enc, reparam=reparam)
        # Decoder
        mu_dec, sigma_dec = self.decoder(z)
        return mu_dec, sigma_dec, mu_enc, sigma_enc


def train(model, train_set, time_size, batch_size, beta_cycle, beta_rate, reparam, epochs,\
learning_rate, summary_dir, add_name, cp_dir):
    train_loss_results = []
    train_metric_results = []
    # Set Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # For File Save Name
    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)

    # Train Loop
    for ep_ in range(epochs):
        epoch_elbo_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        epoch_kld_avg = tf.keras.metrics.Mean()
        # Data Resampling
        # train_set = np.load(numpy_dir)
        # train_set = train_set[np.random.permutation(len(train_set))[:10000]]
        train_dataset = tensorset(arr=train_set, shape=(-1, time_size, 1), batch_size=batch_size)
        # Cal Beta
        beta = cal_beta_basic(ep_, beta_cycle) * beta_rate
        # In Batch
        for x in train_dataset:
            # Get Grad
            elbo, reconstruct_er, kld, mse, grads = grad(model, x, beta, reparam)
            # Apply Gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # For Monitoring
            epoch_elbo_avg(elbo)
            epoch_reconstruct_avg(reconstruct_er)
            epoch_kld_avg(kld)
            epoch_mse_avg(mse)
        train_loss_results.append(epoch_elbo_avg.result())
        train_metric_results.append(epoch_mse_avg.result())
        # Printing Model result
        if ep_ % 1 == 0:
            print("EPOCH : {:03d} | ELBO : {:.3f} | Reconstruct : {:.3f} | KLD : {:.3f} | MSE : {:.6f} | Beta : {} | TrainSet Size : {}".format(\
            ep_, epoch_elbo_avg.result(), epoch_reconstruct_avg.result(), epoch_kld_avg.result(), epoch_mse_avg.result(), beta, train_set.shape))
        # Save Model
        if len(cp_dir) != 0:
            if ep_ % 3 == 0:
                model.save_weights(cp_dir+"/"+log_file_name+"/save")
        if len(summary_dir) != 0 :
            with writer.as_default():
                tf.summary.scalar("ELBO Loss", epoch_elbo_avg.result(), step=ep_)
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                tf.summary.scalar("KLD Loss", epoch_kld_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)
            writer.flush()
    return train_loss_results

# Gradient
def grad(model, inputs, beta, reparam=True):
    with tf.GradientTape() as tape:
        elbo, reconstruct_er, kld, mse = elbo_loss(model, inputs, beta, reparam)
    return elbo, reconstruct_er, kld, mse, tape.gradient(elbo, model.trainable_variables)

# Cal ELBO Loss
def elbo_loss(model, inputs, beta, reparam=True):
    # From model
    mu_dec, sigma_dec, mu_enc, sigma_enc = model(inputs, reparam)
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    mu_dec = tf.squeeze(mu_dec, axis=-1)
    sigma_dec = tf.squeeze(sigma_dec, axis=-1)
    # Resampling controll
    if not reparam:
        sigma_enc = tf.ones(tf.shape(mu_enc))
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(sigma_enc) - tf.math.log(1e-8 + tf.math.square(sigma_enc)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(0.5*tf.math.log(tf.math.square(sigma_dec))+0.5*tf.math.square(inputs-mu_dec)/tf.math.square(sigma_dec), 1)
    marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    ELBO = marginal_likelihood - (beta*KL_divergence)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(mu_dec-inputs))
    return -ELBO, -marginal_likelihood, KL_divergence, MSE

# Return Tensor dataset
def tensorset(arr, shape, batch_size, drop_remainder=False):
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