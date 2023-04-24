import tensorflow as tf
from datetime import datetime
import pytz
from src import utils
import numpy as np

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

def elbo_loss(model, inputs, beta, reparam=True):
    # From model
    y, mu_enc, stddev_enc = model(inputs, reparam)
    # Resampling constroll
    if not reparam:
        stddev_enc = tf.ones(tf.shape(mu_enc))
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(stddev_enc) - tf.math.log(1e-8 + tf.math.square(stddev_enc)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(y) + (1 - inputs) * tf.math.log(1 - y), 1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    ELBO = marginal_likelihood - (beta*KL_divergence)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(y-inputs))
    return -ELBO, -marginal_likelihood, KL_divergence, MSE

def grad(model, inputs, beta, reparam=True):
    with tf.GradientTape() as tape:
        elbo, reconstruct_er, kld, mse = elbo_loss(model, inputs, beta, reparam)
    return elbo, reconstruct_er, kld, mse, tape.gradient(elbo, model.trainable_variables)

def cal_beta_basic(ep_, cycle):
    while ep_>cycle:
        ep_ -= cycle
    beta = ep_*2 / cycle
    if beta >= 1:
        beta = 1
    return beta

def cal_beta_num1(ep_, cycle):
    while ep_>cycle:
        ep_ -= cycle
    beta = ep_*3 / cycle
    if beta <= 1:
        beta = 0
    elif beta <=2:
        beta -= 1
    else:
        beta = 1
    return beta

def cal_beta_num2(ep_, cycle):
    while ep_>cycle:
        ep_ -= cycle
    beta = ep_*2 / cycle
    if beta <= 1:
        beta = 0
    else:
        beta -= 1
    return beta

def train(model, data_col, time_size, over_len, batch_size, beta_cycle, beta_rate, reparam,\
epochs, learning_rate=0.001, summary_dir="/logs", add_name="", cp_dir="/save"):
    train_loss_results = []
    train_metric_results = []
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.05, beta_2=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)
    for ep_ in range(epochs):
        epoch_elbo_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        epoch_kld_avg = tf.keras.metrics.Mean()

        # Data Resampling
        train_dataset = tensorset(arr=data_col, shape=(-1, time_size, 1), batch_size=batch_size)

        # Cal Beta
        beta = cal_beta_basic(ep_, beta_cycle) * beta_rate

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
        if ep_ % 3 == 0:
            print("EPOCH : {:03d} | ELBO : {:.3f} | Reconstruct : {:.3f} | KLD : {:.3f} | MSE : {:.6f}".format(\
            ep_, epoch_elbo_avg.result(), epoch_reconstruct_avg.result(), epoch_kld_avg.result(), epoch_mse_avg.result()))
            print(beta)
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

# Return Tensor dataset
def tensorset(arr, shape, batch_size, drop_remainder=False):
    # type casting & reshaping
    data = arr.astype(np.float32)
    data = np.reshape(data, shape)
    # make to tensor
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=data.shape[0]*3)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds
