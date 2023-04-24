import tensorflow as tf
from datetime import datetime
import pytz

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
        self.decoder_output_mu_dense = tf.keras.layers.Dense(output_depth, name="Decoding_mu_Output")
            
    @tf.function
    def encoder(self, input_tensor):
        # encode layer (Simple lstm)
        x, last_states = self.encode_rnn(input_tensor)
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
    def call(self, input_tensor):
        # From Encoder
        mu_enc, stddev_enc = self.encoder(input_tensor)

        # Re-parameterization
        z = mu_enc + stddev_enc * tf.random.normal(tf.shape(mu_enc), 0, 1, dtype=tf.float32)

        # From Decoder
        y_mu = self.decoder(z)
        
        return y_mu, mu_enc, stddev_enc


def elbo_loss(model, inputs, alpha=1):
    # From model
    y_mu, mu_enc, stddev_enc = model(inputs)
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(stddev_enc) - tf.math.log(1e-8 + tf.math.square(stddev_enc)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(0.5*tf.math.square(inputs-y_mu), 1)
    marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    ELBO = alpha*marginal_likelihood - (1-alpha)*KL_divergence
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(y_mu-inputs))
    return -ELBO, -marginal_likelihood, KL_divergence, MSE

def grad(model, inputs, alpha=1):
    with tf.GradientTape() as tape:
        elbo, reconstruct_er, kld, mse = elbo_loss(model, inputs, alpha=alpha)
    # Alpha Scheduling
    return elbo, reconstruct_er, kld, mse, tape.gradient(elbo, model.trainable_variables)

def train(model, train_dataset, epochs, learning_rate=0.001, summary_dir="/logs", add_name="", cp_dir="/save"):
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

        for x in train_dataset:
            # KLD Scheduling
            alpha = 1 - (ep_/epochs)*0.5
            elbo, reconstruct_er, kld, mse, grads = grad(model, x, alpha=alpha)
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
        if ep_ % 5 == 0:
            print("EPOCH : {:03d} | ELBO : {:.3f} | Reconstruct : {:.3f} | KLD : {:.3f} | MSE : {:.6f}".format(\
            ep_, epoch_elbo_avg.result(), epoch_reconstruct_avg.result(), epoch_kld_avg.result(), epoch_mse_avg.result()))
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
