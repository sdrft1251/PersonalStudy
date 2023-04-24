import tensorflow as tf
from datetime import datetime
import pytz
from src import utils
import numpy as np

class VAE(tf.keras.Model):

    def __init__(self, encoder_filters, latent_length, decoder_filters):
        super(VAE, self).__init__(name='')

        self.encoder_filters = encoder_filters
        self.latent_length = latent_length
        self.decoder_filters = decoder_filters

        # For Encoder
        self.enc_conv_1 = tf.keras.layers.Conv1D(encoder_filters[0], 3, name="Encoding_CNN_1")
        self.enc_lrelu_1 = tf.keras.layers.LeakyReLU(name="Encoding_LRelu_1")

        self.enc_conv_2 = tf.keras.layers.Conv1D(encoder_filters[1], 3, name="Encoding_CNN_2")
        self.enc_lnorm_2 = tf.keras.layers.LayerNormalization(name="Encoding_LNorm_2")
        self.enc_lrelu_2 = tf.keras.layers.LeakyReLU(name="Encoding_LRelu_2")

        self.enc_conv_3 = tf.keras.layers.Conv1D(encoder_filters[2], 3, name="Encoding_CNN_3")
        self.enc_lnorm_3 = tf.keras.layers.LayerNormalization(name="Encoding_LNorm_3")
        self.enc_lrelu_3 = tf.keras.layers.LeakyReLU(name="Encoding_LRelu_3")

        self.enc_conv_4 = tf.keras.layers.Conv1D(encoder_filters[3], 3, name="Encoding_CNN_4")
        self.enc_lnorm_4 = tf.keras.layers.LayerNormalization(name="Encoding_LNorm_4")
        self.enc_lrelu_4 = tf.keras.layers.LeakyReLU(name="Encoding_LRelu_4")

        self.enc_flatten = tf.keras.layers.Flatten(name="Encoding_Flatten")
        self.enc_mu = tf.keras.layers.Dense(latent_length, name="Encoding_MU_Dense")
        self.enc_std = tf.keras.layers.Dense(latent_length, name="Encoding_STD_Dense")

        # Decoder
        self.dec_dense_1 = tf.keras.layers.Dense(decoder_filters[0], name="Decoding_Reconstruct_1")
        self.dec_lnorm_1 = tf.keras.layers.LayerNormalization(name="Decoding_LNorm_1")
        self.dec_lrelu_1 = tf.keras.layers.LeakyReLU(name="Decoding_LRelu_1")
        
        self.dec_dense_2 = tf.keras.layers.Dense(decoder_filters[1], name="Decoding_Reconstruct_2")
        self.dec_lnorm_2 = tf.keras.layers.LayerNormalization(name="Decoding_LNorm_2")
        self.dec_lrelu_2 = tf.keras.layers.LeakyReLU(name="Decoding_LRelu_2")

        self.dec_dense_3 = tf.keras.layers.Dense(decoder_filters[2], name="Decoding_Reconstruct_3")
        self.dec_lnorm_3 = tf.keras.layers.LayerNormalization(name="Decoding_LNorm_3")
        self.dec_lrelu_3 = tf.keras.layers.LeakyReLU(name="Decoding_LRelu_3")

        self.dec_dense_4 = tf.keras.layers.Dense(decoder_filters[3], name="Decoding_Reconstruct_4")
        self.dec_lnorm_4 = tf.keras.layers.LayerNormalization(name="Decoding_LNorm_4")
        self.dec_lrelu_4 = tf.keras.layers.LeakyReLU(name="Decoding_LRelu_4")

        self.dec_output = tf.keras.layers.Dense(decoder_filters[4], name="Decoding_OutPut_Dense")
        self.dec_sigmoid = tf.keras.layers.Activation('sigmoid', name="Decoding_OutPut_sigmoid")
        
    @tf.function
    def encoder(self, x):
        x = self.enc_conv_1(x)
        x = self.enc_lrelu_1(x)
        x = self.enc_conv_2(x)
        x = self.enc_lnorm_2(x)
        x = self.enc_lrelu_2(x)
        x = self.enc_conv_3(x)
        x = self.enc_lnorm_3(x)
        x = self.enc_lrelu_3(x)
        x = self.enc_conv_4(x)
        x = self.enc_lnorm_4(x)
        x = self.enc_lrelu_4(x)
        x = self.enc_flatten(x)
        mu = self.enc_mu(x)
        std = self.enc_std(x)
        return mu, std
    
    @tf.function
    def latent(self, mu, std, reparam=True):
        if reparam:
            return mu + std * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        else:
            return mu

    @tf.function
    def decoder(self, z):
        z = self.dec_dense_1(z)
        z = self.dec_lnorm_1(z)
        z = self.dec_lrelu_1(z)
        z = self.dec_dense_2(z)
        z = self.dec_lnorm_2(z)
        z = self.dec_lrelu_2(z)
        z = self.dec_dense_3(z)
        z = self.dec_lnorm_3(z)
        z = self.dec_lrelu_3(z)
        z = self.dec_dense_4(z)
        z = self.dec_lnorm_4(z)
        z = self.dec_lrelu_4(z)
        z = self.dec_output(z)
        z = self.dec_sigmoid(z)
        z = tf.reshape(z, [z.get_shape()[0], z.get_shape()[1], 1], name="Output_Reshape")
        return z


    @tf.function
    def call(self, input_tensor, reparam=True):
        # From Encoder
        mu_enc, stddev_enc = self.encoder(input_tensor)

        # From Latent
        z = self.latent(mu_enc, stddev_enc, reparam=reparam)

        # From Decoder
        y = self.decoder(z)
        
        return y, mu_enc, stddev_enc

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

def train(model, data_col, time_size, over_len, batch_size, beta_cycle, reparam,\
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
        train_dataset = data_sampling(data_col=data_col, time_size=time_size, over_len=over_len, batch_size=batch_size)

        # Cal Beta
        beta = cal_beta_num2(ep_, beta_cycle)

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
        if ep_ % 5 == 0:
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

def data_sampling(data_col, time_size, over_len, batch_size):
    data_name_list = list(data_col.keys())
    #random_data_name = data_name_list[np.random.permutation(len(data_name_list))[0]]
    #data_sample = utils.make_dataformat_from_mit(data_col=data_col, name=random_data_name, time_len=time_size, over_len=over_len)
    data_sample = utils.make_dataformat_from_mit(data_col=data_col, name=data_name_list[0], time_len=time_size, over_len=over_len)
    train_set = tensorset(arr = data_sample, shape=(-1, time_size, 1), batch_size=batch_size)

    return train_set

# Return Tensor dataset
def tensorset(arr, shape, batch_size, drop_remainder=True):
    # type casting & reshaping
    data = arr.astype(np.float32)
    data = np.reshape(data, shape)
    # make to tensor
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=data.shape[0]*3)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds
