import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import io

####################################################################################
# Encoder Part
####################################################################################
class EncoderGRU(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        """
        hidden_size : feature dimension
        h0 : initial hidden state
        """
        super(EncoderGRU, self).__init__(name="Encoder_GRU")
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

class InputAttention(tf.keras.layers.Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="Input_Attention")
        self.w1 = tf.keras.layers.Dense(T)
        self.w2 = tf.keras.layers.Dense(T)
        self.v = tf.keras.layers.Dense(1)

    def call(self, h_s, x):
        """
        h_s : hidden_state (shape = batch,hidden_size)
        x : time series encoder inputs (shape = batch,T,n)
        """
        query = tf.keras.layers.RepeatVector(x.shape[2])(h_s)  # batch, n, hidden_size
        x_perm = tf.keras.layers.Permute((2, 1))(x)  # batch, n, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T
        score = self.v(score)  # batch, n, 1
        score = tf.keras.layers.Permute((2, 1))(score)  # batch,1,n
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights

class Encoder(tf.keras.layers.Layer):
    def __init__(self, T, hidden_size):
        super(Encoder, self).__init__(name="Encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.gru = EncoderGRU(hidden_size)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, n, training=True):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        self.gru.reset_state(h0=h0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = tf.keras.layers.Lambda(lambda x: data[:, t, :])(data)
            h_s = self.gru(x) # (batch, hidden_size)
            self.alpha_t = self.input_att(h_s, data)  # batch,1,n
            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n
        output = tf.multiply(data, alpha_seq)  # batch, T, n
        return output

####################################################################################
# Decoder Part
####################################################################################
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

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, latent_dims):
        super(TemporalAttention, self).__init__(name="Temporal_Attention")
        self.w1 = tf.keras.layers.Dense(latent_dims)
        self.w2 = tf.keras.layers.Dense(latent_dims)
        self.v = tf.keras.layers.Dense(1)

    def call(self, h_s, latent):
        """
        h_s : hidden_state (shape = batch, hidden_dims)
        latent : time series encoder inputs (shape = batch, latent_dims)
        """
        score = tf.nn.tanh(self.w1(latent) + self.w2(h_s))  # batch, latent_dims
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # Latent Space 안에의 중요성 # batch, latent_dims
        return attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, T, latent_dims):
        super(Decoder, self).__init__(name="Decoder")
        self.T = T
        self.temp_att = TemporalAttention(latent_dims)
        self.gru = DecoderGRU(hidden_size)
        self.decoder_first_state_dense = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.decoder_output_mu_dense = tf.keras.layers.Dense(1)
        self.reinput = tf.keras.layers.Dense(1)
        self.context_v = None
        self.beta_t = None

    def call(self, latent):
        """
        latent : Latent Space state (shape = batch, latent_dims)
        """
        out_collect = []
        h_s = self.decoder_first_state_dense(latent)
        self.gru.reset_state(h0=h_s)
        self.beta_t = self.temp_att(h_s, latent)  # batch, latent_dims
        self.context_v = tf.math.multiply(self.beta_t, latent)  # batch, latent_dims
        for t in range(self.T):
            # Output collect
            out = self.decoder_output_mu_dense(h_s)
            out_collect.append(out)
            # Make New Input
            x = tf.concat([out, self.context_v], axis=-1) # batch, latent_dims+1
            x = self.reinput(x) # batch, 1
            # Input to RNN
            h_s = self.gru(x)  # batch, 1 -> batch, hidden_dims
            self.beta_t = self.temp_att(h_s, latent)  # batch, latent_dims
            self.context_v = tf.math.multiply(self.beta_t, latent)  # batch, latent_dims
        # Last Output
        out = self.decoder_output_mu_dense(h_s)
        out_collect.append(out)
        # Stack
        out_collect = tf.stack(out_collect[1:])
        out_collect = tf.transpose(out_collect, [1, 0, 2])
        return out_collect

####################################################################################
# Model Part
####################################################################################
def Encoder_Module(T, d_model, hidden_size, h0=None, name="Encoder_Module"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    # Compress
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=10, strides=3)(inputs)
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=4)(compress)
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=4)(compress)
    compress = tf.keras.layers.ReLU()(compress)
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=10, strides=3)(compress)
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=4)(compress)
    compress = tf.keras.layers.Conv1D(filters=d_model, kernel_size=4)(compress)
    compress = tf.keras.layers.ReLU()(compress) # batch, compressT, d_model
    # Encoder
    outputs = Encoder(compress.shape[1], hidden_size)(compress, h0, d_model)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def Decoder_Module(latent_dims, hidden_size, T, name="Decoder_Module"):
    inputs = tf.keras.Input(shape=(latent_dims), name="inputs")
    outputs = Decoder(hidden_size, T, latent_dims)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def DAGRU(T, d_model, batch_size, hidden_size, latent_dims, name="DARNN"):
    # Init
    h0 = tf.zeros((batch_size, hidden_size))
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    enc_output = Encoder_Module(T=T, d_model=d_model, hidden_size=hidden_size, h0=h0)(inputs) # batch, compressT, d_model
    _, h_state = tf.keras.layers.RNN(tf.keras.layers.GRUCell(hidden_size), return_state=True)(enc_output) # batch, latent_dims
    latent = tf.keras.layers.Dense(latent_dims)(h_state)
    dec_output = Decoder_Module(latent_dims=latent_dims, hidden_size=hidden_size, T=T)(latent)
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

def train(model, train_set, epochs, batch_size, beta_cycle, beta_rate, learning_rate, summary_dir, add_name, cp_dir, sample_data_set):
    train_loss_results = []
    train_metric_results = []
    # Set Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    # For File Save Name
    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)
        tmp_sample = tensorset_forsee(arr=sample_data_set, shape=(-1, sample_data_set.shape[1], 1))
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
            sample_output, _ = model(tmp_sample)
            figure = image_grid(sample_output[:25].numpy())
            with writer.as_default():
                #tf.summary.scalar("ELBO Loss", epoch_elbo_avg.result(), step=ep_)
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                #tf.summary.scalar("KLD Loss", epoch_kld_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)
                tf.summary.image("Sample image from decoder", plot_to_image(figure), step=ep_)
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

# Return Tensor dataset - Non shuffle
def tensorset_forsee(arr, shape):
    # type casting & reshaping
    data = arr.astype(np.float32)
    data = np.reshape(data, shape)
    # make to tensor
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    return data

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


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def image_grid(sample_data):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i, sam_ in enumerate(sample_data):
        sam_ = sam_.reshape(-1)
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title="Index : {}".format(i))
        plt.plot(np.arange(len(sam_)), sam_)
    return figure