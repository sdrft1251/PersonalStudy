import tensorflow as tf
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import io

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