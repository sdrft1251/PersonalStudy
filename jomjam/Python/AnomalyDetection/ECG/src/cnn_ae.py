import tensorflow as tf
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import io


####################################################################################
# VGG 16
####################################################################################
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

####################################################################################
# Model Part
####################################################################################
def Encoder_Module(T, latent_dims, name="Encoder_Module"):
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    # Compress
    comp = VGG16()(inputs)# batch, compressT, d_model
    # Encoder
    comp = tf.keras.layers.Flatten()(comp)
    outputs = tf.keras.layers.Dense(latent_dims)(comp)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def Decoder_Module(latent_dims, name="Decoder_Module"):
    inputs = tf.keras.Input(shape=(latent_dims), name="inputs")
    expand = tf.keras.layers.Dense(80, activation="relu")(inputs)
    expand = tf.keras.layers.Reshape((80, 1))(expand)
    expand = VGG16_Reverse()(expand)
    outputs = tf.keras.layers.Dense(1)(expand)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def VGG16_AE(T, latent_dims, name="VGG16_AE"):
    # Model Part
    inputs = tf.keras.Input(shape=(T,1), name="inputs")
    latent = Encoder_Module(T=T, latent_dims=latent_dims)(inputs)
    dec_output = Decoder_Module(latent_dims=latent_dims)(latent)
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