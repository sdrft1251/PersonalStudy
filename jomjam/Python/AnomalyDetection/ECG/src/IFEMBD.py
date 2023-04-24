import tensorflow as tf
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as pltx
import io
import math

############################################################################################################################################################
################# Utils
############################################################################################################################################################
class EncoderCompress(tf.keras.layers.Layer):
    def __init__(self, name="EncoderCompress"):
        super(EncoderCompress, self).__init__(name=name)

    def call(self, x):
        outputs = tf.math.reduce_mean(x, axis=1)
        return outputs

############################################################################################################################################################
################# Embedding
############################################################################################################################################################
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        return self.pe[:, :inputs.shape[1], :]

class PositionalEmbeddingDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbeddingDecoder, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        outputs = self.pe[:, :inputs.shape[1], :] + inputs - inputs
        return outputs

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model,
                                    kernel_size=3, padding='causal', activation='linear')
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.tokenConv(inputs)
        x = self.activation(x)
        return x

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, d_model, name="time2vec"):
        super(Time2Vec, self).__init__(name=name)
        self.w0 = tf.keras.layers.Dense(1)
        self.wi = tf.keras.layers.Dense(d_model-1)

    def call(self, x):
        v0 = self.w0(x)
        v1 = self.wi(x)
        v1 = tf.math.sign(v1)
        return tf.concat([v0, v1], axis=-1)

class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.cycle_embdding = Time2Vec(d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, **kwargs):
        x = self.value_embedding(x) + self.position_embedding(x) + self.cycle_embdding(x)
        return self.dropout(x)

############################################################################################################################################################
################# Attention
############################################################################################################################################################
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        # 정확히 분배되는지 확인
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)
        return outputs

def scaled_dot_product_attention(query, key, value):
    # Get Q * K
    matmul_qk = tf.linalg.matmul(query, key, transpose_b=True)
    # For Scaling
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    # Get Attention Weights
    attention_weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.linalg.matmul(attention_weights, value)
    return outputs, attention_weights

############################################################################################################################################################
################# Encoder
############################################################################################################################################################
def encoder_layer(time_len, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    # Self-Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs})
    # Add & Norm
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # FFNN
    outputs = tf.keras.layers.Conv1D(filters=d_model*4, kernel_size=1)(attention)
    outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def compress_layer(time_len, d_model, dropout, name="compress_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    outputs = tf.keras.layers.Conv1D(filters=int(d_model/2), kernel_size=3, padding='causal')(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.ELU()(outputs)
    outputs = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def encoder(time_len, num_layers, d_model, num_heads, dropout, name="encoder"):
    # Model Part
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")
    # Time2Vec
    embeddings = DataEmbedding(d_model=d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    # Encoder layers
    for i in range(num_layers-1):
        outputs = encoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))(outputs)
        outputs = compress_layer(time_len=time_len, d_model=d_model, dropout=dropout, name="compress_layer_{}".format(i))(outputs)
        d_model = int(d_model/2)
        time_len = int(time_len/2)
    outputs = encoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(num_layers-1))(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

############################################################################################################################################################
################# Decoder
############################################################################################################################################################
def decoder_layer(time_len, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(d_model), name="encoder_outputs")
    # Self-Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention_1")({'query': inputs, 'key': inputs, 'value': inputs})
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # Attention with Encoder
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")({'query': attention, 'key': enc_outputs, 'value': enc_outputs})
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention)
    # FFNN
    outputs = tf.keras.layers.Conv1D(filters=d_model*4, kernel_size=1)(attention2)
    outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + outputs)
    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)

def decoder(time_len, num_layers, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")
    enc_outputs = tf.keras.Input(shape=(d_model), name="decoder_outputs")
    dec_input = PositionalEmbeddingDecoder(d_model=d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(dec_input)
    # Decoder layers
    for i in range(num_layers):
        outputs = decoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs])
    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)

############################################################################################################################################################
################# Total Model
############################################################################################################################################################
def IFEMBD(time_len, d_model, enc_layer_num, num_heads, dec_layer_num, dropout, name="IFEMBD"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")

    # Encoder Part
    enc_outputs = encoder(time_len=time_len, num_layers=enc_layer_num, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=inputs)

    # Compress Layer
    enc_outputs = EncoderCompress()(enc_outputs)

    # Decoder Part
    dec_outputs = decoder(time_len=time_len, num_layers=dec_layer_num, d_model=enc_outputs.shape[-1], num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_outputs])
    
    # For Output
    outputs = tf.keras.layers.Dense(1, name="OutPut_Dense")(dec_outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

############################################################################################################################
# Train Part
############################################################################################################################
# Cal ELBO Loss
def elbo_loss(model, inputs, beta):
    # From model
    outputs = model(inputs)
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    #sigma_dec = tf.squeeze(sigma_dec, axis=-1)
    # Latent loss: -KL[q(z|x)|p(z)]
    # KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu) + tf.math.square(sigma) - tf.math.log(1e-8 + tf.math.square(sigma)) - 1, 1)
    # KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(tf.math.square(inputs-outputs), 1)
    marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Reconstruction Loss: log(p(x|z)) - 2
    # marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(mu_dec) + (1 - inputs) * tf.math.log(1 - mu_dec), 1)
    # marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # Reconstruction Loss: log(p(x|z)) - 3
    # marginal_likelihood = tf.reduce_sum(0.5*tf.math.log(tf.math.square(sigma_dec))+0.5*tf.math.square(inputs-mu_dec)/tf.math.square(sigma_dec), 1)
    # marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    # ELBO = marginal_likelihood - (beta*KL_divergence)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(inputs-outputs))
    MAE = tf.math.reduce_mean(tf.math.abs(inputs-outputs))
    #print("ELBO : {} Marginal : {} KLD : {}".format(-ELBO.numpy(), -marginal_likelihood.numpy(), KL_divergence.numpy()))
    #return -ELBO, -marginal_likelihood, KL_divergence, MSE, MAE
    return -marginal_likelihood, MSE, MAE

# Gradient
def grad(model, inputs, beta):
    with tf.GradientTape() as tape:
        reconstruct_er, mse, mae = elbo_loss(model, inputs, beta)
    return reconstruct_er, mse, mae, tape.gradient(reconstruct_er, model.trainable_variables)

def train(model, train_set, epochs, batch_size, beta_cycle, beta_rate, learning_rate, summary_dir, add_name, cp_dir, sample_data_set):
    # train_loss_results = []
    # train_metric_results = []
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
        # epoch_elbo_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        # epoch_kld_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_mae_avg = tf.keras.metrics.Mean()
        # Data Resampling
        train_dataset = tensorset(arr=train_set, shape=(-1, train_set.shape[1], 1), batch_size=batch_size)
        # Cal Beta
        beta = cal_beta_basic(ep_, beta_cycle) * beta_rate
        # In Batch
        for x in train_dataset:
            # Get Grad
            reconstruct_er, mse, mae, grads = grad(model, x, beta)
            # Apply Gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # For Monitoring
            # epoch_elbo_avg(elbo)
            epoch_reconstruct_avg(reconstruct_er)
            # epoch_kld_avg(kld)
            epoch_mse_avg(mse)
            epoch_mae_avg(mae)
        # train_loss_results.append(epoch_reconstruct_avg.result())
        # train_metric_results.append(epoch_mse_avg.result())
        
        # Printing Model result
        if ep_ % 1 == 0:
            print("EPOCH : {:05d} | ReCon : {:.6f} | MSE : {:.6f} | MAE : {:.6f} | Beta : {} | TrainSet Size : {}".format(\
            ep_, epoch_reconstruct_avg.result(), epoch_mse_avg.result(), epoch_mae_avg.result(), beta, train_set.shape))
        # Save Model
        if len(cp_dir) != 0:
            if ep_ % 2 == 0:
                model.save_weights(cp_dir+"/"+log_file_name+"/save")
        if len(summary_dir) != 0 :
            sample_output = model(tmp_sample)
            figure = image_grid(sample_output[:25].numpy())
            with writer.as_default():
                # tf.summary.scalar("ELBO Loss", epoch_elbo_avg.result(), step=ep_)
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                # tf.summary.scalar("KLD Loss", epoch_kld_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)
                tf.summary.scalar("MAE", epoch_mae_avg.result(), step=ep_)
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
