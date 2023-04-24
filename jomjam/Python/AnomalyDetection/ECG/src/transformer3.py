import tensorflow as tf
from datetime import datetime
import pytz
from src import utils
import numpy as np

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, output_dims, name="time2vec"):
        super(Time2Vec, self).__init__(name=name)
        self.output_dims = output_dims

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        # i=0
        self.w0 = tf.Variable(initial_value=w_init(shape=(input_shape[-1], 1),dtype=tf.float32), name="Time2Vec_w0", trainable=True)
        self.b0 = tf.Variable(initial_value=b_init(shape=(1),dtype=tf.float32), name="Time2Vec_b0", trainable=True)
        # i!=0
        self.wi = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.output_dims-1),dtype=tf.float32), name="Time2Vec_wi", trainable=True)
        self.bi = tf.Variable(initial_value=b_init(shape=(self.output_dims-1),dtype=tf.float32), name="Time2Vec_bi", trainable=True)

    def call(self, input_tensor):
        v0 = tf.linalg.matmul(input_tensor, self.w0) + self.b0
        v1 = tf.math.sign(tf.linalg.matmul(input_tensor, self.wi) + self.bi)
        return tf.concat([v0, v1], axis=-1)

class Vec2Time(tf.keras.layers.Layer):
    def __init__(self, name="output"):
        super(Vec2Time, self).__init__(name=name)
        
        self.v0_dense = tf.keras.layers.Dense(units=1)
        self.v1_dense = tf.keras.layers.Dense(units=1)

    def call(self, input_tensor):
        v0 = self.v0_dense(input_tensor)
        v1 = tf.experimental.numpy.arcsin(tf.math.tanh(input_tensor))
        v1 = self.v1_dense(v1)
        return tf.math.sigmoid(v0 + v1)

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

class Latent(tf.keras.layers.Layer):
    def __init__(self, latent_dims, name="latent"):
        super(Latent, self).__init__(name=name)

        self.mu_dense = tf.keras.layers.Dense(units=latent_dims)
        self.sigma_dense = tf.keras.layers.Dense(units=latent_dims)

    def call(self, input_tensor):
        mu = self.mu_dense(input_tensor)
        sigma = self.sigma_dense(input_tensor)
        sigma = tf.math.softplus(sigma)
        return mu, sigma

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

def encoder_layer(time_len, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    # Self-Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs})
    # Add & Norm
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # FFNN
    outputs = tf.keras.layers.Dense(units=int(d_model/2), activation=tf.keras.layers.LeakyReLU())(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def encoder(time_len, num_layers, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")
    # Time2Vec
    embeddings = Time2Vec(output_dims=d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    # Encoder layers
    for i in range(num_layers):
        outputs = encoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def decoder_layer(time_len, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(time_len, d_model), name="inputs")
    # Self_Attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs})
    # Add & Norm
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # FFNN
    outputs = tf.keras.layers.Dense(units=int(d_model/2), activation=tf.keras.layers.LeakyReLU())(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def decoder(time_len, input_dims, num_layers, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(time_len, input_dims), name="inputs")
    # Attention 준비
    outputs = tf.keras.layers.Dense(units=d_model)(inputs)
    # Decoder layers
    for i in range(num_layers):
        outputs = decoder_layer(time_len=time_len, d_model=d_model, num_heads=num_heads, dropout=dropout, name="decoder_layer_{}".format(i))(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def transformer_vae(time_len, d_model, enc_layer_num, num_heads, compress_dims, latent_len, exand_dims, dec_layer_num, dropout, reparam=True, name="transformer"):
    inputs = tf.keras.Input(shape=(time_len,1), name="inputs")

    # Encoder Part
    enc_outputs = encoder(time_len=time_len, num_layers=enc_layer_num, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=inputs)

    # Compress Layer
    enc_outputs = tf.keras.layers.Dense(compress_dims, activation=tf.keras.layers.LeakyReLU(), name="For_Compress_Dense")(enc_outputs)
    enc_outputs = tf.keras.layers.Flatten(name="For_Compress_flatten")(enc_outputs)

    # Latent Space
    enc_mu, enc_sigma = Latent(latent_dims=latent_len)(enc_outputs)
    # Re-Param
    if reparam:
        z = enc_mu + enc_sigma * tf.random.normal(tf.shape(enc_mu), 0, 1, dtype=tf.float32)
    else:
        z = enc_mu

    # Expand Layer
    z = tf.keras.layers.Dense(time_len, name="For_Expand_Dense_2")(z)
    z = tf.keras.layers.Reshape((time_len,1))(z)
    z = tf.keras.layers.Dense(exand_dims, activation=tf.keras.layers.LeakyReLU(), name="For_Expand_Dense_1")(z)

    # Decoder Part
    dec_outputs = decoder(time_len=time_len, input_dims=exand_dims, num_layers=dec_layer_num, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=z)

    # For Output
    outputs = Vec2Time()(dec_outputs)

    return tf.keras.Model(inputs=inputs, outputs=[outputs, enc_mu, enc_sigma], name=name)

# Cal ELBO Loss
def elbo_loss(model, inputs, beta):
    # From model
    mu_dec, mu_enc, sigma_enc = model(inputs)
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    mu_dec = tf.squeeze(mu_dec, axis=-1)
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(sigma_enc) - tf.math.log(1e-8 + tf.math.square(sigma_enc)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    # marginal_likelihood = tf.reduce_sum(tf.math.square(inputs-mu_dec), 1)
    # marginal_likelihood = -tf.reduce_mean(marginal_likelihood)
    # Reconstruction Loss: log(p(x|z)) - 2
    marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(mu_dec) + (1 - inputs) * tf.math.log(1 - mu_dec), 1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    ELBO = 10*marginal_likelihood - (beta*KL_divergence)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(mu_dec-inputs))
    return -ELBO, -marginal_likelihood, KL_divergence, MSE

# Gradient
def grad(model, inputs, beta, reparam=True):
    with tf.GradientTape() as tape:
        elbo, reconstruct_er, kld, mse = elbo_loss(model, inputs, beta)
    return elbo, reconstruct_er, kld, mse, tape.gradient(elbo, model.trainable_variables)

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
        epoch_elbo_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        epoch_kld_avg = tf.keras.metrics.Mean()
        # Data Resampling
        train_dataset = tensorset(arr=train_set, shape=(-1, train_set.shape[1], 1), batch_size=batch_size)
        # Cal Beta
        beta = cal_beta_basic(ep_, beta_cycle) * beta_rate
        # In Batch
        for x in train_dataset:
            # Get Grad
            elbo, reconstruct_er, kld, mse, grads = grad(model, x, beta)
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