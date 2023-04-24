import tensorflow as tf

def reconstruct_loss(inputs, outputs, dist_train=False):
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    # Reconstruction Loss
    if dist_train:
        recon_loss = tf.reduce_sum(tf.math.square(inputs-outputs), axis=1)
        recon_loss = tf.reduce_mean(recon_loss, keepdims=True)
    else:
        recon_loss = tf.reduce_sum(tf.math.square(inputs-outputs), axis=1)
        recon_loss = tf.reduce_mean(recon_loss)
    return recon_loss

def mse_loss(inputs, outputs, dist_train=False):
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    # Reconstruction Loss
    if dist_train:
        mse = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.square(inputs-outputs), axis=1), keepdims=True)
    else:
        mse = tf.math.reduce_mean(tf.math.square(inputs-outputs))
    return mse

def mae_loss(inputs, outputs, dist_train=False):
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    # Reconstruction Loss
    if dist_train:
        mse = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(inputs-outputs), axis=1), keepdims=True)
    else:
        mse = tf.math.reduce_mean(tf.math.abs(inputs-outputs))
    return mse

def elbo_loss(latent_mu, latent_sigma, inputs, outputs, beta=1, recon_opt=1, dist_train=False):
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(latent_mu) + tf.math.square(latent_sigma) - tf.math.log(1e-8 + tf.math.square(latent_sigma)) - 1, 1)
    if dist_train:
        KL_divergence = tf.reduce_mean(KL_divergence, keepdims=True)
    else:
        KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z)) - 1
    if recon_opt==1:
        marginal_likelihood = -tf.reduce_sum(tf.math.square(inputs-outputs), 1)
    # Reconstruction Loss: log(p(x|z)) - 2
    elif recon_opt==2:
        marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(outputs) + (1 - inputs) * tf.math.log(1 - outputs), 1)
    if dist_train:
        marginal_likelihood = tf.reduce_mean(marginal_likelihood, keepdims=True)
    else:
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # Cal ELBO
    ELBO = marginal_likelihood - (beta*KL_divergence)
    return -ELBO, -marginal_likelihood, KL_divergence, MSE, MAE

