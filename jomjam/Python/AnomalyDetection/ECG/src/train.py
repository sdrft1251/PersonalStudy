import tensorflow as tf
from datetime import datetime
import pytz


def elbo_loss(model, inputs, bernoulli=True):
    # From model
    y, mu_enc, stddev_enc = model(inputs)
    # Latent loss: -KL[q(z|x)|p(z)]
    KL_divergence = 0.5 * tf.reduce_sum(tf.math.square(mu_enc) + tf.math.square(stddev_enc) - tf.math.log(1e-8 + tf.math.square(stddev_enc)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(y) + (1 - inputs) * tf.math.log(1 - y), 1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    ELBO = marginal_likelihood - KL_divergence
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(y-inputs))
    return -ELBO, -marginal_likelihood, KL_divergence, MSE

def grad(model, inputs, elbo_use=True):
    with tf.GradientTape() as tape:
        elbo, reconstruct_er, kld, mse = elbo_loss(model, inputs)
    if elbo_use:
        return elbo, reconstruct_er, kld, mse, tape.gradient(elbo, model.trainable_variables)
    else:
        return elbo, reconstruct_er, kld, mse, tape.gradient(reconstruct_er, model.trainable_variables)


def train(model, train_dataset, epochs, learning_rate=0.001, summary_dir="/logs", add_name=""):
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
            if ep_ <= int(epochs/2):
                elbo, reconstruct_er, kld, mse, grads = grad(model, x, elbo_use=False)
            else:
                elbo, reconstruct_er, kld, mse, grads = grad(model, x, elbo_use=True)
            # Apply Gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # For Monitoring
            epoch_elbo_avg(elbo)
            epoch_reconstruct_avg(reconstruct_er)
            epoch_kld_avg(kld)
            epoch_mse_avg(mse)

        train_loss_results.append(epoch_elbo_avg.result())
        train_metric_results.append(epoch_mse_avg.result())

        if ep_ % 5 == 0:
            print("EPOCH : {:03d} | ELBO : {:.3f} | Reconstruct : {:.3f} | KLD : {:.3f} | MSE : {:.6f}".format(\
            ep_, epoch_elbo_avg.result(), epoch_reconstruct_avg.result(), epoch_kld_avg.result(), epoch_mse_avg.result()))

        if len(summary_dir) != 0 :
            with writer.as_default():
                tf.summary.scalar("ELBO Loss", epoch_elbo_avg.result(), step=ep_)
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                tf.summary.scalar("KLD Loss", epoch_kld_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)

            writer.flush()

    return train_loss_results



