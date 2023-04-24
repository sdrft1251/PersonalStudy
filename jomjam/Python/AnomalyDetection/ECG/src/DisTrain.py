import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pytz
import io

def loss_func(model, inputs, batch_size):
    # From model
    outputs = model(inputs)
    # Squeeze Dimension
    inputs = tf.squeeze(inputs, axis=-1)
    outputs = tf.squeeze(outputs, axis=-1)
    # Reconstruction Loss: log(p(x|z))
    marginal_likelihood = tf.reduce_sum(tf.math.square(inputs-outputs), axis=1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood, keepdims=True)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.square(inputs-outputs), axis=1), keepdims=True)
    MAE = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(inputs-outputs), axis=1), keepdims=True)
    #print("Marginal : {}".format(marginal_likelihood.numpy()))
    #return -ELBO, -marginal_likelihood, KL_divergence, MSE, MAE
    return marginal_likelihood, MSE, MAE

@tf.function
def train_step(model, dist_inputs, batch_size, optimizer, mirrored_strategy):
    # For Gradients
    def grad(inputs):
        with tf.GradientTape() as tape:
            reconstruct_er, mse, mae = loss_func(model, inputs, batch_size)
        return reconstruct_er, mse, mae, tape.gradient(reconstruct_er, model.trainable_variables)
    # Step Function
    def step_fn(inputs):
        reconstruct_er, mse, mae, grads = grad(inputs)
        # Apply Gradient
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return reconstruct_er, mse, mae

    per_example_recon, per_example_mse, per_example_mae = mirrored_strategy.run(step_fn, args=(dist_inputs,))
    mean_loss_recon = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_recon, axis=0)
    mean_loss_mse = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_mse, axis=0)
    mean_loss_mae = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_mae, axis=0)
    return mean_loss_recon, mean_loss_mse, mean_loss_mae

def train(model, train_set, epochs, batch_size, learning_rate, mirrored_strategy, summary_dir, add_name, cp_dir, sample_data_set):
    # For 분산학습
    with mirrored_strategy.scope():
        # 옵티마이저 선언
        optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
        # 데이터셋
        train_dataset = tensorset(arr=train_set, shape=(-1, train_set.shape[1], 1), batch_size=batch_size)
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

    # For File Save Name
    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)
        tmp_sample = tensorset_forsee(arr=sample_data_set, shape=(-1, sample_data_set.shape[1], 1))

    # Train Loop
    for ep_ in range(epochs):
        # 기록용
        epoch_reconstruct_avg = tf.keras.metrics.Mean()
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_mae_avg = tf.keras.metrics.Mean()

        with mirrored_strategy.scope():
            for inputs in dist_dataset:
                reconstruct_er, mse, mae = train_step(model=model, dist_inputs=inputs,\
                batch_size=batch_size, optimizer=optimizer, mirrored_strategy=mirrored_strategy)

                epoch_reconstruct_avg(reconstruct_er)
                epoch_mse_avg(mse)
                epoch_mae_avg(mae)

            # Printing Model result
            if ep_ % 1 == 0:
                print("EPOCH : {:05d} | ReCon : {:.6f} | MSE : {:.6f} | MAE : {:.6f} | TrainSet Size : {}".format(\
                ep_, epoch_reconstruct_avg.result(), epoch_mse_avg.result(), epoch_mae_avg.result(), train_set.shape))
            # Save Model
            if len(cp_dir) != 0:
                if ep_ % 2 == 0:
                    model.save_weights(cp_dir+"/"+log_file_name+"/save")

            if len(summary_dir) != 0 :
                sample_output = model(tmp_sample)
                figure = image_grid(sample_output[:25].numpy())
                with writer.as_default():
                    tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                    tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)
                    tf.summary.scalar("MAE", epoch_mae_avg.result(), step=ep_)
                    tf.summary.image("Sample image from decoder", plot_to_image(figure), step=ep_)
                writer.flush()
    return 0

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