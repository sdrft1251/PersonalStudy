import tensorflow as tf
from datetime import datetime
import pytz
# 비교를 위해 만든 임의의 코드 스크립트

class AE(tf.keras.Model):

    def __init__(self,\
                hidden_size,\
                latent_length,\
                output_depth,\
                time_size,\
                dropout=0.0,\
                recurrent_dropout=0.0
                ):
        super(AE, self).__init__(name='')

        self.time_size = time_size
        self.hidden_size = hidden_size

        # For Ecoder
        self.encode_cell = tf.keras.layers.SimpleRNNCell(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Ecoding_RNN_Cell")
        self.encode_rnn = tf.keras.layers.RNN(self.encode_cell, return_state=True, name="RNN_Wrapper")
        self.encode_output_dense = tf.keras.layers.Dense(latent_length, name="Encoding_OutPut_Dense")

        # For Decoder
        self.decoder_first_state_dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name="Decoding_first_state_Dense")
        self.decode_cell = tf.keras.layers.SimpleRNNCell(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, name="Decoding_RNN_Cell")
        self.decoder_output_dense = tf.keras.layers.Dense(output_depth, activation='sigmoid', name="Decoding_Output")
            
    @tf.function
    def encoder(self, input_tensor):
        # encode layer (Simple lstm)
        x, last_states = self.encode_rnn(input_tensor)
        # get encode output
        enc_out = self.encode_output_dense(last_states)
        return enc_out

    @tf.function
    def decoder(self, input_tensor):
        out_collect = []
        # Get First State
        h_state = self.decoder_first_state_dense(input_tensor)
        # Loop Start
        for t in range(self.time_size):
            # Get output
            x_out = self.decoder_output_dense(h_state)
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
        enc_out = self.encoder(input_tensor)

        # From Decoder
        y_hat = self.decoder(enc_out)
        
        return y_hat

def elbo_loss(model, inputs):
    # From model
    y = model(inputs)
    # Reconstruction Loss: log(p(x|z))  
    marginal_likelihood = tf.reduce_sum(inputs * tf.math.log(y) + (1 - inputs) * tf.math.log(1 - y), 1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # For MSE
    MSE = tf.math.reduce_mean(tf.math.square(y-inputs))
    return -marginal_likelihood, MSE

def grad(model, inputs):
    with tf.GradientTape() as tape:
        reconstruct_er, mse = elbo_loss(model, inputs)
    return reconstruct_er, mse, tape.gradient(mse, model.trainable_variables)


def train(model, train_dataset, epochs, learning_rate=0.001, summary_dir="/logs", add_name=""):
    train_loss_results = []
    train_metric_results = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    KST = pytz.timezone('Asia/Seoul')
    log_file_name = datetime.now(KST).strftime("%Y%m%d_%H_%M_%S")+add_name
    if len(summary_dir) != 0 :
        writer = tf.summary.create_file_writer(summary_dir+"/"+log_file_name)
    for ep_ in range(epochs):
        epoch_mse_avg = tf.keras.metrics.Mean()
        epoch_reconstruct_avg = tf.keras.metrics.Mean()

        for x in train_dataset:
            # Get Gradient
            reconstruct_er, mse, grads = grad(model, x)
            # Apply Gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # For Monitoring
            epoch_reconstruct_avg(reconstruct_er)
            epoch_mse_avg(mse)

        train_loss_results.append(epoch_reconstruct_avg.result())
        train_metric_results.append(epoch_mse_avg.result())

        if ep_ % 5 == 0:
            print("EPOCH : {:03d} | Reconstruct : {:.3f} | MSE : {:.6f}".format(\
            ep_, epoch_reconstruct_avg.result(), epoch_mse_avg.result()))

        if len(summary_dir) != 0 :
            with writer.as_default():
                tf.summary.scalar("Reconstruct Loss", epoch_reconstruct_avg.result(), step=ep_)
                tf.summary.scalar("MSE", epoch_mse_avg.result(), step=ep_)

            writer.flush()

    return train_loss_results
