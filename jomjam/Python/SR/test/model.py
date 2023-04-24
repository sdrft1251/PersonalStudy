import tensorflow as tf
import numpy as np
import utils
import os
from time import time
import data

class EDSR(tf.keras.Model):
    def __init__(self, img_size=32, num_layers=32,feature_size=256, scale=2, output_channels=3, log_save_dir="./logs/save_log"):
        super(EDSR, self).__init__()
        self.img_size = img_size
        self.num_layers = num_layers
        self.scale = scale
        self.output_channels = output_channels
        self.feature_size = feature_size
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            self.model = self.comp_graph()
            self.train_op = tf.keras.optimizers.Adam()
            self.loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            log_save_dir = log_save_dir+"_{}".format(time())
            self.summary_writer = tf.summary.create_file_writer(log_save_dir)

    def comp_graph(self):
        input_tensor = tf.keras.Input(shape=(self.img_size,self.img_size,3))
        mean_x = 255
        input_tensor_scaled = input_tensor / mean_x
        conv_1 = tf.keras.layers.Conv2D(self.feature_size, kernel_size=(3,3), padding="same", name="conv_1")(input_tensor_scaled)
        conv_1_copy = tf.identity(conv_1)
        scaling_factor = 0.1
        for i in range(self.num_layers):
            conv_1 = utils.resBlock(conv_1, self.feature_size, scale=scaling_factor)
        conv_1 = tf.keras.layers.Conv2D(self.feature_size, kernel_size=(3,3), padding="same", name="conv_2")(conv_1)
        conv_1 = conv_1 + conv_1_copy
        conv_1 = utils.upsample(conv_1, self.scale, self.feature_size, None)

        out_scaled = conv_1 * mean_x
        out = tf.clip_by_value(out_scaled, 0.0, 255.0)
        model = tf.keras.Model(inputs=input_tensor, outputs=[conv_1, out])
        print(model.summary())
        return model
    
    def train(self, image_dir, batch_size=64, epoch=300, check_dir="./cp/"):
        images = data.load_dataset(image_dir)
        print(len(images))
        with self.mirrored_strategy.scope():
            def compute_loss(target, predictions):
                per_example_loss = self.loss_object(target, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)
            
            def train_step(inputs):
                x, y = inputs
                y = y/255
                with tf.GradientTape() as tape:
                    predictions,_ = self.model(x)
                    loss = compute_loss(y, predictions)
                gradients = tape.gradient(loss,self.model.trainable_variables)
                self.train_op.apply_gradients(zip(gradients, self.model.trainable_variables))                    
                return loss
            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = self.mirrored_strategy.experimental_run_v2(train_step,args=(dataset_inputs,))
                return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

            for ep in range(epoch):
                x, y = data.return_croped_images(image_list=images, img_size=self.img_size, scale=self.scale)
                train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(batch_size)
                train_dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(train_dataset)
                total_loss = 0.0
                num_batches = 0
                for step_data in train_dist_dataset:
                    total_loss += distributed_train_step(step_data)
                    num_batches += 1
                train_loss = total_loss / num_batches
                template = ("Epochs {}, Loss: {}")
                print (template.format(ep+1, train_loss))

                PSNR = tf.constant(255**2,dtype=tf.float32)/train_loss
                PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
                
                sample_image_input, sample_image_target = np.array(x[0:2]).reshape(-1,self.img_size,self.img_size,3), np.array(y[0:2]).reshape(-1,self.img_size*self.scale,self.img_size*self.scale,3)
                _,predicted_sample_image = self.model(sample_image_input)
                with self.summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=ep)
                    tf.summary.scalar("PSNR",PSNR, step=ep)
                    tf.summary.image("input_image",tf.cast(sample_image_input,tf.uint8), step=ep)
                    tf.summary.image("target_image",tf.cast(sample_image_target,tf.uint8), step=ep)
                    tf.summary.image("output_image",tf.cast(predicted_sample_image,tf.uint8), step=ep)

                self.summary_writer.flush()



