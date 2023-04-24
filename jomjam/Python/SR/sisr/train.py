import tensorflow as tf
from sisr import data
import numpy as np
from sisr import utils
from sisr import model
import time

class srgantrainer:
    def __init__(self, gen_model="edsr", num_res_blocks_gen=32, num_filters_gen=256,disc_model="resnet50", content_loss_layer_num=8, learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000],
    values=[1e-5, 1e-6]), batch_size=64, img_size=32, scale=4, log_save_dir="./logs/save_log", checkpoint_dir="./cp/save_cp", pre_train=True, model_reuse_path=(None,None), cost_rate=0.001):
        self.gen_model = gen_model
        self.num_res_blocks_gen = num_res_blocks_gen
        self.num_filters_gen = num_filters_gen
        self.disc_model = disc_model
        self.batch_size = batch_size
        self.img_size = img_size
        self.scale = scale
        self.cost_rate = cost_rate
        self.pre_train = pre_train
        self.log_save_dir_time = log_save_dir+"_Gen@ResBlock@filter@{}_{}_CostRate@{}_{}".format(self.num_res_blocks_gen, self.num_filters_gen, self.cost_rate, time.time())
        self.checkpoint_dir_time = checkpoint_dir+"_{}".format(time.time())
        self.model_reuse_path = model_reuse_path
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            if self.gen_model == "edsr":
                self.generator = model.EDSR(img_size=self.img_size, num_res_blocks=self.num_res_blocks_gen,num_filters=self.num_filters_gen, res_block_scaling=0.1, scale=4)
            if self.disc_model == "resnet50":
                self.discriminator = model.resnet50(in_shape=(self.img_size*scale,img_size*scale,3))
            elif self.disc_model == "resnet32":
                self.discriminator = model.resnet32(in_shape=(self.img_size*scale,img_size*scale,3))
            if self.model_reuse_path[0] != None:
                self.generator.load_weights(model_reuse_path[0])
            if self.model_reuse_path[1] != None:
                self.discriminator.load_weights(model_reuse_path[1])
            self.content_loss_layer_num = content_loss_layer_num
            self.disc_mid_output = tf.keras.Model(inputs=self.discriminator.input, outputs=self.discriminator.layers[content_loss_layer_num].output)
            print(self.disc_mid_output.summary())
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
            self.mean_squared_error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.summary_writer = tf.summary.create_file_writer(self.log_save_dir_time)

    def train(self, data_dir, epochs=200000):
        images = data.load_dataset(data_dir)
        with self.mirrored_strategy.scope():
            @tf.function
            def distributed_train_step(dataset_inputs):
                gen_per_replica_losses, disc_per_replica_losses, psnrs = self.mirrored_strategy.experimental_run_v2(self.train_step, args=(dataset_inputs,))
                return (self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, psnrs,axis=None))
            
            @tf.function
            def distributed_pre_train_step(dataset_inputs):
                gen_per_replica_losses, disc_per_replica_losses, psnr = self.mirrored_strategy.experimental_run_v2(self.pre_train_step,args=(dataset_inputs,))
                return (self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, psnrs,axis=None))

            for ep in range(epochs):
                x, y = data.return_croped_images(image_list=images, img_size=self.img_size, scale=self.scale)
                train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(self.batch_size)
                train_dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(train_dataset)
                if self.pre_train :
                    if ep <= 100:
                        gen_tot_loss = 0.0
                        disc_tot_loss = 0.0
                        psnr_tot = 0.0
                        num_batches = 0.0
                        for step_data in train_dist_dataset:
                            pl, dl, psnr = distributed_pre_train_step(step_data)
                            gen_tot_loss+=pl
                            disc_tot_loss+=dl
                            psnr_tot+=psnr
                            num_batches+=1.0
                        gen_loss = gen_tot_loss/num_batches
                        disc_loss = disc_tot_loss/num_batches
                        psnr_value = psnr_tot/num_batches
                        template = ("====Epochs {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                        print (template.format(ep+1, gen_loss, psnr_value, disc_loss))

                        forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(5)
                        image_batch, target_batch = next(iter(forshowing_set))
                        predict_for_showing = self.generator(image_batch)
                        for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                        with self.summary_writer.as_default():
                            tf.summary.scalar("generator Loss", gen_loss, step=ep)
                            tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                            tf.summary.scalar("PSNR", psnr_value, step=ep)
                            tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)

                        self.summary_writer.flush()

                    else :
                        gen_tot_loss = 0.0
                        disc_tot_loss = 0.0
                        psnr_tot = 0.0
                        num_batches = 0.0
                        for step_data in train_dist_dataset:
                            pl, dl, psnr = distributed_train_step(step_data)
                            gen_tot_loss+=pl
                            disc_tot_loss+=dl
                            psnr_tot+=psnr
                            num_batches+=1.0
                        gen_loss = gen_tot_loss/num_batches
                        disc_loss = disc_tot_loss/num_batches
                        psnr_value = psnr_tot/num_batches
                        template = ("====Epochs {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                        print (template.format(ep+1, gen_loss, psnr_value, disc_loss))
                        
                        forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(3)
                        image_batch, target_batch = next(iter(forshowing_set))
                        predict_for_showing = self.generator(image_batch)
                        for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                        with self.summary_writer.as_default():
                            tf.summary.scalar("generator Loss", gen_loss, step=ep)
                            tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                            tf.summary.scalar("PSNR", psnr_value, step=ep)
                            tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)
                        self.summary_writer.flush()
                else:
                    gen_tot_loss = 0.0
                    disc_tot_loss = 0.0
                    psnr_tot = 0.0
                    num_batches = 0.0
                    for step_data in train_dist_dataset:
                        pl, dl, psnr = distributed_train_step(step_data)
                        gen_tot_loss+=pl
                        disc_tot_loss+=dl
                        psnr_tot+=psnr
                        num_batches+=1.0
                    gen_loss = gen_tot_loss/num_batches
                    disc_loss = disc_tot_loss/num_batches
                    psnr_value = psnr_tot/num_batches
                    template = ("====Epochs {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                    print (template.format(ep+1, gen_loss, psnr_value, disc_loss))
                    
                    forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(3)
                    image_batch, target_batch = next(iter(forshowing_set))
                    predict_for_showing = self.generator(image_batch)
                    for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                    with self.summary_writer.as_default():
                        tf.summary.scalar("generator Loss", gen_loss, step=ep)
                        tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                        tf.summary.scalar("PSNR", psnr_value, step=ep)
                        tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)

                    self.summary_writer.flush()

                if ep%3==0:
                    self.generator.save_weights(self.checkpoint_dir_time+"_w_gen")
                    self.discriminator.save_weights(self.checkpoint_dir_time+"_w_disc")

    def train2(self, data_dir, epochs=200000):
        images = data.load_dataset(data_dir)
        print(len(images))
        with self.mirrored_strategy.scope():
            @tf.function
            def distributed_train_step(dataset_inputs):
                gen_per_replica_losses, gen_per_replica_losses2, disc_per_replica_losses, psnrs = self.mirrored_strategy.experimental_run_v2(self.train_step2, args=(dataset_inputs,))
                return (self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_per_replica_losses2,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, psnrs,axis=None))
            
            @tf.function
            def distributed_pre_train_step(dataset_inputs):
                gen_per_replica_losses, disc_per_replica_losses, psnrs = self.mirrored_strategy.experimental_run_v2(self.pre_train_step,args=(dataset_inputs,))
                return (self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_per_replica_losses,axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, psnrs,axis=None))

            for ep in range(epochs):
                x, y = data.return_croped_images(image_list=images, img_size=self.img_size, scale=self.scale)
                train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(self.batch_size)
                train_dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(train_dataset)
                if self.pre_train :
                    if ep <= 100:
                        gen_tot_loss = 0.0
                        disc_tot_loss = 0.0
                        psnr_tot = 0.0
                        num_batches = 0.0
                        for step_data in train_dist_dataset:
                            pl, dl, psnr = distributed_pre_train_step(step_data)
                            gen_tot_loss+=pl
                            disc_tot_loss+=dl
                            psnr_tot += psnr
                            num_batches+=1.0
                        gen_loss = gen_tot_loss/num_batches
                        disc_loss = disc_tot_loss/num_batches
                        psnr_value = psnr_tot/num_batches
                        template = ("====Epochs {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                        print (template.format(ep+1, gen_loss, psnr_value, disc_loss))

                        forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(5)
                        image_batch, target_batch = next(iter(forshowing_set))
                        predict_for_showing = self.generator(image_batch)
                        for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                        with self.summary_writer.as_default():
                            tf.summary.scalar("generator Loss", gen_loss, step=ep)
                            tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                            tf.summary.scalar("PSNR", psnr_value, step=ep)
                            tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)

                        self.summary_writer.flush()

                    else :
                        gen_tot_loss = 0.0
                        gen_tot_loss2 = 0.0
                        disc_tot_loss = 0.0
                        psnr_tot = 0.0
                        num_batches = 0.0
                        for step_data in train_dist_dataset:
                            mse, pl, dl, psnr = distributed_train_step(step_data)
                            gen_tot_loss+=mse
                            gen_tot_loss2+=pl
                            disc_tot_loss+=dl
                            psnr_tot += psnr
                            num_batches+=1.0
                        gen_loss = gen_tot_loss/num_batches
                        gen_loss2 = gen_tot_loss2/num_batches
                        disc_loss = disc_tot_loss/num_batches
                        psnr_value = psnr_tot/num_batches
                        template = ("====Epochs {}, generator MSE Loss: {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                        print (template.format(ep+1, gen_loss, gen_loss2, psnr_value, disc_loss))
                        
                        forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(3)
                        image_batch, target_batch = next(iter(forshowing_set))
                        predict_for_showing = self.generator(image_batch)
                        for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                        with self.summary_writer.as_default():
                            tf.summary.scalar("generator MSE Loss", gen_loss, step=ep)
                            tf.summary.scalar("generator Loss", gen_loss2, step=ep)
                            tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                            tf.summary.scalar("PSNR", psnr_value, step=ep)
                            tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                            tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)
                        self.summary_writer.flush()
                else:
                    gen_tot_loss = 0.0
                    gen_tot_loss2 = 0.0
                    disc_tot_loss = 0.0
                    psnr_tot = 0.0
                    num_batches = 0.0
                    for step_data in train_dist_dataset:
                        mse, pl, dl, psnr = distributed_train_step(step_data)
                        gen_tot_loss+=mse
                        gen_tot_loss2+=pl
                        disc_tot_loss+=dl
                        psnr_tot += psnr
                        num_batches+=1.0
                    gen_loss = gen_tot_loss/num_batches
                    gen_loss2 = gen_tot_loss2/num_batches
                    disc_loss = disc_tot_loss/num_batches
                    psnr_value = psnr_tot/num_batches
                    template = ("====Epochs {}, generator MSE Loss: {}, generator Loss: {}, PSNR: {}==== discriminator Loss: {}")
                    print (template.format(ep+1, gen_loss, gen_loss2, psnr_value, disc_loss))
                    
                    forshowing_set = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(3)
                    image_batch, target_batch = next(iter(forshowing_set))
                    predict_for_showing = self.generator(image_batch)
                    for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                    with self.summary_writer.as_default():
                        tf.summary.scalar("generator MSE Loss", gen_loss, step=ep)
                        tf.summary.scalar("generator Loss", gen_loss2, step=ep)
                        tf.summary.scalar("discriminator Loss", disc_loss, step=ep)
                        tf.summary.scalar("PSNR", psnr_value, step=ep)
                        tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                        tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)

                    self.summary_writer.flush()

                if ep%3==0:
                    self.generator.save_weights(self.checkpoint_dir_time+"_w_gen")
                    self.discriminator.save_weights(self.checkpoint_dir_time+"_w_disc")


    @tf.function
    def train_step(self, inputs):
        lr, hr = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.generator(lr, training=True)
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)
            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)

            con_loss = tf.math.reduce_mean(con_loss, axis=[1,2])
            perc_loss = con_loss + self.cost_rate * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

            perc_loss = tf.nn.compute_average_loss(perc_loss, global_batch_size=self.batch_size)
            disc_loss = tf.nn.compute_average_loss(disc_loss, global_batch_size=self.batch_size)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)


        sr2 = tf.clip_by_value(sr, 0, 255)
        sr2 = tf.round(sr2)
        sr2 = tf.cast(sr2, tf.uint8)
        hr = tf.cast(hr, tf.uint8)
        psnr_value = utils.psnr(hr, sr2)
        psnr_value = tf.nn.compute_average_loss(psnr_value, global_batch_size=self.batch_size)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return perc_loss, disc_loss, psnr_value

    @tf.function
    def train_step2(self, inputs):
        lr, hr = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.generator(lr, training=True)
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)
            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)

            mse = self.mean_squared_error(hr/127.5, sr/127.5)
            mse_unit = tf.nn.compute_average_loss(mse, global_batch_size=self.batch_size)
            con_loss = tf.math.reduce_mean(con_loss, axis=[1,2])

            perc_loss = mse_unit + con_loss + self.cost_rate * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

            perc_loss = tf.nn.compute_average_loss(perc_loss, global_batch_size=self.batch_size)
            disc_loss = tf.nn.compute_average_loss(disc_loss, global_batch_size=self.batch_size)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        sr2 = tf.clip_by_value(sr, 0, 255)
        sr2 = tf.round(sr2)
        sr2 = tf.cast(sr2, tf.uint8)
        hr = tf.cast(hr, tf.uint8)
        psnr_value = utils.psnr(hr, sr2)
        psnr_value = tf.nn.compute_average_loss(psnr_value, global_batch_size=self.batch_size)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return mse_unit, perc_loss, disc_loss, psnr_value

    @tf.function
    def pre_train_step(self, inputs):
        lr, hr = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.generator(lr, training=True)
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            mse = self.mean_squared_error(hr/255, sr/255)
            mse_unit = tf.nn.compute_average_loss(mse, global_batch_size=self.batch_size)
            disc_loss = self._discriminator_loss(hr_output, sr_output)
            disc_loss = tf.nn.compute_average_loss(disc_loss, global_batch_size=self.batch_size)

        gradients_of_generator = gen_tape.gradient(mse_unit, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        sr2 = tf.clip_by_value(sr, 0, 255)
        sr2 = tf.round(sr2)
        sr2 = tf.cast(sr2, tf.uint8)
        hr = tf.cast(hr, tf.uint8)
        psnr_value = utils.psnr(hr, sr2)
        psnr_value = tf.nn.compute_average_loss(psnr_value, global_batch_size=self.batch_size)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return mse_unit, disc_loss, psnr_value


    @tf.function
    def _content_loss(self, hr, sr):
        sr_features = self.disc_mid_output(sr)
        hr_features = self.disc_mid_output(hr)
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss




"""
class edsrtrainer:
    def train(self, train_image_dir, val_image_dir, batch_size=32, epoch=300, check_dir="./cp/"):
        checkpoint_dir=check_dir + 'edsr'
        now = time.perf_counter()
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), psnr=tf.Variable(-1.0), model = self.model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=3)
        images = data.load_dataset(train_image_dir)
        print(len(images))

        val_images = data.load_dataset(val_image_dir)
        print(len(val_images))
        with self.mirrored_strategy.scope():
            def compute_loss(target, predictions):
                per_example_loss = self.loss_object(target, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

            def train_step(inputs):
                x, y = inputs
                y = tf.cast(y, dtype=tf.float32)
                y = y/255.0
                with tf.GradientTape() as tape:
                    predictions,_ = self.model(x, training=True)
                    loss = compute_loss(y, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.train_op.apply_gradients(zip(gradients, self.model.trainable_variables))
                return loss

            def test_step(inputs):
                x, y = inputs
                y = tf.cast(y, dtype=tf.float32)
                y = y/255.0
                predictions,_ = self.model(x, training=True)
                t_loss = compute_loss(y, predictions)
                self.test_loss.update_state(t_loss)
                

            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = self.mirrored_strategy.experimental_run_v2(train_step,args=(dataset_inputs,))
                return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
            @tf.function
            def distributed_test_step(dataset_inputs):
                return self.mirrored_strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

            for ep in range(epoch):
                checkpoint.step.assign_add(1)

                x, y = data.return_croped_images(image_list=images, img_size=self.img_size, scale=self.scale)
                train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(500000).batch(batch_size)
                train_dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(train_dataset)

                val_x, val_y = data.return_croped_images(image_list=val_images, img_size=self.img_size, scale=self.scale)
                val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y)).shuffle(500000).batch(batch_size)
                val_dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(val_dataset)

                total_loss = 0.0
                num_batches = 0
                for step_data in train_dist_dataset:
                    total_loss += distributed_train_step(step_data)
                    print(total_loss.numpy(),end=" ")
                    num_batches += 1
                if num_batches == 0:
                    print("???")
                    num_batches = 1
                train_loss = total_loss / num_batches
                
                psnr_value = utils.evaluate(self.model, train_dataset)
                print("\n")

                for step_data in val_dist_dataset:
                    distributed_test_step(step_data)
                test_psnr_value = utils.evaluate(self.model, val_dataset)

                template = ("====Epochs {}, Loss: {}, PSNR: {}==== && =====Test Loss: {} Test PSNR: {}")
                print (template.format(ep+1, train_loss, psnr_value, self.test_loss.result(), test_psnr_value))
                
                forshowing_set = tf.data.Dataset.from_tensor_slices((val_x,val_y)).shuffle(500000).batch(5)
                image_batch, target_batch = next(iter(forshowing_set))
                _,predict_for_showing = self.model(image_batch)
                for_comparings = tf.image.resize(image_batch, (self.img_size*self.scale, self.img_size*self.scale), method=tf.image.ResizeMethod.BICUBIC)

                with self.summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=ep)
                    tf.summary.scalar("PSNR", psnr_value, step=ep)
                    tf.summary.scalar("Test loss", self.test_loss.result(), step=ep)
                    tf.summary.scalar("Test PSNR", test_psnr_value, step=ep)
                    tf.summary.image("input_image",tf.cast(image_batch,tf.uint8), step=ep, max_outputs=5)
                    tf.summary.image("target_image",tf.cast(target_batch,tf.uint8), step=ep, max_outputs=5)
                    tf.summary.image("output_image",tf.cast(predict_for_showing,tf.uint8), step=ep, max_outputs=5)
                    tf.summary.image("for_comparing_image",tf.cast(for_comparings,tf.uint8), step=ep, max_outputs=5)

                self.summary_writer.flush()
                self.test_loss.reset_states()

                if psnr_value <= checkpoint.psnr:
                    now = time.perf_counter()
                    continue
                checkpoint.psnr = psnr_value
                checkpoint_manager.save()
                now = time.perf_counter()
"""