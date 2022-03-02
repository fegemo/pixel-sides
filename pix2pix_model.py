import abc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import datetime
from IPython import display

import io_utils
from configuration import *
import frechet_inception_distance as fid
from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

class S2SModel(abc.ABC):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="s2smodel"):
        """
        Params:
        - train_ds: the dataset used for training. Should have target images as labels
        - test_ds: dataset used for validation. Should have target images as labels
        - model_name: the specific direction of source to target image (eg, front2right).
                      Should be path-friendly
        - architecture_name: the network architecture + variation used (eg, pix2pix, pix2pix-wgan).
                             Should be path-friendly
        """
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model_name = model_name
        self.architecture_name = architecture_name
        self.checkpoint_dir = os.sep.join(
            [TEMP_FOLDER, "training-checkpoints", self.architecture_name, self.model_name])
        # WGAN's sigmoid(critic) outputs 0 when it is sure a patch is real
        # but GAN's sigmoid(discriminator) outputs 1... so WGAN should invert: 1 - value
        # when showing the debug of the discriminator/critic output
        self.invert_discriminator_value = False
        
    def predict(self, images, **kwargs):
        pass
    
    def fit(self, steps, UPDATE_STEPS, callbacks=["fid"]):
        log_folders = [TEMP_FOLDER, "logs", self.architecture_name, self.model_name]
        now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(os.sep.join([*log_folders, now_string]))
        try:
            result = self.do_fit(steps, now_string, UPDATE_STEPS, callbacks)
        finally:
            self.summary_writer.close()
        
        return result

    @abc.abstractmethod
    def do_fit(self, steps):
        pass


    def report_fid(self, number_of_test_images=TEST_SIZE, step=None):
        target_images = np.ndarray((number_of_test_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        generated_images = np.ndarray((number_of_test_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))

        for i, (input_image, target_image) in self.test_ds.take(number_of_test_images).enumerate():
            generated_image = self.generator(input_image, training=True)
            target_images[i] = tf.squeeze(target_image).numpy()
            generated_images[i] = tf.squeeze(generated_image).numpy()

        value = fid.compare(target_images, generated_images)

        if self.summary_writer and step != None:
            with self.summary_writer.as_default():
                tf.summary.scalar("fid", value, step=step, description=f"Frechét Inception Distance using {number_of_test_images} images")

        return value

    def save_generator(self, save_js_too=False):
        py_model_path = os.sep.join(["models", "py", "generator", self.architecture_name, self.model_name])
        js_model_path = os.sep.join(["models", "js", self.architecture_name, self.model_name])
        
        io_utils.delete_folder(py_model_path)
        io_utils.ensure_folder_structure(py_model_path)
        self.generator.save(py_model_path)

        if save_js_too:
            import tensorflowjs as tfjs
            io_utils.delete_folder(js_model_path)
            io_utils.ensure_folder_structure(js_model_path)
            tfjs.converters.save_keras_model(generator, js_model_path)
            
    def load_generator(self):
        self.generator = tf.keras.models.load_model(os.sep.join(["models", "py", "generator", self.architecture_name, self.model_name]))
    
    
    def save_discriminator(self):
        py_model_path = os.sep.join(["models", "py", "discriminator", self.architecture_name, self.model_name])
        
        io_utils.delete_folder(py_model_path)
        io_utils.ensure_folder_structure(py_model_path)
        self.discriminator.save(py_model_path)

            
    def load_discriminator(self):
        self.discriminator = tf.keras.models.load_model(os.sep.join(["models", "py", "discriminator", self.architecture_name, self.model_name]))
    
    
    def generate_images_from_dataset(self, maximum=None, dataset="test", steps=None):
        is_test = dataset == "test"
        
        if maximum == None:
            maximum = TEST_SIZE if is_test else TRAIN_SIZE
        
        dataset = self.test_ds if is_test else self.train_ds
        dataset = list(self.test_ds.take(maximum).as_numpy_iterator())
        
        base_image_path = os.sep.join([TEMP_FOLDER, "generated-images", self.architecture_name, self.model_name])
        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)
        for i, images in enumerate(dataset):
            image_path = os.sep.join([base_image_path, f"{i}.png"])
            io_utils.generate_comparison_input_target_generated(self.generator, [images], image_path, steps)

        print(f"Generated {i} images using the generator")

    def generate_discriminated_images(self, dataset="test", number_of_images=5):
        is_test = dataset == "test"
        dataset = self.test_ds if is_test else self.train_ds
        
        for (input_image, target_image) in dataset.take(number_of_images):
            io_utils.generate_discriminated_image(input_image, target_image, self.discriminator, self.generator, self.invert_discriminator_value)
        
    
class Pix2PixModel(S2SModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="pix2pix", num_patches=30, LAMBDA=100):
        super().__init__(train_ds, test_ds, model_name, architecture_name)
        self.LAMBDA = LAMBDA
        self.generator = self.Generator()
        self.discriminator = self.Discriminator(num_patches)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=5)


    @staticmethod
    def downsample(filters, size, apply_batchnorm=True, constraint=None):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            kernel_constraint=constraint,
            use_bias=False))

        if apply_batchnorm:
            result.add(layers.BatchNormalization())

        result.add(layers.LeakyReLU())

        return result

    
    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            layers.Conv2DTranspose(filters, size, strides=2,
                                    padding="same",
                                    kernel_initializer=initializer,
                                    use_bias=False))

        result.add(layers.BatchNormalization())

        if apply_dropout:
            result.add(layers.Dropout(0.5))

        result.add(layers.ReLU())

        return result

    
    def Generator(self):
        inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS]) #(batch_size, 64, 64, 4)

        down_stack = [
            Pix2PixModel.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 32, 32, 64)
            Pix2PixModel.downsample(128, 4),  # (batch_size, 16, 16, 128)
            Pix2PixModel.downsample(256, 4),  # (batch_size, 8, 8, 256)
            Pix2PixModel.downsample(512, 4),  # (batch_size, 4, 4, 512)
            Pix2PixModel.downsample(512, 4),  # (batch_size, 2, 2, 512)
            Pix2PixModel.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            Pix2PixModel.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            Pix2PixModel.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            Pix2PixModel.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            Pix2PixModel.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            Pix2PixModel.upsample(256, 4),  # (batch_size, 32, 32, 512)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding="same",
                                         kernel_initializer=initializer,
                                         activation="tanh")  # (batch_size, 64, 64, 4)

        x = inputs

        # downsampling e adicionando as skip-connections
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # camadas de upsampling e skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator(self, num_patches):
        initializer = tf.random_normal_initializer(0., 0.02)

        input = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
        target = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

        x = layers.concatenate([input, target])  # (batch_size, 64, 64, channels*2)

        
        if num_patches > 30:
            raise ValueError(f"num_patches for the discriminator should not exceed 30, but {num_patches} was given")
 
        # down = Pix2PixModel.downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
        # current_patches = 30
        # next_layer_filters = 64*2
        # while current_patches > max(num_patches, 2):
        #     print(f"Created downsample with {next_layer_filters}")
        #     down = Pix2PixModel.downsample(next_layer_filters, 4)(down)
        #     current_patches = ((current_patches+2) / 2) - 2
        #     next_layer_filters *= 2
        #     print(f"current_patches now {current_patches}")
        down = Pix2PixModel.downsample(8, 4, False)(x)  # (batch_size, 32, 32, 64)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(8, 2, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(16, 2, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)
        down = layers.Dropout(0.5)(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(16, 4, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(32, 2, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)
        down = layers.Dropout(0.5)(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(32, 4, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(64, 4, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)
        down = layers.Dropout(0.5)(down)

        down = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        down = layers.Conv2D(64, 3, strides=1, kernel_initializer=initializer)(down)
        down = layers.BatchNormalization()(down)
        down = layers.LeakyReLU()(down)        
        
        zero_pad1 = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
        conv = layers.Conv2D(128, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 128)
        batchnorm1 = layers.BatchNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(batchnorm1)

        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 128)
        last_filter_size = 4#4 if num_patches > 1 else 5
        last = layers.Conv2D(1, last_filter_size, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[input, target], outputs=last)



    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        # l2_loss = tf.reduce_mean(tf.square(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, input_image, target_image, step, UPDATE_STEPS):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target_image], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("disc_loss", disc_loss, step=step//UPDATE_STEPS)
            
            
            

    def do_fit(self, steps, run_folder, UPDATE_STEPS=1000, callbacks=[]):
        examples = list(self.test_ds.take(5).as_numpy_iterator())
      
        start = time.time()
        start_initial_value = start

        # try:
        #     tf.profiler.experimental.start("tblogdir")
        for step, (input_image, target_image) in self.train_ds.repeat().take(steps).enumerate():
            # with tf.profiler.experimental.Trace("train", step_num=step):

            # every UPDATE_STEPS and in the beginning, visualize 5x images to see
            # how training is going...
            if (step + 1) % UPDATE_STEPS == 0 or step == 0:
                display.clear_output(wait=True)

                if step != 0:
                    eta = tf.cast(time.time() - start_initial_value, tf.float32)
                    eta /= tf.cast(step, tf.float32) + tf.constant(1, dtype=tf.float32)
                    eta *= tf.cast(steps - step, tf.float32), 
                    print(f"Last {UPDATE_STEPS} steps took: {time.time()-start:.2f}s")
                    print(f"Estimated time to finish: {eta.numpy()[0] / 60:.2f}min\n")

                start = time.time()

                with self.summary_writer.as_default():
                    save_image_name = os.sep.join([TEMP_FOLDER, "logs", self.architecture_name, self.model_name, run_folder, "step_{:06d}.png".format(step + 1)])
                    image_data = io_utils.generate_comparison_input_target_generated(self.generator, examples, save_image_name, step + 1)
                    image_data = io_utils.plot_to_image(image_data)
                    tf.summary.image(save_image_name, image_data, step=(step+1)//UPDATE_STEPS, max_outputs=5)
                
                if "fid" in callbacks:
                    print(f"Calculating Fréchet Inception Distance at {(step + 1) / 1000}k with {TEST_SIZE} test examples...", end="", flush=True)
                    fid = self.report_fid(step=(step+1)//UPDATE_STEPS)
                    print(f" FID: {fid:.5f}")
                if "show_patches" in callbacks:
                    print(f"Showing discriminator patches")
                    self.generate_discriminated_images()
                    
                print(f"Step: {(step + 1)/1000}k")
                if step < steps - 1:
                    print("˯" * (UPDATE_STEPS // 10))

            # actually TRAIN
            self.train_step(input_image, target_image, step, UPDATE_STEPS)

            # dot feedback for every 10 training steps
            if (step + 1) % 10 == 0 and step < steps - 1:
                print(".", end="", flush=True)

            # saves a training checkpoint UPDATE_STEPS*5
            if (step + 1) % (UPDATE_STEPS*5) == 0 or (step + 1) == steps:
                self.checkpoint_manager.save()
               
        # finally:
        #     try:
        #         tf.profiler.experimental.stop()
        #     except:
        #         pass

        
class Pix2PixSegNetModel(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="pix2pix-segnet", LAMBDA=100):
        super().__init__(train_ds, test_ds, model_name, architecture_name, LAMBDA=LAMBDA)
    
#     def Generator():
#         pass
    
    @staticmethod
    def downsample_segnet(inputs, filters, size, convolution_steps=2, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        x = inputs
        for i in range(convolution_steps):
            x = layers.Conv2D(
                filters,
                size,
                strides=1,
                padding="same",
                kernel_initializer=initializer)(x)

            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)


        x, indices = MaxPoolingWithArgmax2D(pool_size=2)(x)
        # x, indices = tf.raw_ops.MaxPoolWithArgmax(input=x, ksize=[size, size], strides=[2, 2], padding="valid", Targmax=tf.dtypes.int32)
        return x, indices


    @staticmethod
    def upsample_segnet(inputs, filters, size, convolution_steps=2, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        x, indices = inputs
        # print(f"x is {x}")
        # print(f"indices is {indices}")
        x = MaxUnpooling2D(size=2)([x, indices])
        # x = tfa.layers.MaxUnpooling2D(pool_size=size)([x, indices])

        for i in range(convolution_steps):
            x = layers.Conv2D(filters, size, strides=1,
                                padding="same",
                                kernel_initializer=initializer
                             )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        if apply_dropout:
            x = layers.Dropout(0.5)(x)

        # print(f"shape of x after upsample: {tf.shape(x)}")
        return x
    
    def Discriminator(self, num_patches):
        initializer = tf.random_normal_initializer(0., 0.02)

        input = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
        target = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

        inputs = layers.concatenate([input, target])  # (batch_size, 64, 64, channels*2)

        output = Pix2PixSegNetModel.downsample_segnet(inputs, 64, 4, 2),  # (batch_size, 32, 32, 64)
        x, indices0 = output[0][0], output[0][1]
        output = Pix2PixSegNetModel.downsample_segnet(x, 128, 4, 1),  # (batch_size, 16, 16, 128)
        x, indices1 = output[0][0], output[0][1]
        output = Pix2PixSegNetModel.downsample_segnet(x, 256, 4, 2),  # (batch_size, 8, 8, 256)
        x, indices2 = output[0][0], output[0][1]
        output = Pix2PixSegNetModel.downsample_segnet(x, 512, 4, 1),  # (batch_size, 4, 4, 512)
        x, indices3 = output[0][0], output[0][1]


        x = Pix2PixSegNetModel.upsample_segnet([x, indices3], 256, 4, 3, apply_dropout=True),  # (batch_size, 8, 8, 512)
        x = Pix2PixSegNetModel.upsample_segnet([x, indices2], 128, 4, 3),  # (batch_size, 16, 16, 512)
        x = Pix2PixSegNetModel.upsample_segnet([x, indices1], 64, 4, 2),  # (batch_size, 32, 32, 256)

        last = layers.Conv2D(1, 3, strides=1,
                                padding="valid",
                                kernel_initializer=initializer)



        x = last(x[0])

        return tf.keras.Model(inputs=[input, target], outputs=x)
    
    
    
class Pix2PixFFTModel(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="pix2pix-fft", LAMBDA_L1=100, LAMBDA_FFT=100):
        super().__init__(train_ds, test_ds, model_name, architecture_name, LAMBDA=LAMBDA_L1)
        self.LAMBDA_FFT = LAMBDA_FFT
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        # print(target)
        # print(tf.shape(target).numpy())
        # print(gen_output)
        # print(tf.shape(gen_output).numpy())
        target = tf.squeeze(target[:, :, :, 0:3])
        target = tf.image.rgb_to_grayscale(target)

        gen_output = tf.squeeze(gen_output[:, :, :, 0:3])
        gen_output = tf.image.rgb_to_grayscale(gen_output)
        
        target_fft = tf.cast(tf.signal.rfft2d(target), "float32")
        generated_fft = tf.cast(tf.signal.rfft2d(gen_output), "float32")
        # target_fft = tf.signal.dct(tf.transpose(tf.signal.dct(tf.transpose(target), norm="ortho")), norm="ortho")
        # generated_fft = tf.signal.dct(tf.transpose(tf.signal.dct(tf.transpose(gen_output), norm="ortho")), norm="ortho")
        fft_loss = tf.reduce_mean(tf.square(target_fft - generated_fft))
        
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss) + (self.LAMBDA_FFT * fft_loss)

        return total_gen_loss, gan_loss, l1_loss, fft_loss
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, input_image, target_image, step, UPDATE_STEPS):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target_image], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss, gen_fft_loss = self.generator_loss(disc_generated_output, gen_output, target_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//UPDATE_STEPS)
            tf.summary.scalar("gen_fft_loss", gen_fft_loss, step=step//UPDATE_STEPS)            
            tf.summary.scalar("disc_loss", disc_loss, step=step//UPDATE_STEPS)
        
        
        
        
class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

# clip model weights to hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)


class Pix2PixWassersteinModel(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="pix2pix-wgan", LAMBDA=100):
        super().__init__(train_ds, test_ds, model_name, architecture_name, LAMBDA=LAMBDA)
        self.weight_clipper = ClipConstraint(0.01)
        self.discriminator = self.Critic()
        self.loss_object = WassersteinLoss()
        self.generator_optimizer = tf.keras.optimizers.RMSprop(0.0002)
        self.discrimintator_optimizer = tf.keras.optimizers.RMSprop(0.0002)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_dir, max_to_keep=5)
        self.invert_discriminator_value = True


    def Critic(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        input = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
        target = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

        # (batch_size, 64, 64, channels*2)
        x = layers.concatenate([input, target])

        down = Pix2PixModel.downsample(64, 4, False, self.weight_clipper)(x)  # (batch_size, 32, 32, 64)
        zero_pad1 = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)

        conv = layers.Conv2D(128, 4, strides=1,
                             kernel_initializer=initializer,
                             kernel_constraint=self.weight_clipper,
                             use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 128)
        # batchnorm1 = layers.BatchNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(conv)

        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 128)
        last = layers.Conv2D(1, 4, strides=1,
                             kernel_initializer=initializer,
                             kernel_constraint=self.weight_clipper)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[input, target], outputs=last)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output) * -1.0, disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        # l2_loss = tf.reduce_mean(tf.square(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, critic_real_output, critic_generated_output):
        real_loss = self.loss_object(tf.ones_like(critic_real_output)*-1.0, critic_real_output)
        generated_loss = self.loss_object(tf.ones_like(critic_generated_output), critic_generated_output)
        total_critic_loss = real_loss + generated_loss

        return total_critic_loss

    
    # @tf.function(experimental_relax_shapes=True)
    # def train_step(self, input_image, target, step, UPDATE_STEPS):
    #     with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
    #         gen_output = self.generator(input_image, training=True)

    #         critic_real