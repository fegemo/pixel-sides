import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from abc import abstractmethod

import histogram
import io_utils
from networks import *
from side2side_model import S2SModel, WassersteinLoss


class Pix2PixModel(S2SModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, mode,
                 discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name)

        default_kwargs = {"num_patches": 30}
        kwargs = {**default_kwargs, **kwargs}

        self.mode = mode
        self.lambda_l1 = lambda_l1
        self.lambda_gp = lambda_gp
        self.lambda_histogram = lambda_histogram
        self.discriminator_steps = discriminator_steps

        self.generator = self.create_generator(generator_type)
        self.discriminator = self.create_discriminator(discriminator_type, **kwargs)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_object = self.define_loss_object()

        generator_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in self.generator.trainable_weights])
        discriminator_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in self.discriminator.trainable_weights])

        print(f"Generator: {self.generator.name} with {generator_params:,} parameters")
        print(f"Discriminator: {self.discriminator.name} with {discriminator_params:,} parameters")
        print(f"Mode {self.mode} with {self.loss_object.name} loss")

        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,
                                                             max_to_keep=5)

    @staticmethod
    def build(train_ds, test_ds, model_name, architecture_name,
              discriminator_type, generator_type, mode="gan", discriminator_steps=1,
              lambda_l1=100., lambda_gp=10., lambda_histogram=1., **kwargs):
        constructor = None
        if discriminator_steps > 1:
            constructor = Pix2PixModelMultipleSteps
        elif discriminator_steps == 1:
            if mode == "gan":
                constructor = Pix2PixModelSingleStepWithoutGP
            else:
                constructor = Pix2PixModelSingleStep
        else:
            raise ValueError(
                f"Tried to build a Pix2PixModel, but the user provided {discriminator_steps} discriminator steps")
        return constructor(
            train_ds, test_ds,
            model_name, architecture_name,
            discriminator_type, generator_type,
            mode, discriminator_steps,
            lambda_l1, lambda_gp, lambda_histogram,
            **kwargs)

    @abstractmethod
    def train_step(self, batch, step, update_steps):
        pass

    def define_loss_object(self):
        if self.mode == "gan":
            return self.bce
        else:
            return WassersteinLoss()

    def create_discriminator(self, discriminator_type, **kwargs):
        if discriminator_type == "patch":
            if "num_patches" not in kwargs:
                raise ValueError(
                    f"The 'num_patches' kw argument should have been passed to create_discriminator,"
                    f"but it was not. kwargs: {kwargs}")
            return PatchDiscriminator(kwargs["num_patches"])
        elif discriminator_type == "patch-resnet":
            return PatchResnetDiscriminator()
        elif discriminator_type == "deeper":
            return Deeper2x2PatchDiscriminator()
        elif discriminator_type == "u-net" or discriminator_type == "unet":
            return UnetDiscriminator()
        elif discriminator_type == "segnet":
            return SegnetDiscriminator()
        elif discriminator_type == "atrous":
            return AtrousDiscriminator()
        elif discriminator_type == "indexed-patch":
            return IndexedPatchDiscriminator(kwargs["num_patches"])
        elif discriminator_type == "indexed-patch-resnet":
            return IndexedPatchResnetDiscriminator()
        else:
            raise NotImplementedError(f"The {discriminator_type} type of discriminator has not been implemented")

    def create_generator(self, generator_type, **kwargs):
        if generator_type == "u-net" or generator_type == "unet":
            return UnetGenerator()
        elif generator_type == "u-net2" or generator_type == "unet2":
            return Unet2Generator()
        elif generator_type == "segnet":
            return SegnetGenerator()
        elif generator_type == "atrous":
            raise NotImplementedError(f"The {generator_type} type of generator has not been implemented")
        elif generator_type == "indexed-unet":
            return IndexedUnetGenerator()
        else:
            raise NotImplementedError(f"The {generator_type} type of generator has not been implemented")

    def generator_loss(self, fake_predicted, fake_image, real_image, training_t):
        # fake_predicted, fake_classification = fake_predicted
        # adversarial_classification_loss = self.bce(tf.ones_like(fake_classification), fake_classification)

        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        # desired_fake_prediction = self.calculate_desired_fake_prediction(fake_image, real_image, fake_predicted)
        # adversarial_loss = 1. - self.loss_object(desired_fake_prediction, fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))

        # real_histogram = histogram.calculate_rgbuv_histogram(real_image)
        # fake_histogram = histogram.calculate_rgbuv_histogram(fake_image)
        # histogram_loss = histogram.hellinger_loss(real_histogram, fake_histogram)
        histogram_loss = tf.constant(0., "float32")
        # total_variation_loss = tf.reduce_sum(tf.image.total_variation(fake_image))
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss) + (self.lambda_histogram * histogram_loss)
        # total_loss = adversarial_loss + adversarial_classification_loss + (self.lambda_l1 * l1_loss) + (self.lambda_histogram * histogram_loss)

        return total_loss, adversarial_loss, l1_loss, histogram_loss
        # return total_loss, adversarial_loss, l1_loss, histogram_loss, adversarial_classification_loss

    def discriminator_loss(self, real_predicted, fake_predicted,
                           fake_image, real_image,
                           gradient_penalty=tf.constant(0.)):
        # for u-net discriminator, uncomment:
        # real_predicted, real_classification = real_predicted
        # fake_predicted, fake_classification = fake_predicted
        # real_classification_loss = self.bce(tf.ones_like(real_classification), real_classification)
        # fake_classification_loss = self.bce(tf.zeros_like(fake_classification), fake_classification)
        # classification_loss = real_classification_loss + fake_classification_loss
        # real_classification_loss = tf.constant(0.)
        # fake_classification_loss = tf.constant(0.)
        # classification_loss = tf.constant(0.)

        real_loss = self.loss_object(tf.ones_like(real_predicted), real_predicted)
        fake_segmentation_mask = self.calculate_desired_fake_prediction(fake_image, real_image, fake_predicted)
        fake_loss = self.loss_object(fake_segmentation_mask, fake_predicted)
        total_loss = fake_loss + real_loss + self.lambda_gp * gradient_penalty

        return total_loss, real_loss, fake_loss, gradient_penalty
        # return total_loss, real_loss, fake_loss, real_classification_loss, fake_classification_loss, classification_loss, gradient_penalty

    @tf.function
    def calculate_gradient_penalty(self, source_image, real_image, fake_image):
        gradient_penalty = tf.constant(0.)
        batch_size = tf.shape(real_image)[0]

        if self.mode == "wgan":
            alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
            fake_image_mixed = alpha * real_image + (1 - alpha) * fake_image

            with tf.GradientTape() as tape:
                tape.watch(fake_image_mixed)
                fake_mixed_predicted = self.discriminator([fake_image_mixed, source_image], training=True)

            # computando o gradient penalty
            gp_grads = tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

        elif self.mode == "dragan":
            raise ValueError(f"{self.mode} gradient penalty has not been implemented.")

        elif self.mode == "gan":
            pass

        return gradient_penalty

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)
        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)

        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image) in dataset.enumerate():
            fake_image = self.generator(source_image, training=True)
            real_images[i] = tf.squeeze(real_image).numpy()
            fake_images[i] = tf.squeeze(fake_image).numpy()

        return real_images, fake_images

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def generate_comparison(self, examples, save_name=None, step=None, predicted_images=None):
        # invoca o gerador e mostra a imagem de entrada, sua imagem objetivo e a imagem gerada
        title = ["Input", "Target", "Generated"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            title[-1] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        if predicted_images is None:
            predicted_images = []

        for i, (source_image, target_image) in enumerate(examples):
            if i >= len(predicted_images):
                predicted_images.append(self.generator(source_image, training=True))

            images = [source_image, target_image, predicted_images[i]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(images[j][0] * 0.5 + 0.5)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        display.display(figure)
        # plt.show()

        return figure

    def show_discriminated_image(self, batch_of_one):
        # generates the fake image and the discriminations of the real and fake
        source_image, real_image = batch_of_one
        fake_image = self.generator(source_image, training=True)

        real_predicted = self.discriminator([real_image, source_image])
        fake_predicted = self.discriminator([fake_image, source_image])
        # for u-net discriminator, uncomment:
        # real_predicted = self.discriminator([real_image, source_image])[0]
        # fake_predicted = self.discriminator([fake_image, source_image])[0]
        desired_fake_prediction = self.calculate_desired_fake_prediction(fake_image, real_image, fake_predicted)
        real_predicted = real_predicted[0]
        fake_predicted = fake_predicted[0]

        if self.mode == "wgan":
            # wgan associates negative numbers to real images and positive to fake
            # but we need to provide them in the [0, 1] range
            concatenated_predictions = tf.concat([real_predicted, fake_predicted], axis=-1)
            min_value = tf.math.reduce_min(concatenated_predictions)
            max_value = tf.math.reduce_max(concatenated_predictions)
            amplitude = max_value - min_value
            real_predicted = (real_predicted - min_value) / amplitude
            fake_predicted = (fake_predicted - min_value) / amplitude
        else:
            real_predicted = tf.math.sigmoid(real_predicted)
            fake_predicted = tf.math.sigmoid(fake_predicted)

        # finds the mean value of the patches (to display on the titles)
        real_predicted_mean = tf.reduce_mean(real_predicted)
        fake_predicted_mean = tf.reduce_mean(fake_predicted)
        desired_fake_prediction_mean = tf.reduce_mean(desired_fake_prediction)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[0]
        lower_bound_scaling_factor = IMG_SIZE // num_patches
        pad_before = (IMG_SIZE - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (IMG_SIZE - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        real_predicted = tf.pad(real_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        source_image = source_image[0]
        desired_fake_prediction = desired_fake_prediction[0]

        # display the images: source / real / discr. real / fake / discr. fake / desired discr. fake
        plt.figure(figsize=(6 * 6, 6 * 1))
        plt.subplot(1, 6, 1)
        plt.title("Source", fontdict={"fontsize": 20})
        plt.imshow(source_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 2)
        plt.title("Target", fontdict={"fontsize": 20})
        plt.imshow(real_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 3)
        plt.title(tf.strings.join([
            "Discriminated label ",
            tf.strings.as_string(real_predicted_mean, precision=3)]).numpy().decode("UTF-8"), fontdict={"fontsize": 20})
        plt.imshow(real_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 6, 4)
        plt.title("Generated", fontdict={"fontsize": 20})
        plt.imshow(fake_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 5)
        plt.title(tf.strings.reduce_join([
            "Discriminated generated ",
            tf.strings.as_string(fake_predicted_mean, precision=3)]).numpy().decode("UTF-8"), fontdict={"fontsize": 20})
        plt.imshow(fake_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 6, 6)
        plt.title(tf.strings.reduce_join([
            "Desired disc. output ",
            tf.strings.as_string(desired_fake_prediction_mean, precision=3)]).numpy().decode("UTF-8"), fontdict={"fontsize": 20})
        plt.imshow(desired_fake_prediction, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()

    def calculate_desired_fake_prediction(self, fake_image, real_image, fake_predicted):
        # fake_segmentation_mask = 1. - tf.reduce_sum(tf.abs(real_image - fake_image), -1)[..., tf.newaxis] * 0.125
        # fake_segmentation_mask = tf.where(fake_segmentation_mask < 0.98, 0., 1.)
        # fake_segmentation_mask = tf.image.resize(fake_segmentation_mask, tf.shape(fake_predicted)[1:3], method="nearest")
        fake_segmentation_mask = tf.zeros_like(fake_predicted)
        return fake_segmentation_mask


# Training loop with support for discriminator_steps > 1 and gradient penalties
class Pix2PixModelMultipleSteps(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, mode,
                 discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name,
                         discriminator_type, generator_type, mode,
                         discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs)

    @tf.function
    def train_step(self, batch, step, update_steps):
        source_image, real_image = batch

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            # Training the DISCRIMINATOR
            gradient_penalty = self.calculate_gradient_penalty(source_image, real_image, fake_image)
            d_loss = self.discriminator_loss(real_predicted, fake_predicted, gradient_penalty)
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss, discriminator_gp_loss = d_loss

            with tape.stop_recording():
                discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(
                    zip(discriminator_gradients, self.discriminator.trainable_variables))
                with self.summary_writer.as_default():
                    with tf.name_scope("discriminator"):
                        tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                        tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                        tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
                        tf.summary.scalar("gp_loss", discriminator_gp_loss, step=step // update_steps)

            # Training the GENERATOR
            if step % self.discriminator_steps == 0:
                g_loss = self.generator_loss(fake_predicted, fake_image, real_image)
                generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss = g_loss

                with tape.stop_recording():
                    generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                    self.generator_optimizer.apply_gradients(
                        zip(generator_gradients, self.generator.trainable_variables))
                    with self.summary_writer.as_default():
                        with tf.name_scope("generator"):
                            tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                            tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                            tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                            tf.summary.scalar("histogram_loss", generator_histogram_loss, step=step // update_steps)


# Training loop when discriminator_steps == 1 (can calculate gradient penalty)
class Pix2PixModelSingleStep(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, mode,
                 discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name,
                         discriminator_type, generator_type, mode,
                         discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs)

    @tf.function
    def train_step(self, batch, step, update_steps):
        source_image, real_image = batch

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image, real_image)
            # generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss = g_loss
            generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss, gen_cls_loss = g_loss

            gradient_penalty = self.calculate_gradient_penalty(source_image, real_image, fake_image)
            d_loss = self.discriminator_loss(real_predicted, fake_predicted, fake_image, real_image, gradient_penalty)
            # discriminator_loss, discriminator_real_loss, discriminator_fake_loss, discriminator_gp_loss = d_loss
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss, discriminator_gp_loss, disc_fake_cls_loss, disc_real_cls_loss = d_loss

        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("discriminator"):
                tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
                tf.summary.scalar("gp_loss", discriminator_gp_loss, step=step // update_steps)
                tf.summary.scalar("fake_classification_loss", disc_fake_cls_loss, step=step // update_steps)
                tf.summary.scalar("real_classification_loss", disc_real_cls_loss, step=step // update_steps)

            with tf.name_scope("generator"):
                tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                tf.summary.scalar("histogram_loss", generator_histogram_loss, step=step // update_steps)
                tf.summary.scalar("classification_loss", gen_cls_loss, step=step // update_steps)


# Traditional training loop, without WGAN concepts (cost and gradient penalty)
class Pix2PixModelSingleStepWithoutGP(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, mode,
                 discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name,
                         discriminator_type, generator_type, mode,
                         discriminator_steps, lambda_l1, lambda_gp, lambda_histogram, **kwargs)

    @tf.function
    def train_step(self, batch, step, update_steps):
        source_image, real_image = batch

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image, real_image, tf.cast(step, "float32")/10000.)
            generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss = g_loss
            # generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss, gen_cls_loss = g_loss

            d_loss = self.discriminator_loss(real_predicted, fake_predicted, fake_image, real_image)
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss, _ = d_loss
            # discriminator_loss, discriminator_real_loss, discriminator_fake_loss, disc_real_cls_loss, disc_fake_cls_loss, disc_cls_loss, _  = d_loss
            # real_patches, real_cls = real_predicted
            # fake_patches, fake_cls = fake_predicted
            #
            # tf.print("tf.shape(real_cls)", tf.shape(real_cls))
            # tf.print("real_cls", real_cls)
            # disc_real_cls_loss = self.bce(tf.ones_like(real_cls), real_cls)
            # disc_fake_cls_loss = self.bce(tf.zeros_like(fake_cls), fake_cls)
            # disc_cls_loss = disc_real_cls_loss + disc_fake_cls_loss
            # discriminator_loss = discriminator_real_loss = discriminator_fake_loss = tf.constant(0.)

        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        # generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        # discriminator_gradients = tape.gradient([discriminator_loss, disc_cls_loss],
        #                                         self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("discriminator"):
                tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
                # tf.summary.scalar("fake_classification_loss", disc_fake_cls_loss, step=step // update_steps)
                # tf.summary.scalar("real_classification_loss", disc_real_cls_loss, step=step // update_steps)
            with tf.name_scope("generator"):
                tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                tf.summary.scalar("histogram_loss", generator_histogram_loss, step=step // update_steps)
                # tf.summary.scalar("classification_loss", gen_cls_loss, step=step // update_steps)


class Pix2PixIndexedModel(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, mode="gan",
                 discriminator_steps=1, lambda_l1=100., lambda_gp=10., lambda_segmentation=0.5, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name,
                         discriminator_type, generator_type, mode,
                         discriminator_steps, lambda_l1, lambda_gp, **kwargs)
        self.lambda_segmentation = lambda_segmentation
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def generator_loss(self, fake_predicted, fake_image, real_image):
        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        segmentation_loss = self.generator_loss_object(real_image, fake_image)
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss) + (self.lambda_segmentation * segmentation_loss)

        return total_loss, adversarial_loss, l1_loss, segmentation_loss

    def discriminator_loss(self, real_predicted, fake_predicted, gradient_penalty=tf.constant(0.)):
        real_loss = self.loss_object(tf.ones_like(real_predicted), real_predicted)
        fake_loss = self.loss_object(tf.zeros_like(fake_predicted), fake_predicted)
        gp_loss = self.lambda_gp * gradient_penalty
        total_loss = real_loss + fake_loss + gp_loss

        return total_loss, real_loss, fake_loss, gp_loss

    def train_step(self, batch, step, update_steps):
        source_image, real_image, palette = batch
        batch_size = tf.shape(source_image)[0]

        real_image_one_hot = tf.reshape(tf.one_hot(real_image, MAX_PALETTE_SIZE, axis=-1), [batch_size, IMG_SIZE, IMG_SIZE, -1])
        # tf.print("tf.shape(real_image_one_hot)", tf.shape(real_image_one_hot))

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)
            # tf.print("tf.shape(fake_image)", tf.shape(fake_image))
            fake_image_for_discriminator = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image_for_discriminator, source_image], training=True)

            generator_loss, generator_adversarial_loss, generator_l1_loss, generator_segmentation_loss = \
                self.generator_loss(fake_predicted, fake_image, real_image_one_hot)
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss, _ = self.discriminator_loss(
                real_predicted, fake_predicted)

        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("discriminator"):
                tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
            with tf.name_scope("generator"):
                tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                tf.summary.scalar("segmentation_loss", generator_segmentation_loss, step=step // update_steps)

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)
        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)
        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def generate_comparison(self, examples, save_name=None, step=None, predicted_images=None):
        # invoca o gerador e mostra a imagem de entrada, sua imagem objetivo e a imagem gerada
        num_images = len(examples)
        num_columns = 3

        title = ["Input", "Target", "Generated"]
        if step is not None:
            title[-1] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        if predicted_images is None:
            predicted_images = []

        for i, (source_image, target_image, palette) in enumerate(examples):
            palette = palette[0]

            if i >= len(predicted_images):
                generated_image = self.generator(source_image, training=True)
                # tf.print("tf.shape(generated_image)", tf.shape(generated_image))
                generated_image = tf.expand_dims(tf.argmax(generated_image, axis=-1, output_type="int32"), -1)
                # tf.print("tf.shape(generated_image) after argmax", tf.shape(generated_image))
                predicted_images.append(generated_image)

            images = [source_image, target_image, predicted_images[i]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                image = images[j][0]
                # tf.print("i", i, "j", j)
                # tf.print("tf.shape(image)", tf.shape(image))
                image = io_utils.indexed_to_rgba(image, palette)
                # tf.print("tf.shape(image in rgb)", tf.shape(image))
                plt.imshow(image)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        display.display(figure)
        # plt.show()

        return figure

    def show_discriminated_image(self, batch_of_one):
        # generates the fake image and the discriminations of the real and fake
        source_image, real_image, palette = batch_of_one

        fake_image = self.generator(source_image, training=True)
        fake_image = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

        real_predicted = self.discriminator([real_image, source_image])[0]
        fake_predicted = self.discriminator([fake_image, source_image])[0]

        if self.mode == "wgan":
            # wgan associates negative numbers to real images and positive to fake
            # but we need to provide them in the [0, 1] range
            concatenated_predictions = tf.concat([real_predicted, fake_predicted], axis=-1)
            min_value = tf.math.reduce_min(concatenated_predictions)
            max_value = tf.math.reduce_max(concatenated_predictions)
            amplitude = max_value - min_value
            real_predicted = (real_predicted - min_value) / amplitude
            fake_predicted = (fake_predicted - min_value) / amplitude
        else:
            real_predicted = tf.math.sigmoid(real_predicted)
            fake_predicted = tf.math.sigmoid(fake_predicted)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[0]
        lower_bound_scaling_factor = IMG_SIZE // num_patches
        pad_before = (IMG_SIZE - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (IMG_SIZE - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        real_predicted = tf.pad(real_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        palette = palette[0]

        # looks up the actual colors in the palette
        real_image = io_utils.indexed_to_rgba(real_image, palette)
        fake_image = io_utils.indexed_to_rgba(fake_image, palette)

        # display the images: real / discr. real / fake / discr. fake
        figure = plt.figure(figsize=(6 * 4, 6 * 1))
        plt.subplot(1, 4, 1)
        plt.title("Label", fontdict={"fontsize": 20})
        plt.imshow(real_image, vmin=0, vmax=255)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Discriminated label", fontdict={"fontsize": 20})
        plt.imshow(real_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Generated", fontdict={"fontsize": 20})
        plt.imshow(fake_image, vmin=0, vmax=255)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Discriminated generated", fontdict={"fontsize": 20})
        plt.imshow(fake_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image, palette) in dataset.enumerate():
            fake_image = self.generator(source_image, training=True)
            fake_image = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

            real_image = real_image[0]
            fake_image = fake_image[0]
            palette = palette[0]

            real_image = io_utils.indexed_to_rgba(real_image, palette)
            fake_image = io_utils.indexed_to_rgba(fake_image, palette)

            real_images[i] = real_image.numpy()
            fake_images[i] = fake_image.numpy()

        return real_images, fake_images


# class Pix2PixFFTModel(Pix2PixModel):
#     def __init__(self, train_ds, test_ds, model_name, architecture_name="pix2pix-fft", LAMBDA_L1=100, LAMBDA_FFT=100):
#         super().__init__(train_ds, test_ds, model_name, architecture_name, LAMBDA=LAMBDA_L1)
#         self.LAMBDA_FFT = LAMBDA_FFT

#     def generator_loss(self, disc_generated_output, gen_output, target):
#         gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
#         l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

#         # print(target)
#         # print(tf.shape(target).numpy())
#         # print(gen_output)
#         # print(tf.shape(gen_output).numpy())
#         target = tf.squeeze(target[:, :, :, 0:3])
#         target = tf.image.rgb_to_grayscale(target)

#         gen_output = tf.squeeze(gen_output[:, :, :, 0:3])
#         gen_output = tf.image.rgb_to_grayscale(gen_output)

#         target_fft = tf.cast(tf.signal.rfft2d(target), "float32")
#         generated_fft = tf.cast(tf.signal.rfft2d(gen_output), "float32")
#         # target_fft = tf.signal.dct(tf.transpose(tf.signal.dct(tf.transpose(target), norm="ortho")), norm="ortho")
#         # generated_fft = tf.signal.dct(tf.transpose(tf.signal.dct(tf.transpose(gen_output), norm="ortho")), norm="ortho")
#         fft_loss = tf.reduce_mean(tf.square(target_fft - generated_fft))

#         total_gen_loss = gan_loss + (self.LAMBDA * l1_loss) + (self.LAMBDA_FFT * fft_loss)

#         return total_gen_loss, gan_loss, l1_loss, fft_loss

#     @tf.function(experimental_relax_shapes=True)
#     def train_step(self, input_image, target_image, step, UPDATE_STEPS):
#         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#             gen_output = self.generator(input_image, training=True)

#             disc_real_output = self.discriminator([input_image, target_image], training=True)
#             disc_generated_output = self.discriminator([input_image, gen_output], training=True)

#             gen_total_loss, gen_gan_loss, gen_l1_loss, gen_fft_loss = self.generator_loss(disc_generated_output, gen_output, target_image)
#             disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

#         generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
#         discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

#         self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
#         self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

#         with self.summary_writer.as_default():
#             tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//UPDATE_STEPS)
#             tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//UPDATE_STEPS)
#             tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//UPDATE_STEPS)
#             tf.summary.scalar("gen_fft_loss", gen_fft_loss, step=step//UPDATE_STEPS)            
#             tf.summary.scalar("disc_loss", disc_loss, step=step//UPDATE_STEPS)
