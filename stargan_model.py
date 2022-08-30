import numpy as np

from IPython import display
from matplotlib import pyplot as plt

import io_utils
from side2side_model import S2SModel
from networks import *


class StarGANModel(S2SModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name, discriminator_type="stargan",
                 generator_type="stargan", lambda_gp=10., lambda_domain=1., lambda_reconstruction=10.,
                 discriminator_steps=5):
        super().__init__(train_ds, test_ds, model_name, architecture_name)

        self.lambda_gp = lambda_gp
        self.lambda_domain = lambda_domain
        self.lambda_reconstruction = lambda_reconstruction
        self.discriminator_steps = discriminator_steps

        self.generator = self.create_generator(generator_type)
        self.discriminator = self.create_discriminator(discriminator_type)
        self.domain_classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_dir, max_to_keep=5)

    def create_generator(self, generator_type):
        if generator_type == "stargan":
            return StarGANGenerator()
        elif generator_type == "unet":
            return StarGANUnetGenerator()
        else:
            raise ValueError(f"The provided {generator_type} type for generator has not been implemented.")

    def create_discriminator(self, discriminator_type):
        if discriminator_type == "stargan":
            return StarGANDiscriminator()
        else:
            raise ValueError(f"The provided {discriminator_type} type for discriminator has not been implemented.")

    def generator_loss(self, critic_fake_patches, critic_fake_domain, fake_domain, fake_image, reconstructed_image):
        fake_loss = -tf.reduce_mean(critic_fake_patches)
        fake_domain_loss = self.lambda_domain * self.domain_classification_loss(fake_domain, critic_fake_domain)
        reconstruction_loss = self.lambda_reconstruction * tf.reduce_mean(tf.abs(reconstructed_image - fake_image))

        total_loss = fake_loss + fake_domain_loss + reconstruction_loss
        return total_loss, fake_loss, fake_domain_loss, reconstruction_loss

    def discriminator_loss(self, critic_real_patches, critic_real_domain, real_domain, critic_fake_patches,
                           gradient_penalty):
        real_loss = -tf.reduce_mean(critic_real_patches)
        fake_loss = tf.reduce_mean(critic_fake_patches)
        real_domain_loss = self.lambda_domain * self.domain_classification_loss(real_domain, critic_real_domain)
        gp_regularization = self.lambda_gp * gradient_penalty

        total_loss = fake_loss + real_loss + real_domain_loss + gp_regularization
        return total_loss, real_loss, fake_loss, real_domain_loss, gp_regularization

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        random_domain_selector = lambda image, label: (image, label, io_utils.random_domain_label(1))
        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1).map(random_domain_selector)
        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1).map(random_domain_selector)

        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        real_dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (real_image, label) in real_dataset.enumerate():
            random_domain = io_utils.random_domain_label(1)
            image_and_label = io_utils.concat_image_and_domain_label(real_image, random_domain)

            fake_image = self.generator(image_and_label, training=True)
            real_images[i] = tf.squeeze(real_image).numpy()
            fake_images[i] = tf.squeeze(fake_image).numpy()

        return real_images, fake_images

    @tf.function
    def train_step(self, batch, step, update_steps):
        real_image, real_domain = batch

        # TREINANDO O DISCRIMINADOR
        # =========================
        #
        batch_size = tf.shape(batch[0])[0]
        gp_epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        domain_target_onehot = io_utils.random_domain_label(batch_size)
        image_and_label = io_utils.concat_image_and_domain_label(real_image, domain_target_onehot)

        with tf.GradientTape() as disc_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = self.generator(image_and_label, training=True)
                fake_image_mixed = gp_epsilon * real_image + (1 - gp_epsilon) * fake_image
                fake_mixed_predicted, _ = self.discriminator(fake_image_mixed, training=True)

            # computando o gradient penalty
            gp_grads = gp_tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

            # passando imagens reais e fake pelo crítico
            real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)

            c_loss = self.discriminator_loss(real_predicted_patches, real_predicted_domain, real_domain,
                                             fake_predicted_patches, gradient_penalty)
            critic_total_loss, critic_real_loss, critic_fake_loss, critic_real_domain_loss, gp_regularization = c_loss

        discriminator_gradients = disc_tape.gradient(critic_total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", critic_total_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", critic_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", critic_fake_loss, step=step // update_steps)
                tf.summary.scalar("real_domain_loss", critic_real_domain_loss, step=step // update_steps)
                tf.summary.scalar("gradient_penalty", gp_regularization, step=step // update_steps)

        # TREINANDO O GERADOR
        # ===================
        #
        if step % self.discriminator_steps == 0:
            domain_target_onehot = io_utils.random_domain_label(batch_size)
            image_and_label = io_utils.concat_image_and_domain_label(real_image, domain_target_onehot)
            with tf.GradientTape() as gen_tape:
                fake_image = self.generator(image_and_label, training=True)
                fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)

                # reconstruct the image to the original domain
                image_and_label = io_utils.concat_image_and_domain_label(fake_image, real_domain)
                reconstructed_image = self.generator(image_and_label)

                g_loss = self.generator_loss(fake_predicted_patches, fake_predicted_domain, domain_target_onehot,
                                             real_image, reconstructed_image)
                generator_total_loss, generator_adversarial_loss, generator_domain_loss, generator_reconstruction_loss = g_loss

            generator_gradients = gen_tape.gradient(generator_total_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            with tf.name_scope("generator"):
                with self.summary_writer.as_default():
                    tf.summary.scalar("total_loss", generator_total_loss, step=step // update_steps)
                    tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                    tf.summary.scalar("domain_loss", generator_domain_loss, step=step // update_steps)
                    tf.summary.scalar("reconstruction_loss", generator_reconstruction_loss, step=step // update_steps)

    # chama o gerador em examples e gera imagem de 3 colunas: original, gerada em outro domínio, reconstruída
    def generate_comparison(self, examples, save_name=None, step=None):
        num_images = len(examples)
        num_columns = 3

        title = ["Input", "Generated", "Reconstructed"]
        if step is not None:
            if step == 1:
                step = 0
            title[-1] += f" ({step / 1000}k)"
            title[-2] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        for i, (real_image, real_domain, target_domain) in enumerate(examples):
            target_domain_side_index = tf.argmax(target_domain[0], axis=0)
            target_domain_name = DIRECTION_FOLDERS[
                target_domain_side_index]  # tf.gather(DIRECTION_FOLDERS, target_domain_side_index)

            image_and_label = io_utils.concat_image_and_domain_label(real_image, target_domain)
            generated_image = self.generator(image_and_label, training=True)

            image_and_label = io_utils.concat_image_and_domain_label(generated_image, real_domain)
            reconstructed_image = self.generator(image_and_label, training=True)

            images = [real_image, generated_image, reconstructed_image]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                if i == 0:
                    plt.title(f"{title[j]}\n{target_domain_name}" if j == 1 else title[j] + "\n",
                              fontdict={"fontsize": 24})
                elif j == 1:
                    plt.title(target_domain_name, fontdict={"fontsize": 24})
                plt.imshow(tf.squeeze(images[j]) * 0.5 + 0.5)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        display.display(figure)
        # plt.show()

        return figure

    def get_predicted_real_and_fake(self, batch_of_one):
        real_image, source_label = batch_of_one
        target_label = io_utils.random_domain_label(1)
        image_and_label = io_utils.concat_image_and_domain_label(real_image, target_label)
        fake_image = self.generator(image_and_label, training=True)

        real_predicted, _ = self.discriminator(real_image)
        fake_predicted, _ = self.discriminator(fake_image)
        real_predicted = real_predicted[0]
        fake_predicted = fake_predicted[0]

        return real_image, fake_image, real_predicted, fake_predicted

    def show_discriminated_image(self, batch_of_one):
        # generates the fake image and the discriminations of the real and fake
        real_image, fake_image, real_predicted, fake_predicted = self.get_predicted_real_and_fake(batch_of_one)

        # wgan associates negative numbers to real images and positive to fake
        # but we need to provide them in the [0, 1] range
        concatenated_predictions = tf.concat([real_predicted, fake_predicted], axis=-1)
        min_value = tf.math.reduce_min(concatenated_predictions)
        max_value = tf.math.reduce_max(concatenated_predictions)
        amplitude = max_value - min_value
        real_predicted = (real_predicted - min_value) / amplitude
        fake_predicted = (fake_predicted - min_value) / amplitude

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

        # display the images: real / discr. real / fake / discr. fake
        figure = plt.figure(figsize=(6 * 4, 6 * 1))
        plt.subplot(1, 4, 1)
        plt.title("Label", fontdict={"fontsize": 20})
        plt.imshow(real_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Discriminated label", fontdict={"fontsize": 20})
        plt.imshow(real_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Generated", fontdict={"fontsize": 20})
        plt.imshow(fake_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Discriminated generated", fontdict={"fontsize": 20})
        plt.imshow(fake_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()


class PairedStarGANModel(StarGANModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name, discriminator_type="twopairedstargan",
                 generator_type="stargan", lambda_gp=10., lambda_domain=1., lambda_reconstruction=10., lambda_l1=100.,
                 discriminator_steps=5):
        super().__init__(train_ds, test_ds, model_name, architecture_name, discriminator_type, generator_type,
                         lambda_gp, lambda_domain, lambda_reconstruction, discriminator_steps)
        self.lambda_l1 = lambda_l1

    def create_discriminator(self, discriminator_type="twopairedstargan"):
        if discriminator_type == "twopairedstargan":
            return TwoPairedStarGANDiscriminator()
        else:
            raise ValueError(f"The provided {discriminator_type} type for generator has not been implemented.")

    def generator_loss(self, critic_fake_patches, critic_fake_domain, fake_domain, fake_image, reconstructed_image,
                       target_image, source_image):
        fake_loss = -tf.reduce_mean(critic_fake_patches)
        fake_domain_loss = self.domain_classification_loss(fake_domain, critic_fake_domain)
        reconstruction_loss = tf.reduce_mean(tf.abs(source_image - reconstructed_image))
        l1_loss = tf.reduce_mean(tf.abs(target_image - fake_image))

        total_loss = fake_loss + \
            self.lambda_domain * fake_domain_loss +\
            self.lambda_reconstruction * reconstruction_loss +\
            self.lambda_l1 * l1_loss
        return total_loss, fake_loss, fake_domain_loss, reconstruction_loss, l1_loss

    def discriminator_loss(self, critic_real_patches, critic_real_domain, real_domain, critic_fake_patches,
                           gradient_penalty):
        real_loss = -tf.reduce_mean(critic_real_patches)
        fake_loss = tf.reduce_mean(critic_fake_patches)
        real_domain_loss = self.domain_classification_loss(real_domain, critic_real_domain)

        total_loss = fake_loss + real_loss + \
            self.lambda_domain * real_domain_loss +\
            self.lambda_gp * gradient_penalty
        return total_loss, real_loss, fake_loss, real_domain_loss, gradient_penalty

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)
        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)

        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        real_dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source, target) in real_dataset.enumerate():
            source_image, _ = source
            target_image, target_domain = target

            fake_image = self.generator(io_utils.concat_image_and_domain_label(source_image, target_domain),
                                        training=True)
            real_images[i] = tf.squeeze(target_image).numpy()
            fake_images[i] = tf.squeeze(fake_image).numpy()

        return real_images, fake_images

    @tf.function
    def train_step(self, batch, step, update_steps):
        #         (back_image, back_domain), (left_image, left_domain), (front_image, front_domain), (right_image, right_domain) = batch

        #         random_source_index = tf.random.uniform(shape=[BATCH_SIZE,], minval=0, maxval=NUMBER_OF_DOMAINS, dtype=tf.int32)
        #         source_image, source_domain = tf.gather(batch, random_source_index)

        #         random_target_index = tf.random.uniform(shape=[BATCH_SIZE,], minval=0, maxval=NUMBER_OF_DOMAINS, dtype=tf.int32)
        #         target_image, target_domain = tf.gather(batch, random_target_index)
        # source, target = io_utils.random_pair(batch)
        # source_image, souce_domain = source
        # target_image, target_domain = target
        source, target = batch
        source_image, source_domain = source
        target_image, target_domain = target

        batch_size = tf.shape(source_image)[0]

        # TREINANDO O DISCRIMINADOR
        # =========================
        #
        gp_epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        image_and_label = io_utils.concat_image_and_domain_label(source_image, target_domain)

        with tf.GradientTape() as disc_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = self.generator(image_and_label, training=True)
                fake_image_mixed = gp_epsilon * target_image + (1 - gp_epsilon) * fake_image
                fake_mixed_predicted, _ = self.discriminator([fake_image_mixed, source_image], training=True)

            # computando o gradient penalty
            gp_grads = gp_tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

            # passando imagens reais e fake pelo crítico
            real_predicted_patches, real_predicted_domain = self.discriminator([target_image, source_image],
                                                                               training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator([fake_image, source_image],
                                                                               training=True)

            c_loss = self.discriminator_loss(real_predicted_patches, real_predicted_domain, target_domain,
                                             fake_predicted_patches, gradient_penalty)
            critic_total_loss, critic_real_loss, critic_fake_loss, critic_real_domain_loss, gp_regularization = c_loss

        discriminator_gradients = disc_tape.gradient(critic_total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", critic_total_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", critic_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", critic_fake_loss, step=step // update_steps)
                tf.summary.scalar("real_domain_loss", critic_real_domain_loss, step=step // update_steps)
                tf.summary.scalar("gradient_penalty", gp_regularization, step=step // update_steps)

        # TREINANDO O GERADOR
        # ===================
        #
        if step % self.discriminator_steps == 0:
            with tf.GradientTape() as gen_tape:
                image_and_label_forward = io_utils.concat_image_and_domain_label(source_image, target_domain)
                fake_image = self.generator(image_and_label_forward, training=True)

                # reconstruct the image to the original domain
                image_and_label_backward = io_utils.concat_image_and_domain_label(fake_image, source_domain)
                reconstructed_image = self.generator(image_and_label_backward, training=True)

                fake_predicted_patches, fake_predicted_domain = self.discriminator([fake_image, source_image],
                                                                                   training=True)

                g_loss = self.generator_loss(fake_predicted_patches, fake_predicted_domain, target_domain, source_image,
                                             reconstructed_image, target_image, source_image)
                generator_loss, generator_adversarial_loss, generator_domain_loss, generator_reconstruction_loss, generator_l1_loss = g_loss

            generator_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            with tf.name_scope("generator"):
                with self.summary_writer.as_default():
                    tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                    tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                    tf.summary.scalar("domain_loss", generator_domain_loss, step=step // update_steps)
                    tf.summary.scalar("reconstruction_loss", generator_reconstruction_loss, step=step // update_steps)
                    tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)

    def generate_comparison(self, examples, save_name=None, step=None):
        num_images = len(examples)
        num_columns = 4

        title = ["Input", "Reconstructed", "Generated", "Target"]
        if step is not None:
            if step == 1:
                step = 0
            title[-2] += f" ({step / 1000}k)"
            title[-3] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        for i, ((source_image, source_domain), (target_image, target_domain)) in enumerate(examples):
            target_domain_side_index = tf.argmax(target_domain[0], axis=0)
            target_domain_name = DIRECTION_FOLDERS[target_domain_side_index]

            image_and_label_forward = io_utils.concat_image_and_domain_label(source_image, target_domain)
            generated_image = self.generator(image_and_label_forward, training=True)

            image_and_label_backward = io_utils.concat_image_and_domain_label(generated_image, source_domain)
            reconstructed_image = self.generator(image_and_label_backward, training=True)

            images = [source_image, reconstructed_image, generated_image, target_image]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                if i == 0:
                    plt.title(f"{title[j]}\n{target_domain_name}" if j == 2 else title[j] + "\n",
                              fontdict={"fontsize": 24})
                elif j == 2:
                    plt.title(target_domain_name, fontdict={"fontsize": 24})
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

    def get_predicted_real_and_fake(self, batch_of_one):
        source, target = batch_of_one
        source_image, source_domain = source
        target_image, target_domain = target
        image_and_label = io_utils.concat_image_and_domain_label(source_image, target_domain)
        fake_image = self.generator(image_and_label, training=True)

        real_predicted, _ = self.discriminator([target_image, source_image])
        fake_predicted, _ = self.discriminator([fake_image, source_image])
        real_predicted = real_predicted[0]
        fake_predicted = fake_predicted[0]

        return target_image, fake_image, real_predicted, fake_predicted

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))


class CollaGANModel(PairedStarGANModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name, discriminator_type,
                 generator_type, lambda_gp=10., lambda_domain=1., lambda_reconstruction=10., lambda_l1=100.,
                 discriminator_steps=5):
        super().__init__(train_ds, test_ds, model_name, architecture_name, discriminator_type, generator_type,
                         lambda_gp, lambda_domain, lambda_reconstruction, lambda_l1, discriminator_steps)

    def create_discriminator(self, discriminator_type):
        if discriminator_type == "collagan":
            return CollaGANDiscriminator()
        else:
            raise ValueError(f"The provided {discriminator_type} type for generator has not been implemented.")

    def create_generator(self, generator_type):
        if generator_type == "collagan":
            return CollaGANGenerator()
        else:
            raise ValueError(f"The provided {generator_type} type for generator has not been implemented.")


# modifica StarGAN para receber um número variado de inputs de domínios etiquetados
# na tentativa de melhorar o resultado de tradução já que mais informação está sendo
# fornecida
class FegeGAN():
    def __init__(self):
        pass
