import tensorflow as tf

import io_utils
from configuration import *


# Some images have transparent pixels with colors other than black
# This function turns all transparent pixels to black
# TFJS does this by default, but TF does not
# The TFJS imported model was having bad inference because of this
def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, INPUT_CHANNELS)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


# replaces the alpha channel with a white color (only 100% transparent pixels)
def replace_alpha_with_white(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, INPUT_CHANNELS)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        255.,
        image)

    # drops the A in RGBA
    image = image[:, :, :3]
    return image


def normalize(image):
    """
    Turns an image from the [0, 255] range into [-1, 1], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [-1, 1] range
    -------
    """
    return (image / 127.5) - 1


def denormalize(image):
    """
    Turns an image from the [-1, 1] range into [0, 255], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [0, 255] range
    -------
    """
    return (image + 1) * 127.5


def rotate_hue(image, angle):
    return tf.image.adjust_hue(image, angle)


# loads an image from the file system and transforms it for the network:
# (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
# puts the values in the range of [-1, 1]
def load_image(path, hue_angle, should_augment, should_normalize=True):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=INPUT_CHANNELS)
    image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS))
    image = tf.cast(image, "float32")
    image = blacken_transparent_pixels(image)
    if OUTPUT_CHANNELS == 3:
        image = replace_alpha_with_white(image)
    if should_augment:
        image_rgb, image_alpha = image[..., 0:3], image[..., 3]
        image_rgb = rotate_hue(image_rgb, hue_angle)
        image = tf.concat([image_rgb, image_alpha[..., tf.newaxis]], axis=-1)
    if should_normalize:
        image = normalize(image)
    return image


def create_paired_s2s_image_loader_indexed_images(sprite_side_source, sprite_side_target, dataset_sizes,
                                                  train_or_test_folder, should_augment):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide) representing images by
    its palette and indexed colors.
    """
    def load_indexed_images(dataset, image_number):
        folders = DIRECTION_FOLDERS
        source_path = tf.strings.join([dataset, train_or_test_folder, folders[sprite_side_source], image_number + ".png"], os.sep)
        target_path = tf.strings.join([dataset, train_or_test_folder, folders[sprite_side_target], image_number + ".png"], os.sep)

        source_image = tf.cast(load_image(source_path, 0., should_augment, should_normalize=False), "int32")
        target_image = tf.cast(load_image(target_path, 0., should_augment, should_normalize=False), "int32")

        # concatenates source and target so the colors in one have the same palette indices as the other
        concatenated_image = tf.concat([source_image, target_image], axis=-1)

        # finds the unique colors in both images
        palette = io_utils.extract_palette(concatenated_image)

        # converts source and target_images from RGB into indexed, using the extracted palette
        source_image = io_utils.rgba_to_indexed(source_image, palette)
        target_image = io_utils.rgba_to_indexed(target_image, palette)

        return source_image, target_image, palette

    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        # finds the dataset index and image number considering the param is an int
        # in an imaginary concatenated array of all datasets
        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source_image, target_image, palette = load_indexed_images(dataset, image_number)
        return source_image, target_image, palette

    return load_images


def create_paired_s2s_image_loader(sprite_side_source, sprite_side_target, dataset_sizes, train_or_test_folder,
                                   should_augment):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide).
    """

    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        # finds the dataset index and image number considering the param is an int
        # in an imaginary concatenated array of all datasets
        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        # defines the angle to which rotate hue
        hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        input_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_source], os.sep, image_number,
             ".png"]), hue_angle, should_augment)
        real_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_target], os.sep, image_number,
             ".png"]), hue_angle, should_augment)

        return input_image, real_image

    return load_images


def create_paired_star_image_loader(dataset_sizes, train_or_test_folder, should_augment):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    all directions of the same character are grouped together.
    Used for paired (supervised) learning such as PairedStarGAN and later.
    """

    def load_image_and_label(dataset, side_index, image_number, hue_angle):
        path = tf.strings.join(
            [dataset, train_or_test_folder, tf.gather(DIRECTION_FOLDERS, side_index), image_number + ".png"], os.sep)
        image = load_image(path, hue_angle, should_augment)
        domain = tf.one_hot(side_index, len(DIRECTION_FOLDERS))
        return image, domain

    @tf.function
    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        # defines the angle to which rotate hue
        hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # finds random source and target side
        indices = tf.random.uniform(shape=[2], minval=0, maxval=len(DIRECTION_FOLDERS), dtype="int32")
        source_index = indices[0]
        target_index = indices[1]

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source = load_image_and_label(dataset, source_index, image_number, hue_angle)
        # source_domain = tf.one_hot(source_index, len(DIRECTION_FOLDERS))

        target = load_image_and_label(dataset, target_index, image_number, hue_angle)
        # target_domain = tf.one_hot(target_index, len(DIRECTION_FOLDERS))

        # back, _ = load_image_and_label(dataset, 0, image_number, hue_angle)
        # left, _ = load_image_and_label(dataset, 1, image_number, hue_angle)
        # front, _ = load_image_and_label(dataset, 2, image_number, hue_angle)
        # right, _ = load_image_and_label(dataset, 3, image_number, hue_angle)

        # images = tf.concat(back[0], left[0], front[0], right[0], axis=-1)
        # labels = tf.concat(back[1], left[1], front[1], right[1], axis=-1)
        # images = (back[0], left[0], front[0], right[0])
        # labels = (back[1], left[1], front[1], right[1])

        # tf.print(labels)
        # return images, labels
        # return back, left, front, right

        # return (source_image, source_domain), (target_image, target_domain)
        return source, target

    return load_images


def create_unpaired_image_loader(dataset_sizes, train_or_test_folder, should_augment):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    images are all unrelated but keep a label of which side it is from.
    They are unrelated because, e.g., the front and right sides of a sprite are not paired.
    Used for unpaired (unsupervised) learning such as StarGAN.
    """

    @tf.function
    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset) * 4
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset) * 4,
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])
        file_number = image_number // 4
        side_index = image_number % 4

        # defines the angle to which rotate hue
        hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        file_number = tf.strings.as_string(file_number)
        side_folder = tf.gather(DIRECTION_FOLDERS, side_index)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        image_path = tf.strings.join([dataset, train_or_test_folder, side_folder, file_number + ".png"], os.sep)
        image = load_image(image_path, hue_angle, should_augment)
        label = tf.one_hot(side_index, len(DIRECTION_FOLDERS))

        return image, label

    return load_images
