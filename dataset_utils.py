import tensorflow as tf
from functools import partial

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


# def rotate_hue(image, angle):
#     return tf.image.adjust_hue(image, angle)


# loads an image from the file system and transforms it for the network:
# (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
# puts the values in the range of [-1, 1]
def load_image(path, should_normalize=True):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=INPUT_CHANNELS)
        image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS))
        image = tf.cast(image, "float32")
        image = blacken_transparent_pixels(image)
        if OUTPUT_CHANNELS == 3:
            image = replace_alpha_with_white(image)
        if should_normalize:
            image = normalize(image)
    except UnicodeDecodeError:
        print("Error opening image in ", path)
    return image


def augment_hue_rotation(image, seed):
    image_rgb, image_alpha = image[..., 0:3], image[..., 3]
    image_rgb = tf.image.stateless_random_hue(image_rgb, 0.5, seed)
    image = tf.concat([image_rgb, image_alpha[..., tf.newaxis]], axis=-1)
    return image


def augment_translation(images):
    image = tf.concat([*images], axis=-1)
    translate = tf.keras.layers.RandomTranslation((-0.15, 0.075), 0.125, fill_mode="constant", interpolation="nearest")
    image = translate(image)
    images = tf.split(image, len(images), axis=-1)
    return tf.tuple(images)


def augment_two(first, second):
    # hue rotation
    hue_seed = tf.random.uniform(shape=[2], minval=0, maxval=65536, dtype="int32")
    first = augment_hue_rotation(first, hue_seed)
    second = augment_hue_rotation(second, hue_seed)
    # translation
    first, second = augment_translation((first, second))
    return first, second


def augment_two_with_labels(first, second):
    first_image, first_label = first
    second_image, second_label = second
    # hue rotation
    hue_seed = tf.random.uniform(shape=[2], minval=0, maxval=65536, dtype="int32")
    first_image = augment_hue_rotation(first_image, hue_seed)
    second_image = augment_hue_rotation(second_image, hue_seed)
    # translation
    first_image, second_image = augment_translation((first_image, second_image))
    return (first_image, first_label), (second_image, second_label)


def normalize_two(first, second):
    return normalize(first), normalize(second)


def normalize_two_with_labels(first, second):
    first_image, first_label = first
    second_image, second_label = second
    return (normalize(first_image), first_label), (normalize(second_image), second_label)


def create_augmentation_with_prob(prob=0.8):
    prob = tf.constant(prob)

    def augmentation_wrapper(first, second):
        choice = tf.random.uniform(shape=[])
        should_augment = choice < prob
        if should_augment:
            return augment_two(first, second)
        else:
            return first, second

    return augmentation_wrapper


def create_augmentation_with_prob_with_labels(prob=0.8):
    prob = tf.constant(prob)

    def augmentation_wrapper(first, second):
        choice = tf.random.uniform(shape=[])
        should_augment = choice < prob
        if should_augment:
            return augment_two_with_labels(first, second)
        else:
            return first, second

    return augmentation_wrapper


def create_paired_s2s_image_loader_indexed_images(sprite_side_source, sprite_side_target, dataset_sizes,
                                                  train_or_test_folder, should_augment=False):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide) representing images by
    its palette and indexed colors.
    """
    def load_indexed_images(dataset, image_number):
        folders = DIRECTION_FOLDERS
        source_path = tf.strings.join([dataset, train_or_test_folder, folders[sprite_side_source], image_number + ".png"], os.sep)
        target_path = tf.strings.join([dataset, train_or_test_folder, folders[sprite_side_target], image_number + ".png"], os.sep)

        source_image = tf.cast(load_image(source_path, should_normalize=False), "int32")
        target_image = tf.cast(load_image(target_path, should_normalize=False), "int32")

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


def create_paired_s2s_image_loader(sprite_side_source, sprite_side_target, dataset_sizes, train_or_test_folder):
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
        # augmentation_dice = np.random.random()
        # tf.print("augmentation_dice", augmentation_dice)
        # should_augment = augmentation_dice < augmentation_probability
        # tf.print("should_augment", should_augment)
        # should_augment = True
        # hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)
        # tf.print("hue_angle", hue_angle)
        # augmentation_seed = np.random.randint(0, 65536)
        # augmentation_seed = image_number #tf.random.uniform(shape=[], dtype="int32", minval=0, maxval=65536)
        # tf.print("augmentation_seed", augmentation_seed)
        # augmentation_seed = image_number

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        # tf.print("loading input ", tf.strings.join(
        #     [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_source], os.sep, image_number,
        #      ".png"]))
        # tf.print("loading target ", tf.strings.join(
        #     [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_target], os.sep, image_number,
        #      ".png"]))
        input_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_source], os.sep, image_number,
             ".png"]), False)#, hue_angle, augmentation_seed, should_augment)
        real_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_target], os.sep, image_number,
             ".png"]), False)#, hue_angle, augmentation_seed, should_augment)

        # # augmentation
        # if should_augment:
        #     hue_seed = tf.random.uniform(shape=[2], minval=0, maxval=65536, dtype="int32")
        #     # translate_seed = tf.random.uniform(shape=[], minval=0, maxval=65536, dtype="int32").numpy()
        #
        #     input_image = augment_hue_rotation(input_image, hue_seed)
        #     real_image = augment_hue_rotation(real_image, hue_seed)
        #
        #     input_image, real_image = augment_translation((input_image, real_image))
        #
        # # transforms from [0,255] to [-1,1]
        # input_image = normalize(input_image)
        # real_image = normalize(real_image)

        return input_image, real_image

    return load_images


def create_paired_star_image_loader(dataset_sizes, train_or_test_folder, should_normalize=True, identity_prob=0.05):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    all directions of the same character are grouped together.
    Used for paired (supervised) learning such as PairedStarGAN and later.
    """

    def load_image_and_label(dataset, side_index, image_number):
        path = tf.strings.join(
            [dataset, train_or_test_folder, tf.gather(DIRECTION_FOLDERS, side_index), image_number + ".png"], os.sep)
        image = load_image(path, should_normalize)
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
        # hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # finds random source and target side
        should_target_and_source_be_the_same = tf.random.uniform(shape=()) < identity_prob
        source_index = tf.random.uniform(shape=(), minval=0, maxval=len(DIRECTION_FOLDERS), dtype="int32")
        if should_target_and_source_be_the_same:
            target_index = source_index
        else:
            target_index = tf.random.uniform(shape=(), minval=0, maxval=len(DIRECTION_FOLDERS), dtype="int32")
            while target_index == source_index:
                target_index = tf.random.uniform(shape=(), minval=0, maxval=len(DIRECTION_FOLDERS), dtype="int32")

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source = load_image_and_label(dataset, source_index, image_number)
        # source_domain = tf.one_hot(source_index, len(DIRECTION_FOLDERS))

        target = load_image_and_label(dataset, target_index, image_number)
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


def create_unpaired_image_loader(dataset_sizes, train_or_test_folder, should_normalize):
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
        image = load_image(image_path, should_normalize)
        label = tf.one_hot(side_index, len(DIRECTION_FOLDERS))

        return image, label

    return load_images


def create_collaborative_image_loader(dataset_sizes, train_or_test_folder, should_normalize=True, input_dropout=[1,2,3]):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    all directions of the same character are grouped together.
    Used for CollaGAN.
    """

    # def load_image_and_label(dataset, side_index, image_number):
    #     path = tf.strings.join(
    #         [dataset, train_or_test_folder, tf.gather(DIRECTION_FOLDERS, side_index), image_number + ".png"], os.sep)
    #     image = load_image(path, should_normalize)
    #     domain = tf.one_hot(side_index, len(DIRECTION_FOLDERS))
    #     return image, domain

    def create_input_dropout_index_list(inputs_to_drop):
        """
        Creates a list shape=(DOMAINS, TO_DROP, ? DOMAINS) that is, per possible target pose index (first dimension),
        for each possible number of dropped inputs (second dimension): all permutations of a boolean array that
        (a) nullifies the target index and (b) nullifies a number of additional inputs equal to 0, 1 or 2 (determined
        by inputs_to_drop).
        Parameters
        ----------
        inputs_to_drop a list of the number of inputs we want to drop. Must be at least [1], but can be [1, 2],
        [1, 2, 3], or [1, 3].

        Returns a 4d array with all the permutations described.
        -------

        """
        null_lists_per_target_index = []
        for target_index in range(NUMBER_OF_DOMAINS):
            null_list_for_current_target = []
            for number_of_inputs_to_drop in inputs_to_drop:
                tmp_a = []
                if number_of_inputs_to_drop == 1:
                    tmp = [bX == target_index for bX in range(NUMBER_OF_DOMAINS)]
                    tmp_a.append(tmp)

                elif number_of_inputs_to_drop == 2:
                    for i_in in range(NUMBER_OF_DOMAINS):
                        if not i_in == target_index:
                            tmp = [bX in [i_in, target_index] for bX in range(NUMBER_OF_DOMAINS)]
                            tmp_a.append(tmp)

                elif number_of_inputs_to_drop == 3:
                    for i_in in range(NUMBER_OF_DOMAINS):
                        if not (i_in == target_index):
                            tmp = [(bX == target_index or (not bX == i_in)) for bX in range(NUMBER_OF_DOMAINS)]
                            tmp_a.append(tmp)

                null_list_for_current_target.append(tmp_a)
            null_lists_per_target_index.append(null_list_for_current_target)

        return null_lists_per_target_index

    def load_a_side_image(dataset, side_index, image_number):
        path = tf.strings.join(
            [dataset, train_or_test_folder, tf.gather(DIRECTION_FOLDERS, side_index), image_number + ".png"], os.sep
        )
        return load_image(path, should_normalize)

    def channelize_domain(index):
        one_hot_domain = tf.one_hot(index, NUMBER_OF_DOMAINS)
        channelized_domain = tf.tile(one_hot_domain[tf.newaxis, tf.newaxis, :], [IMG_SIZE, IMG_SIZE, 1])
        return channelized_domain

    @tf.function
    def load_images(dropout_null_list, image_number):
        image_number = tf.cast(image_number, "int32")

        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads all images from the disk
        back_image = load_a_side_image(dataset, 0, image_number)
        left_image = load_a_side_image(dataset, 1, image_number)
        front_image = load_a_side_image(dataset, 2, image_number)
        right_image = load_a_side_image(dataset, 3, image_number)

        # finds a random target side
        target_domain_index = tf.random.uniform(shape=[], maxval=NUMBER_OF_DOMAINS, dtype="int32")
        target_domain_mask = channelize_domain(target_domain_index)

        # applies input dropout as described in the CollaGAN paper and implemented in the code
        #  this is adapted from getBatch_RGB_varInp in CollaGAN
        #  a. randomly choose an input dropout mask such as [True, False, False, True]
        dropout_null_list_for_target = tf.gather(dropout_null_list, target_domain_index)
        random_number_of_inputs_to_drop = tf.random.uniform(shape=[], maxval=tf.shape(dropout_null_list_for_target)[0], dtype="int32")
        dropout_null_list_for_target_and_number_of_inputs = tf.gather(dropout_null_list_for_target, random_number_of_inputs_to_drop)
        random_permutation_index = tf.random.uniform(shape=[], maxval=tf.shape(dropout_null_list_for_target_and_number_of_inputs)[0], dtype="int32")
        input_dropout_mask = tf.gather(dropout_null_list_for_target_and_number_of_inputs, random_permutation_index)

        #  b. do apply the dropout by creating an input mask
        # TODO... the CollaGAN implementation does not do anything with the input images at this point

        return back_image, left_image, front_image, right_image, target_domain_index, target_domain_mask, input_dropout_mask

    null_list = tf.ragged.constant(create_input_dropout_index_list(input_dropout), ragged_rank=2, dtype="bool")
    return partial(load_images, null_list)
