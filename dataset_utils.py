import tensorflow as tf
from configuration import *

def load_paired_images(image_paths, image_size):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(lambda x: load_image(x, image_size, 4, "nearest"))
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    # Include file paths for images as attribute.
    dataset.file_paths = image_paths
    return dataset



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


# rescales images in the interval [-1, 1]
def normalize(image):
    return (image / 127.5) - 1


# loads an image from the file system and transforms it for the network:
# (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
# puts the values in the range of [-1, 1]
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=INPUT_CHANNELS)
    image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS))
    image = tf.cast(image, tf.float32)
    image = blacken_transparent_pixels(image)
    if OUTPUT_CHANNELS == 3:
        image = replace_alpha_with_white(image)
    image = normalize(image)
    return image

def create_image_loader(sprite_side_source, sprite_side_target, dataset_sizes, train_or_test_folder):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide).
    """
    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        # finds the dataset index and image number considering the param is an int in an imaginary concatenated array of all datasets
        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda image_number, dataset_index: image_number >= tf.gather(dataset_sizes, dataset_index)
        body = lambda image_number, dataset_index: [image_number - tf.gather(dataset_sizes, dataset_index), dataset_index + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])
        
        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)
        input_image = load_image(tf.strings.join([dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_source], os.sep, image_number, ".png"]))
        real_image = load_image(tf.strings.join([dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_target], os.sep, image_number, ".png"]))        
        
        return input_image, real_image
    return load_images
