import shutil
import io
import tensorflow as tf
from matplotlib import pyplot as plt

from configuration import *


def ensure_folder_structure(*folders):
    provided_paths = []
    for path_part in folders:
        provided_paths.extend(path_part.split(os.sep))
    folder_path = os.getcwd()
    
    for folder_name in provided_paths:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)

    
    
def random_domain_label(batch_size=BATCH_SIZE):
    random_domain = tf.random.uniform(shape=[batch_size,], minval=0, maxval=NUMBER_OF_DOMAINS, dtype="int32")
    onehot_domain_label = tf.one_hot(random_domain, NUMBER_OF_DOMAINS)
    return onehot_domain_label

def channelize_domain_label(onehot_domain_label):
    channelized_domain_label = onehot_domain_label[:, tf.newaxis, tf.newaxis, :]
    channelized_domain_label = tf.tile(channelized_domain_label, [1, IMG_SIZE, IMG_SIZE, 1])
    return channelized_domain_label

def concat_image_and_domain_label(image, onehot_label):
    channelized_domain_label = channelize_domain_label(onehot_label)
    return tf.concat([image, channelized_domain_label], axis=-1)

# def random_pair(images_and_labels_batch):
#     random_source_indices = tf.random.uniform(shape=[BATCH_SIZE,], minval=0, maxval=NUMBER_OF_DOMAINS, dtype="int32")
#     random_target_indices = tf.random.uniform(shape=[BATCH_SIZE,], minval=0, maxval=NUMBER_OF_DOMAINS, dtype="int32")
#     # tf.print("random_source_indices", random_source_indices)
#     # tf.print("images_and_labels", images_and_labels_batch)
#     print("random_source_indices", random_source_indices)
#     # print("images_and_labels", images_and_labels_batch)
    
    
#     # for images_and_labels in images_and_labels_batch:
#     print("type(images....)", type(images_and_labels_batch))
#     print("images_and_labels_batch[0]", images_and_labels_batch[0])
#     print("images_and_labels_batch[1]", images_and_labels_batch[1])
#     random_source_image = tf.gather(images_and_labels_batch, random_source_indices, axis=0)
#     random_source_domain = tf.one_hot(random_source_indices, NUMBER_OF_DOMAINS)
    
#     # random_source = images_and_labels[random_source_indices]
#     random_target_image = tf.gather(images_and_labels_batch, random_target_indices)
#     random_target_domain = tf.one_hot(random_target_indices, NUMBER_OF_DOMAINS)
    
#     return (random_source_image, random_source_domain), (random_target_image, random_target_domain)

# def random_pair_out(images_and_labels):
#     random_source_index = np.random.randint(0, NUMBER_OF_DOMAINS)
#     random_target_index = np.random.randint(0, NUMBER_OF_DOMAINS)
#     # tf.print("random_source_index", random_source_index)
#     # tf.print("images_and_labels", images_and_labels)
#     # print("random_source_index", random_source_index)
#     # print("images_and_labels", images_and_labels)
#     random_source = images_and_labels[random_source_index]
#     random_target = images_and_labels[random_target_index]
#     return random_source, random_target

# @tf.function
def extract_palette(image, channels=OUTPUT_CHANNELS, fill_until_size=256):
    """
    Extracts the unique colors from an image (3D tensor)
    Parameters
    ----------
    image a 3D tensor with shape (height, width, channels)
    channels the number of channels of the image

    Returns a tensor of colors (RGB) sorted by the number of times each one appears and from dark to light as a
    second sorting key.
    -------
    """
    # incoming image shape: (IMG_SIZE, IMG_SIZE, channels)
    # reshaping to: (IMG_SIZE*IMG_SIZE, channels)
    image = tf.cast(image, "int32")
    # image = image[::-1, ::-1]  # sorting by appearance: bottom-right to top-left
    image = tf.reshape(image, [-1, channels])
    colors, _, count = tf.raw_ops.UniqueWithCountsV2(x=image, axis=[0])

    # colors = tf.reverse(colors, [0])  # sorting by appearance: last appeared color comes first
    # colors = tf.random.shuffle(colors)  # shuffled colors
    # sorts the colors by the amount of times it appeared
    # indices_sorted_by_count = tf.argsort(count, direction="DESCENDING", stable=True)
    # colors = tf.gather(colors, indices_sorted_by_count)

    # sorts them again (stably) putting the darker tones first
    # here, lightness is given by (max(r,g,b) + min(r,g,b)) / 2.0 (as in the RGB to HSL conversion)
    # lightness = ((tf.reduce_max(colors, axis=-1) + tf.reduce_min(colors, axis=-1)) / 2)
    # indices_sorted_by_lightness = tf.argsort(lightness, direction="ASCENDING", stable=True)
    # colors = tf.gather(colors, indices_sorted_by_lightness)
    # in_gray = tf.image.rgb_to_grayscale(colors)
    gray_coefficients = tf.constant([0.2989, 0.5870, 0.1140, 0.])[..., tf.newaxis]
    grayness = tf.squeeze(tf.matmul(tf.cast(colors, "float32"), gray_coefficients))
    indices_sorted_by_grayness = tf.argsort(grayness, direction="ASCENDING", stable=True)
    colors = tf.gather(colors, indices_sorted_by_grayness)

    # fills the palette to have 256 colors, so batches can be created (otherwise they can't, tf complains)
    # TODO do something when the palette exceeds MAX_PALETTE_SIZE
    num_colors = tf.shape(colors)[0]
    if fill_until_size is not None:
        fillers = tf.repeat([INVALID_INDEX_COLOR], [MAX_PALETTE_SIZE - num_colors], axis=0)
        colors = tf.concat([colors, fillers], axis=0)

    return colors


@tf.function
def rgba_to_single_int(values_in_rgba):
    shape = tf.shape(values_in_rgba)
    converted = tf.zeros(shape=shape[:-1], dtype="int32")
    # multiplier: 2 ** [24, or 16, or 8, or 0]
    for i, multiplier in enumerate([16777216, 65536, 256, 0]):
        converted += values_in_rgba[..., i] * multiplier
    return converted


# @tf.function
def rgba_to_indexed(image, palette):
    original_shape = tf.shape(image)
    flattened_image = tf.reshape(image, [-1, original_shape[-1]])
    num_pixels, num_components = tf.shape(flattened_image)[0], tf.shape(flattened_image)[1]

    indices = flattened_image == palette[:, None]
    row_sums = tf.reduce_sum(tf.cast(indices, "int32"), axis=2)
    results = tf.cast(tf.where(row_sums == num_components), "int32")

    color_indices, pixel_indices = results[:, 0], results[:, 1]
    pixel_indices = tf.expand_dims(pixel_indices, -1)

    indexed = tf.scatter_nd(pixel_indices, color_indices, [num_pixels])
    indexed = tf.reshape(indexed, [original_shape[0], original_shape[1], 1])
    return indexed


# def rgba_to_indexed(image_tensor, palette_tensor):
#     def convert(image, palette):
#         result = np.ndarray(shape=[image.shape[0], image.shape[1]])
#
#         for i, row in enumerate(image):
#             for j, color in enumerate(row):
#                 index, = np.where(np.all(palette == color, axis=1))
#                 result[i, j] = index
#         return result
#     # return tf.py_function(func=convert, inp=[image_tensor, palette_tensor], Tout="int32")
#     return convert(image_tensor, palette_tensor)
# def rgba_to_indexed(image, palette):
#     # convert image and palette from RGBA into a single integer for each color
#     image_as_int = tf.map_fn(rgba_to_single_int, image)
#     palette_as_int = tf.map_fn(rgba_to_single_int, palette)
#
#     # creates a hash table for looking up the indices in the palette
#     palette_size = tf.shape(palette)[0]
#     initializer = tf.lookup.KeyValueTensorInitializer(palette_as_int, tf.range(palette_size))
#     palette_table = tf.lookup.StaticHashTable(
#         initializer,
#         default_value=-42,
#         experimental_is_anonymous=True)
#
#     # looks up the color indices of the image (as_int) in the palette hash table
#     indexed_image = palette_table.lookup(image_as_int)
#     indexed_image = tf.expand_dims(indexed_image, -1)
#     return indexed_image


@tf.function
def indexed_to_rgba(indexed_image, palette):
    image_shape = tf.shape(indexed_image)
    # tf.print("image_shape", image_shape)
    # tf.print("tf.shape(palette)", tf.shape(palette))
    image_rgb = tf.gather(palette, indexed_image)

    # now the shape is (HEIGHT, WIDTH, 1, CHANNELS), so we need to reshape
    # tf.print("tf.shape(image_rgb) inside indexed_to_rgb", tf.shape(image_rgb))
    image_rgb = tf.reshape(image_rgb, [image_shape[0], image_shape[1], -1])
    # tf.print("tf.shape(image_rgb) inside indexed_to_rgb (after reshape)", tf.shape(image_rgb))
    return image_rgb

def plot_to_image(matplotlib_figure, channels=OUTPUT_CHANNELS):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(matplotlib_figure)
    buffer.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buffer.getvalue(), channels=channels)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    







# # generates images depicting what the discriminator thinks of a target image and a generated image - 
# # how did it find each one's patches as real or fake
# def generate_discriminated_image(input_image, target_image, discriminator, generator, invert_discriminator_value=False):
#     generated_image = generator(input_image, training=True)

#     discriminated_target_image = tf.math.sigmoid(tf.squeeze(discriminator([input_image, target_image], training=True), axis=[0]))
#     discriminated_generated_image = tf.math.sigmoid(tf.squeeze(discriminator([input_image, generated_image], training=True), axis=[0]))
#     if invert_discriminator_value:
#         discriminated_target_image = 1. - discriminated_target_image 
#         discriminated_generated_image = 1. - discriminated_generated_image 
    
#     # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)}")

#     patches = tf.shape(discriminated_target_image).numpy()[0]
#     lower_bound_scaling_factor = IMG_SIZE // patches
#     # print(f"lower_bound_scaling_factor {lower_bound_scaling_factor}, shape {tf.shape(discriminated_target_image).numpy()}")
#     pad_before = (IMG_SIZE - patches*lower_bound_scaling_factor)//2
#     pad_after = (IMG_SIZE - patches*lower_bound_scaling_factor) - pad_before
#     # print(f"pad_before {pad_before}, pad_after {pad_after}")
#     discriminated_target_image = tf.repeat(tf.repeat(discriminated_target_image, lower_bound_scaling_factor, axis=0), lower_bound_scaling_factor, axis=1)
#     # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)} - after repeat")
#     discriminated_target_image = tf.pad(discriminated_target_image, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
#     # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)} - after pad")

#     discriminated_generated_image = tf.repeat(tf.repeat(discriminated_generated_image, lower_bound_scaling_factor, axis=0), lower_bound_scaling_factor, axis=1)
#     discriminated_generated_image = tf.pad(discriminated_generated_image, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

#     generated_image = tf.squeeze(generated_image)
#     target_image = tf.squeeze(target_image)


#     figure = plt.figure(figsize=(6*2, 6*2))
#     for i, (image, disc_output, image_label, output_label) in enumerate(zip([target_image, generated_image], [discriminated_target_image, discriminated_generated_image], ["Target", "Generated"], ["Discriminated target", "Discriminated generated"])):
#         plt.subplot(2, 2, i*2 + 1)
#         plt.title(image_label, fontdict={"fontsize": 20})
#         plt.imshow(image * 0.5 + 0.5)
#         plt.axis("off")

#         plt.subplot(2, 2, i*2 + 2)
#         plt.title(output_label, fontdict={"fontsize": 20})
#         plt.imshow(disc_output, cmap="gray", vmin=0.0, vmax=1.0)
#         plt.axis("off")

    
#     plt.show()
#     # print(discriminated_generated_image)
    
    

def seconds_to_human_readable(time):
    days = time // 86400         # (60 * 60 * 24)
    hours = time // 3600 % 24    # (60 * 60) % 24
    minutes = time // 60 % 60 
    seconds = time % 60
    
    time_string = ""
    if days > 0:
        time_string += f"{days:.0f} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:
        time_string += f"{hours:02.0f}h:"
    time_string += f"{minutes:02.0f}m:{seconds:02.0f}s"
    
    return time_string