import tensorflow as tf

def load_image(path, image_size, num_channels, interpolation,
               crop_to_aspect_ratio=False):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
      img, channels=num_channels, expand_animations=False)
    if crop_to_aspect_ratio:
    img = keras_image_ops.smart_resize(img, image_size,
                                       interpolation=interpolation)
    else:
    img = tf.image.resize(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

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