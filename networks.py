import tensorflow as tf
from tensorflow.keras import layers
from configuration import *
from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from tensorflow_addons import layers as tfalayers


def unet_downsample(filters, size, apply_batchnorm=True, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    #     result.add(layers.Conv2D(filters, size, strides=1, padding="same", kernel_initializer=initializer, use_bias=False))
    #     if apply_batchnorm:
    #         result.add(layers.BatchNormalization())
    #     result.add(layers.LeakyReLU())

    #     result.add(layers.Conv2D(filters, size, strides=1, padding="same", kernel_initializer=initializer, use_bias=False))
    #     if apply_batchnorm:
    #         result.add(layers.BatchNormalization())
    #     result.add(layers.LeakyReLU())

    result.add(layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=init,
        use_bias=False))
    if apply_batchnorm:
        result.add(tfalayers.InstanceNormalization())
    result.add(layers.LeakyReLU())

    return result


def unet_upsample(filters, size, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))

    result.add(tfalayers.InstanceNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    #     result.add(layers.Conv2D(filters, size, strides=1, padding="same", kernel_initializer=initializer, use_bias=False))
    #     result.add(layers.BatchNormalization())
    #     result.add(layers.ReLU())

    #     result.add(layers.Conv2D(filters, size, strides=1, padding="same", kernel_initializer=initializer, use_bias=False))
    #     result.add(layers.BatchNormalization())
    #     result.add(layers.ReLU())

    return result


def PatchDiscriminator(num_patches, **kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)
    # x = CrossProduct()([target_image, source_image])

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    last = layers.Conv2D(1, 4, padding="same",
                         kernel_initializer=initializer)(down)  # (batch_size, 32, 32, 1)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="patch-disc")


def Deeper2x2PatchDiscriminator(**kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    down = unet_downsample(128, 4, False)(down)  # (batch_size, 16, 16, 128)
    # down = unet_downsample(256, 4, False)(down)                 # (batch_size,  8,  8, 256)

    # x = layers.Conv2D(256, 4, padding="same",
    #                   kernel_initializer=initializer, use_bias=False)(down)     # (batch_size,  8,  8, 512)
    # x = tfalayers.InstanceNormalization()(x)
    # x = layers.LeakyReLU()(x)

    last = layers.Conv2D(1, 4, padding="same",
                         kernel_initializer=initializer)(down)  # (batch_size,  8,  8,   1)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last)


class CrossProduct(tf.keras.layers.Layer):
    def __init__(self):
        super(CrossProduct, self).__init__()

    def call(self, inputs):
        x, y = inputs
        # (None, 64, 64, 4)
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        # print("x", x)
        # print("y", y)
        batch_size = x_shape[0]

        # strips away with the alpha channel
        if OUTPUT_CHANNELS == 4:
            x_alpha = tf.expand_dims(x[:, :, :, 3], -1)
            # print("x_alpha", x_alpha)
            x = x[:, :, :, 0:3]
            y = y[:, :, :, 0:3]

        # reshape to have a flattened list of pixels (all pixels of all images in the batch)
        x = tf.reshape(x, [batch_size * IMG_SIZE * IMG_SIZE, 3])
        y = tf.reshape(y, [batch_size * IMG_SIZE * IMG_SIZE, 3])
        # print("x after flattening", x)
        # print("y after flattening", y)

        # effectively calculate the cross product between paired pixels from images, then reshapes back to a batch
        cross_product = tf.linalg.cross(x, y)
        # print("cross_product", cross_product)
        cross_product = tf.reshape(cross_product, [batch_size, IMG_SIZE, IMG_SIZE, 3])
        # print("cross_product after deflattening", cross_product)

        # inserts the alpha channel back, picking the same from x (target/generated image)
        if OUTPUT_CHANNELS == 4:
            # print("right before concating")
            cross_product = tf.concat([cross_product, x_alpha], -1)
            # print("cross_product after concat", cross_product)

        return cross_product


def PatchResnetDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([target_image, source_image])
    # inputs = CrossProduct()([target_image, source_image])
    x = inputs

    x = layers.Conv2D(128, 4, padding="same", kernel_initializer=init)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 4, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = resblock(x, 64, 4, init)
    x = layers.Dropout(0.5)(x)
    x = resblock(x, 64, 4, init)
    x = layers.Dropout(0.5)(x)
    x = resblock(x, 64, 4, init)
    x = layers.Dropout(0.5)(x)
    x = resblock(x, 64, 4, init)
    x = layers.Dropout(0.5)(x)
    x = resblock(x, 64, 4, init)
    x = resblock(x, 64, 4, init)

    last = layers.Conv2D(1, 4, padding="same", kernel_initializer=init)(x)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="patch-resnet-disc")


def IndexedPatchDiscriminator(num_patches, **kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, 2)

    if num_patches > 30:
        raise ValueError(f"num_patches for the discriminator should not exceed 30, but {num_patches} was given")

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    current_patches = 30
    next_layer_filters = 64 * 2
    while current_patches > max(num_patches, 2):
        # print(f"Created downsample with {next_layer_filters}")
        down = unet_downsample(next_layer_filters, 4)(down)
        current_patches = ((current_patches + 2) / 2) - 2
        next_layer_filters *= 2
        # print(f"current_patches now {current_patches}")
    zero_pad2 = layers.ZeroPadding2D()(down)  # (batch_size, 34, 34, 64)
    last_filter_size = 5 if num_patches > 1 else 5
    last = layers.Conv2D(1, last_filter_size, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="indexed-patch-disc")


def IndexedPatchResnetDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="target_image")

    inputs = layers.concatenate([target_image, source_image])
    x = inputs

    x = layers.Conv2D(64, 4, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = resblock(x, 64, 4, init)
    x = layers.Dropout(0.5)(x)
    # x = resblock(x, 64, 4, init)
    # x = layers.Dropout(0.5)(x)
    # x = resblock(x, 64, 4, init)
    # x = layers.Dropout(0.5)(x)

    last = layers.Conv2D(1, 4, padding="same", kernel_initializer=init)(x)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="indexed-patch-resnet-disc")


def UnetDiscriminator(**kwargs):
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64,    8)

    down_stack = [
        unet_downsample(16, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   32)
        unet_downsample(32, 4, init=init),  # (batch_size, 16, 16,   64)
        unet_downsample(64, 4, init=init),  # (batch_size,  8,  8,  128)
        unet_downsample(128, 4, init=init),  # (batch_size,  4,  4,  256)
        unet_downsample(256, 4, init=init),  # (batch_size,  2,  2,  512)
        # unet_downsample(512, 4, init=init),                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        # unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  2,  2, 1024)
        unet_upsample(128, 4, apply_dropout=True, init=init),  # (batch_size,  4,  4,  512)
        unet_upsample(64, 4, apply_dropout=True, init=init),  # (batch_size,  8,  8,  256)
        unet_upsample(32, 4, init=init),  # (batch_size, 16, 16,  128)
        unet_upsample(16, 4, init=init),  # (batch_size, 32, 32,   64)
    ]

    last = layers.Conv2DTranspose(1, 4, padding="same", strides=2,
                                  kernel_initializer=init)  # (batch_size, 64, 64,    1)

    x = inputs

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    code_classification = layers.Conv2D(64, 2)(x)
    code_classification = layers.LeakyReLU()(code_classification)
    code_classification = layers.Conv2D(1, 1)(code_classification)
    code_classification = layers.Reshape([1])(code_classification)
    # code_classification = layers.Flatten()(x)
    # code_classification = layers.Dense(1)(code_classification)

    # # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=[x, code_classification], name="unet-disc")


def segnet_downsample(inputs, filters, size, convolution_steps=2, init=tf.random_normal_initializer(0., 0.02)):
    x = inputs
    for i in range(convolution_steps):
        x = layers.Conv2D(
            filters,
            size,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=init,
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x, indices = MaxPoolingWithArgmax2D(pool_size=2)(x)
    return x, indices


def segnet_upsample(inputs, filters, size, convolution_steps=2, apply_dropout=False,
                    init=tf.random_normal_initializer(0., 0.02)):
    x, indices = inputs
    x = MaxUnpooling2D(size=2)([x, indices])

    for i in range(convolution_steps):
        x = layers.Conv2D(filters, size, strides=1,
                          padding="same",
                          use_bias=False,
                          kernel_initializer=init,
                          )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    if apply_dropout:
        x = layers.Dropout(0.5)(x)

    return x


def SegnetDiscriminator(**kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)

    output = segnet_downsample(inputs, 32, 4, 1),  # (batch_size, 32, 32, 8)
    x, indices0 = output[0][0], output[0][1]
    output = segnet_downsample(x, 64, 4, 2),  # (batch_size, 16, 16, 16)
    x, indices1 = output[0][0], output[0][1]
    output = segnet_downsample(x, 128, 4, 2),  # (batch_size, 8, 8, 32)
    x, indices2 = output[0][0], output[0][1]
    output = segnet_downsample(x, 256, 4, 2),  # (batch_size, 4, 4, 64)
    x, indices3 = output[0][0], output[0][1]

    x = segnet_upsample([x, indices3], 128, 4, 2, apply_dropout=True),  # (batch_size, 8, 8, 32)
    x = segnet_upsample([x, indices2], 64, 4, 2, apply_dropout=True),  # (batch_size, 16, 16, 16)
    x = segnet_upsample([x, indices1], 32, 4, 1),  # (batch_size, 32, 32, 8)

    last = layers.Conv2D(1, 4, strides=1,
                         padding="same",
                         kernel_initializer=initializer
                         )

    x = last(x[0])

    return tf.keras.Model(inputs=[target_image, source_image], outputs=x, name="segnet-disc")


# Adapted from GANIMORPH: https://arxiv.org/pdf/1808.04325.pdf
def AtrousDiscriminator():
    def conv_block(x, filters, kernel_size, strides, initializer, skip_normalization=False, dilation=1, use_bias=True):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same",
            dilation_rate=dilation,
            use_bias=use_bias,
            kernel_initializer=initializer)(x)
        if not skip_normalization:
            x = tfalayers.InstanceNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)
    # inputs = CrossProduct()([target_image, source_image])  # (batch_size, 64, 64, channels*2)

    x = conv_block(inputs, 128, 4, 1, init, skip_normalization=True)
    x = conv_block(x, 256, 4, 1, init)
    x = conv_block(x, 512, 4, 2, init)
    x = conv_block(x, 512, 3, 1, init)
    skip = x

    x = conv_block(x, 512, 3, 1, init, dilation=2, use_bias=False)
    x = conv_block(x, 512, 3, 1, init, dilation=4, use_bias=False)
    x = conv_block(x, 512, 3, 1, init, dilation=8, use_bias=False)

    x = layers.Concatenate()([x, skip])
    x = conv_block(x, 512, 3, 1, init)
    x = layers.Conv2D(1, 4, strides=1, padding="valid", kernel_initializer=init, use_bias=False)(x)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=x, name="atrous-disc")


def AtrousDiscriminator_out():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)

    x = layers.Conv2D(64, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    skip_downsampling = x

    x = layers.Conv2D(128, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    skip = x

    x2 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=2, kernel_initializer=init, use_bias=False)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)

    x4 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=4, kernel_initializer=init, use_bias=False)(x)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)

    x8 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=8, kernel_initializer=init, use_bias=False)(x)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.LeakyReLU()(x8)

    x = layers.Concatenate()([x2, x4, x8, skip])
    x = layers.Conv2D(512, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    skip = x

    x2 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=2, kernel_initializer=init, use_bias=False)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)

    x4 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=4, kernel_initializer=init, use_bias=False)(x)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)

    x8 = layers.Conv2D(128, 4, strides=1, padding="same", dilation_rate=8, kernel_initializer=init, use_bias=False)(x)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.LeakyReLU()(x8)

    x = layers.Concatenate()([x2, x4, x8, skip])
    x = layers.Conv2D(512, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Concatenate()([x, skip_downsampling])
    x = layers.Conv2D(128, 4, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(1, 3, strides=1, kernel_initializer=init)(x)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=x, name="atrous-disc")


def UnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])  # (batch_size, 64, 64, 4)

    down_stack = [
        unet_downsample(64, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),  # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),  # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),  # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),  # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),  # (batch_size,  4,  4, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),  # (batch_size,  8,  8,  512)
        unet_upsample(128, 4, init=init),  # (batch_size, 16, 16,  256)
        unet_upsample(64, 4, init=init),  # (batch_size, 32, 32,  128)
        unet_upsample(32, 4, init=init),  # (batch_size, 64, 64,   36)
    ]

    last = layers.Conv2D(OUTPUT_CHANNELS, 4,
                         strides=1,
                         padding="same",
                         kernel_initializer=init,
                         activation="tanh")  # (batch_size, 64, 64, 4)

    x = inputs

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignora a última skip e inverte a ordem
    skips = list(reversed(skips[:-1]))

    # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="unet-gen")


def IndexedUnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="input_image")  # (batch_size, 64, 64, 1)
    # layers.Input(shape=[], name="palette_size")]

    down_stack = [
        unet_downsample(32, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(64, 4, init=init),  # (batch_size, 16, 16,  128)
        unet_downsample(128, 4, init=init),  # (batch_size,  8,  8,  256)
        unet_downsample(256, 4, init=init),  # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),  # (batch_size,  2,  2, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),  # (batch_size,  4,  4, 1024)
        unet_upsample(128, 4, apply_dropout=True, init=init),  # (batch_size,  8,  8,  512)
        unet_upsample(64, 4, init=init),  # (batch_size, 16, 16,  256)
        unet_upsample(64, 4, init=init),  # (batch_size, 32, 32,  128)
        unet_upsample(128, 4, init=init),  # (batch_size, 64, 64,   64)
    ]

    last = layers.Conv2D(MAX_PALETTE_SIZE, 4,
                         strides=1,
                         padding="same",
                         kernel_initializer=init,
                         activation="softmax"
                         )  # (batch_size, 64, 64, 4)

    x = inputs

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignora a última skip e inverte a ordem
    skips = list(reversed(skips[:-1]))

    # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet_conv_block(inputs, filters, kernel_size, initializer, apply_batchnorm=True, apply_dropout=False,
                    activation="leaky", bias=False):
    x = inputs

    # "same" conv2d
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same", kernel_initializer=initializer, use_bias=bias)(x)
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)

    if activation == "leaky":
        x = layers.LeakyReLU()(x)
    elif activation == "relu":
        x = layers.ReLU()(x)
    elif activation == "tanh":
        x = layers.Activation("tanh")(x)

    return x


def unet_encoder_block(inputs, filters, kernel_size, initializer, apply_batchnorm=True):
    x = inputs

    # "same" conv2d
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=apply_batchnorm, activation="leaky")
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=apply_batchnorm, activation="leaky")
    skip_activations = x

    # downsampling
    x = layers.Conv2D(filters, kernel_size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False)(
        x)
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    output = x

    return output, skip_activations


def unet_decoder_block(inputs, features_from_skip, filters, kernel_size, initializer, apply_dropout=False):
    x = inputs

    # upsampling
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same", kernel_initializer=initializer,
                               use_bias=False)(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.ReLU()(x)

    # "same" conv2d
    x = layers.Concatenate()([x, features_from_skip])
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=False, apply_dropout=apply_dropout,
                        activation="relu")
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=False, apply_dropout=apply_dropout,
                        activation="relu")

    return x


def Unet2Generator():
    initializer = tf.random_normal_initializer(0., 0.002)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    e0 = inputs

    e1, s1 = unet_encoder_block(e0, 32, 4, initializer, False)  # (None, 32, 32,   32)
    e2, s2 = unet_encoder_block(e1, 64, 4, initializer)  # (None, 16, 16,   64)
    e3, s3 = unet_encoder_block(e2, 128, 4, initializer)  # (None,  8,  8,  128)
    e4, s4 = unet_encoder_block(e3, 256, 4, initializer)  # (None,  4,  4,  256)
    #     e5, s5 = unet_encoder_block(e4, 512, 4, initializer)          # (None,  2,  2,  512)
    #     e6, s6 = unet_encoder_block(e5, 512, 4, initializer)          # (None,  1,  1,  512)

    #     d6 = unet_decoder_block(e6, s6, 512, 4, initializer, True)    # (None,  2,  2, 1024)
    #     d5 = unet_decoder_block(d6, s5, 512, 4, initializer, True)    # (None,  4,  4, 1024)
    d4 = unet_decoder_block(e4, s4, 256, 4, initializer, True)  # (None,  8,  8,  512)
    d3 = unet_decoder_block(d4, s3, 128, 4, initializer)  # (None, 16, 16,  256)
    d2 = unet_decoder_block(d3, s2, 64, 4, initializer)  # (None, 32, 32,  128)
    d1 = unet_decoder_block(d2, s1, 32, 4, initializer)  # (None, 64, 64,   64)

    output = unet_conv_block(d1, OUTPUT_CHANNELS, 4, initializer,
                             apply_batchnorm=False,
                             activation="tanh", bias=True)  # (None, 64, 64,    4)

    return tf.keras.Model(inputs=inputs, outputs=output)


def SegnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])  # (batch_size, 64, 64, 4)

    output = segnet_downsample(inputs, 64, 4, 3, init=init),  # (batch_size, 32, 32, 64)
    x, indices0 = output[0][0], output[0][1]
    output = segnet_downsample(x, 128, 4, 3, init=init),  # (batch_size, 16, 16, 128)
    x, indices1 = output[0][0], output[0][1]
    output = segnet_downsample(x, 256, 4, 4, init=init),  # (batch_size, 8, 8, 256)
    x, indices2 = output[0][0], output[0][1]
    output = segnet_downsample(x, 512, 4, 4, init=init),  # (batch_size, 4, 4, 512)
    x, indices3 = output[0][0], output[0][1]

    x = segnet_upsample([x, indices3], 256, 4, 4, apply_dropout=True, init=init),  # (batch_size, 8, 8, 256)
    x = segnet_upsample([x, indices2], 128, 4, 4, apply_dropout=True, init=init),  # (batch_size, 16, 16, 128)
    x = segnet_upsample([x, indices1], 64, 4, 3, apply_dropout=True, init=init),  # (batch_size, 32, 32, 64)
    x = segnet_upsample([x, indices0], 32, 4, 3),  # (batch_size, 64, 64, 32)

    last = layers.Conv2D(OUTPUT_CHANNELS, 4, strides=1,
                         padding="same",
                         activation="tanh",
                         kernel_initializer=init)
    # last = layers.Activation("tanh")

    x = last(x[0])

    return tf.keras.Model(inputs=inputs, outputs=x)


def AtrousGenerator():
    pass


def resblock(x, filters, kernel_size, init):
    original_x = x

    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.Add()([original_x, x])

    # the StarGAN official implementation skips this last activation of the resblock
    # https://github.com/yunjey/stargan/blob/master/model.py
    # x = layers.ReLU()(x)
    return x


def StarGANDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    x = source_image

    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    filters = 64
    downsampling_blocks = 5  # 128, 256, 512, 1024, 2048
    for i in range(downsampling_blocks):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = layers.LeakyReLU(0.01)(x)

    # 2x2 patches output (2x2x1)
    patches = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=init, use_bias=False,
                            name="discriminator_patches")(x)

    # domain classifier output (1x1xdomain)
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, kernel_initializer=init,
                                   use_bias=False)(x)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="StarGANDiscriminator")


def StarGANGenerator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS + NUMBER_OF_DOMAINS])
    x = source_image

    filters = 64
    x = layers.Conv2D(filters, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
    x = layers.ReLU()(x)

    # downsampling blocks: 128, then 256
    for i in range(2):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    # bottleneck blocks
    for i in range(6):
        x = resblock(x, filters, 3, init)

    # upsampling blocks: 128, then 64
    for i in range(2):
        filters /= 2
        x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init,
                                   use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    x = layers.Conv2D(OUTPUT_CHANNELS, kernel_size=7, strides=1, padding="same", kernel_initializer=init,
                      use_bias=False)(x)
    activation = layers.Activation("tanh", name="generated_image")(x)

    return tf.keras.Model(inputs=source_image, outputs=activation, name="StarGANGenerator")


# 2-paired star model
# ===================
#
def TwoPairedStarGANDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")
    x = layers.concatenate([target_image, source_image], axis=-1)

    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    filters = 64
    downsampling_blocks = 5  # 128, 256, 512, 1024, 2048
    for i in range(downsampling_blocks):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = layers.LeakyReLU(0.01)(x)

    # 2x2 patches output (2x2x1)
    patches = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=init, use_bias=False,
                            name="discriminator_patches")(x)

    # domain classifier output (1x1xdomain)
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, kernel_initializer=init,
                                   use_bias=False)(x)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=[patches, classification],
                          name="PairedStarGANDiscriminator")


# def TwoPairedStarGANGenerator():
#     init = tf.random_normal_initializer(0., 0.02)

#     input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS + NUMBER_OF_DOMAINS])
#     x = input_image

#     filters = 64
#     x = layers.Conv2D(filters, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
#     x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
#     x = layers.ReLU()(x)

#     # downsampling blocks: 128, then 256
#     for i in range(2):
#         filters *= 2
#         x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
#         x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
#         x = layers.ReLU()(x)

#     # bottleneck blocks
#     for i in range(6):
#         x = resblock(x, filters, 3, init)

#     # upsampling blocks: 128, then 64
#     for i in range(2):
#         filters /= 2
#         x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
#         x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
#         x = layers.ReLU()(x)

#     x = layers.Conv2D(OUTPUT_CHANNELS, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
#     activation = layers.Activation("tanh", name="generated_image")(x)

#     return tf.keras.Model(inputs=input_image, outputs=activation, name="StarGANGenerator")


def unet_star_downsample(filters, size, initializer):
    return tf.keras.Sequential([
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False),
        tfalayers.InstanceNormalization(epsilon=0.00001),
        layers.LeakyReLU()
    ])


def unet_star_upsample(filters, size, initializer):
    return tf.keras.Sequential([
        layers.Conv2DTranspose(filters, size, strides=2,
                               padding="same",
                               kernel_initializer=initializer,
                               use_bias=False),
        tfalayers.InstanceNormalization(epsilon=0.00001),
        layers.ReLU()
    ])

    return result


def StarGANUnetGenerator():
    initializer = tf.random_normal_initializer(0., 0.02)
    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS + NUMBER_OF_DOMAINS])

    down_stack = [
        unet_star_downsample(64, 4, initializer),  # (batch_size, 32, 32,   64)
        unet_star_downsample(128, 4, initializer),  # (batch_size, 16, 16,  128)
        unet_star_downsample(256, 4, initializer),  # (batch_size,  8,  8,  256)
        unet_star_downsample(512, 4, initializer),  # (batch_size,  4,  4,  512)
        unet_star_downsample(1024, 4, initializer),  # (batch_size,  2,  2,  512)
        unet_star_downsample(1024, 4, initializer),  # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_star_upsample(1024, 4, initializer),  # (batch_size,  2,  2, 1024)
        unet_star_upsample(512, 4, initializer),  # (batch_size,  4,  4, 1024)
        unet_star_upsample(256, 4, initializer),  # (batch_size,  8,  8,  512)
        unet_star_upsample(128, 4, initializer),  # (batch_size, 16, 16,  256)
        unet_star_upsample(64, 4, initializer),  # (batch_size, 32, 32,  128)
        unet_star_upsample(32, 4, initializer),  # (batch_size, 64, 64,   64)
    ]

    last = layers.Conv2D(OUTPUT_CHANNELS, 4,
                         strides=1,
                         padding="same",
                         kernel_initializer=initializer,
                         activation="tanh")  # (batch_size, 64, 64, 4)

    x = source_image

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        # x = resblock(x, 64, 4, initializer)

    # ignora a última skip e inverte a ordem
    skips = list(reversed(skips[:-1]))

    # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, [*skips, source_image]):
        # x = resblock(x, 64, 4, initializer)
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=source_image, outputs=x, name="StarGANUnetGenerator")


def CollaGANDiscriminator():
    # Discriminator extracted from:
    # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/CollaGAN_fExp8.py#L521

    def downsample(block_input, filters):
        # Conv2d2x2 + lReLU function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py
        x = block_input
        x = layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=False, )(x)
        x = layers.LeakyReLU()(x)
        return x

    base_filters = 64

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")

    conv_0______ = downsample(source_image, base_filters * 1)
    conv_1______ = downsample(conv_0______, base_filters * 2)
    conv_2______ = downsample(conv_1______, base_filters * 4)
    conv_3______ = downsample(conv_2______, base_filters * 8)
    conv_4______ = downsample(conv_3______, base_filters * 16)
    conv_last___ = downsample(conv_4______, base_filters * 32)

    conv_last___ = layers.Dropout(0.5)(conv_last___)

    # outputs: patches + classification
    patches = layers.Conv2D(1, 3, strides=1, padding="same", use_bias=False, name="discriminator_patches")(conv_last___)

    downsampling_blocks = 6
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, use_bias=False)(
        conv_last___)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="CollaGANDiscriminator")


def CollaGANGenerator():
    # UnetINDiv4 extracted from:
    # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L941
    def conv_block(block_input, filters, regularizer="l2"):
        # CNR function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L44
        x = block_input
        x = layers.Conv2D(filters, 3, strides=1, padding="same", kernel_regularizer=regularizer)(x)
        x = tfalayers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def downsample(block_input, filters):
        # Pool2d function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L23
        x = layers.Conv2D(filters, 2, strides=2, padding="same", use_bias=False, )(block_input)
        return x

    def upsample__(block_input, filters):
        # Conv2dT function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L29
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(block_input)
        return x

    def conv_1x1__(block_input, filters):
        # Conv1x1 function from (with an additional tanh activation by us):
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L38
        x = layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False, activation="tanh")(block_input)
        return x

    source_image_back = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    source_image_left = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    source_image_front = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    source_image_right = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    target_domain_mask = layers.Input(shape=[IMG_SIZE, IMG_SIZE, NUMBER_OF_DOMAINS])
    inputs = [source_image_back, source_image_left, source_image_front, source_image_right, target_domain_mask]

    # encoder starts here...
    base_filters = 64
    filters_per_pose = base_filters // 4

    # path for 'back'
    back_conv_0_0 = tf.concat([source_image_back, target_domain_mask], axis=-1)
    back_conv_0_1 = conv_block(back_conv_0_0, filters_per_pose * 1)
    back_conv_0_2 = conv_block(back_conv_0_1, filters_per_pose * 1)
    back_down_1__ = downsample(back_conv_0_2, filters_per_pose * 2)
    back_conv_1_1 = conv_block(back_down_1__, filters_per_pose * 2)
    back_conv_1_2 = conv_block(back_conv_1_1, filters_per_pose * 2)
    back_down_2__ = downsample(back_conv_1_2, filters_per_pose * 4)
    back_conv_2_1 = conv_block(back_down_2__, filters_per_pose * 4)
    back_conv_2_2 = conv_block(back_conv_2_1, filters_per_pose * 4)
    back_down_3__ = downsample(back_conv_2_2, filters_per_pose * 8)
    back_conv_3_1 = conv_block(back_down_3__, filters_per_pose * 8)
    back_conv_3_2 = conv_block(back_conv_3_1, filters_per_pose * 8)
    back_down_4__ = downsample(back_conv_3_2, filters_per_pose * 16)

    # path for 'left'
    left_conv_0_0 = tf.concat([source_image_left, target_domain_mask], axis=-1)
    left_conv_0_1 = conv_block(left_conv_0_0, filters_per_pose * 1)
    left_conv_0_2 = conv_block(left_conv_0_1, filters_per_pose * 1)
    left_down_1__ = downsample(left_conv_0_2, filters_per_pose * 2)
    left_conv_1_1 = conv_block(left_down_1__, filters_per_pose * 2)
    left_conv_1_2 = conv_block(left_conv_1_1, filters_per_pose * 2)
    left_down_2__ = downsample(left_conv_1_2, filters_per_pose * 4)
    left_conv_2_1 = conv_block(left_down_2__, filters_per_pose * 4)
    left_conv_2_2 = conv_block(left_conv_2_1, filters_per_pose * 4)
    left_down_3__ = downsample(left_conv_2_2, filters_per_pose * 8)
    left_conv_3_1 = conv_block(left_down_3__, filters_per_pose * 8)
    left_conv_3_2 = conv_block(left_conv_3_1, filters_per_pose * 8)
    left_down_4__ = downsample(left_conv_3_2, filters_per_pose * 16)

    # path for 'front'
    front_conv_0_0 = tf.concat([source_image_front, target_domain_mask], axis=-1)
    front_conv_0_1 = conv_block(front_conv_0_0, filters_per_pose * 1)
    front_conv_0_2 = conv_block(front_conv_0_1, filters_per_pose * 1)
    front_down_1__ = downsample(front_conv_0_2, filters_per_pose * 2)
    front_conv_1_1 = conv_block(front_down_1__, filters_per_pose * 2)
    front_conv_1_2 = conv_block(front_conv_1_1, filters_per_pose * 2)
    front_down_2__ = downsample(front_conv_1_2, filters_per_pose * 4)
    front_conv_2_1 = conv_block(front_down_2__, filters_per_pose * 4)
    front_conv_2_2 = conv_block(front_conv_2_1, filters_per_pose * 4)
    front_down_3__ = downsample(front_conv_2_2, filters_per_pose * 8)
    front_conv_3_1 = conv_block(front_down_3__, filters_per_pose * 8)
    front_conv_3_2 = conv_block(front_conv_3_1, filters_per_pose * 8)
    front_down_4__ = downsample(front_conv_3_2, filters_per_pose * 16)

    # path for 'right'
    right_conv_0_0 = tf.concat([source_image_right, target_domain_mask], axis=-1)
    right_conv_0_1 = conv_block(right_conv_0_0, filters_per_pose * 1)
    right_conv_0_2 = conv_block(right_conv_0_1, filters_per_pose * 1)
    right_down_1__ = downsample(right_conv_0_2, filters_per_pose * 2)
    right_conv_1_1 = conv_block(right_down_1__, filters_per_pose * 2)
    right_conv_1_2 = conv_block(right_conv_1_1, filters_per_pose * 2)
    right_down_2__ = downsample(right_conv_1_2, filters_per_pose * 4)
    right_conv_2_1 = conv_block(right_down_2__, filters_per_pose * 4)
    right_conv_2_2 = conv_block(right_conv_2_1, filters_per_pose * 4)
    right_down_3__ = downsample(right_conv_2_2, filters_per_pose * 8)
    right_conv_3_1 = conv_block(right_down_3__, filters_per_pose * 8)
    right_conv_3_2 = conv_block(right_conv_3_1, filters_per_pose * 8)
    right_down_4__ = downsample(right_conv_3_2, filters_per_pose * 16)

    # decoder starts here...
    concat_down_4__ = tf.concat([back_down_4__, left_down_4__, front_down_4__, right_down_4__], axis=-1)
    concat_conv_4_1 = conv_block(concat_down_4__, filters_per_pose * 16)
    concat_conv_4_2 = conv_block(concat_conv_4_1, filters_per_pose * 16)
    up_4___________ = upsample__(concat_conv_4_2, filters_per_pose * 8)

    concat_down_3_2 = tf.concat([back_conv_3_2, left_conv_3_2, front_conv_3_2, right_conv_3_2], axis=-1)
    concat_skip_3__ = tf.concat([concat_down_3_2, up_4___________], axis=-1)
    up_conv_3_1____ = conv_block(concat_skip_3__, filters_per_pose * 8)
    up_conv_3_2____ = conv_block(up_conv_3_1____, filters_per_pose * 8)
    up_3___________ = upsample__(up_conv_3_2____, filters_per_pose * 4)

    concat_down_2_2 = tf.concat([back_conv_2_2, left_conv_2_2, front_conv_2_2, right_conv_2_2], axis=-1)
    concat_skip_2__ = tf.concat([concat_down_2_2, up_3___________], axis=-1)
    up_conv_2_1____ = conv_block(concat_skip_2__, filters_per_pose * 4)
    up_conv_2_2____ = conv_block(up_conv_2_1____, filters_per_pose * 4)
    up_2___________ = upsample__(up_conv_2_2____, filters_per_pose * 2)

    concat_down_1_2 = tf.concat([back_conv_1_2, left_conv_1_2, front_conv_1_2, right_conv_1_2], axis=-1)
    concat_skip_1__ = tf.concat([concat_down_1_2, up_2___________], axis=-1)
    up_conv_1_1____ = conv_block(concat_skip_1__, filters_per_pose * 2)
    up_conv_1_2____ = conv_block(up_conv_1_1____, filters_per_pose * 2)
    up_1___________ = upsample__(up_conv_1_2____, filters_per_pose * 1)

    concat_down_0_2 = tf.concat([back_conv_0_2, left_conv_0_2, front_conv_0_2, right_conv_0_2], axis=-1)
    concat_skip_0__ = tf.concat([concat_down_0_2, up_1___________], axis=-1)
    up_conv_0_1____ = conv_block(concat_skip_0__, filters_per_pose * 1)
    up_conv_0_2____ = conv_block(up_conv_0_1____, filters_per_pose * 1)

    # added beyond CollaGAN to make pixel values between [-1,1]
    output = conv_1x1__(up_conv_0_2____, OUTPUT_CHANNELS)

    return tf.keras.Model(inputs=inputs, outputs=output, name="CollaGANGenerator")
