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
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())

    return result


def unet_upsample(filters, size, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                                padding="same",
                                kernel_initializer=init,
                                use_bias=False))

    result.add(layers.BatchNormalization())

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

    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    x = layers.concatenate([input_image, target_image])  # (batch_size, 64, 64, channels*2)

    if num_patches > 30:
        raise ValueError(f"num_patches for the discriminator should not exceed 30, but {num_patches} was given")

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    current_patches = 30
    next_layer_filters = 64*2
    while current_patches > max(num_patches, 2):
        # print(f"Created downsample with {next_layer_filters}")
        down = unet_downsample(next_layer_filters, 4)(down)
        current_patches = ((current_patches+2) / 2) - 2
        next_layer_filters *= 2
        # print(f"current_patches now {current_patches}")
    zero_pad2 = layers.ZeroPadding2D()(down)  # (batch_size, 33, 33, 128)
    last_filter_size = 4 if num_patches > 1 else 5
    last = layers.Conv2D(1, last_filter_size, strides=1,
                            kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[input_image, target_image], outputs=last, name="patch-disc")
    

def Deeper2x2PatchDiscriminator(**kwargs):
    initializer = tf.random_normal_initializer(0., 0.02)

    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    x = layers.concatenate([input_image, target_image])  # (batch_size, 64, 64, channels*2)


    x = layers.Conv2D(64, 4, 1, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 4, 1, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(256, 4, 1, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    # x = layers.Conv2D(64, 4, 1, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    
    last = layers.Conv2D(1, 4, 1, kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=[input_image, target_image], outputs=last)



def UnetDiscriminator(**kwargs):
    init = tf.random_normal_initializer(0., 0.02)

    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([input_image, target_image])        # (batch_size, 64, 64,    8)
    
    down_stack = [
        unet_downsample( 64, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),                         # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),                         # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),                         # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  4,  4, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  8,  8, 1024)
        unet_upsample(512, 4, init=init),                           # (batch_size, 16, 16, 1024)
        unet_upsample(256, 4, init=init),                           # (batch_size, 32, 32,  512)
    ]

    last = layers.Conv2D(1, 3, strides=1, kernel_initializer=init)  # (batch_size, 30, 30,    1)

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

    return tf.keras.Model(inputs=[input_image, target_image], outputs=x)







from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


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

    
    
def segnet_upsample(inputs, filters, size, convolution_steps=2, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
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
    
    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([input_image, target_image])  # (batch_size, 64, 64, channels*2)

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

    last = layers.Conv2D(1, 3, strides=1,
                            padding="valid",
                            kernel_initializer=initializer
                        )



    x = last(x[0])

    return tf.keras.Model(inputs=[input_image, target_image], outputs=x, name="segnet-disc")
    

def AtrousDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)
    
    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([input_image, target_image])  # (batch_size, 64, 64, channels*2)
    
    x = layers.Conv2D(128, 4, strides=1, padding="same", activation="relu", kernel_initializer=init)(inputs)
    x = layers.Conv2D(256, 4, strides=1, padding="same", activation="relu", kernel_initializer=init)(x)
    x = layers.Conv2D(512, 4, strides=2, padding="same", activation="relu", kernel_initializer=init)(x)
    x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu", kernel_initializer=init)(x)
    skip = x
    
    x = layers.Conv2D(512, 3, strides=1, dilation_rate=2, padding="same", activation="relu", kernel_initializer=init)(x)
    x = layers.Conv2D(512, 3, strides=1, dilation_rate=4, padding="same", activation="relu", kernel_initializer=init)(x)
    x = layers.Conv2D(512, 3, strides=1, dilation_rate=8, padding="same", activation="relu", kernel_initializer=init)(x)
    
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu", kernel_initializer=init)(x)
    x = layers.Conv2D(  1, 3, strides=1, padding="valid", kernel_initializer=init)(x)
    
    return tf.keras.Model(inputs=[input_image, target_image], outputs=x, name="atrous-disc")


def AtrousDiscriminator_out():
    init = tf.random_normal_initializer(0., 0.02)
    
    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    inputs = layers.concatenate([input_image, target_image])  # (batch_size, 64, 64, channels*2)
    
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

    return tf.keras.Model(inputs=[input_image, target_image], outputs=x, name="atrous-disc")



def UnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS]) #(batch_size, 64, 64, 4)

    down_stack = [
        unet_downsample( 64, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),                         # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),                         # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),                         # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  4,  4, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),       # (batch_size,  8,  8,  512)
        unet_upsample(128, 4, init=init),                           # (batch_size, 16, 16,  256)
        unet_upsample( 64, 4, init=init),                           # (batch_size, 32, 32,  128)
        unet_upsample( 32, 4, init=init),                           # (batch_size, 64, 64,   64)
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

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet_conv_block(inputs, filters, kernel_size, initializer, apply_batchnorm=True, apply_dropout=False, activation="leaky", bias=False):
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
    x = layers.Conv2D(filters, kernel_size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    if apply_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    output = x
    
    return output, skip_activations 


def unet_decoder_block(inputs, features_from_skip, filters, kernel_size, initializer, apply_dropout=False):   
    x = inputs
    
    # upsampling
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.ReLU()(x)

    # "same" conv2d
    x = layers.Concatenate()([x, features_from_skip])
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=False, apply_dropout=apply_dropout, activation="relu")
    x = unet_conv_block(x, filters, kernel_size, initializer, apply_batchnorm=False, apply_dropout=apply_dropout, activation="relu")
    
    return x


def Unet2Generator():
    initializer = tf.random_normal_initializer(0., 0.002)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    e0 = inputs
    
    e1, s1 = unet_encoder_block(e0,  32, 4, initializer, False)   # (None, 32, 32,   32)
    e2, s2 = unet_encoder_block(e1,  64, 4, initializer)          # (None, 16, 16,   64)
    e3, s3 = unet_encoder_block(e2, 128, 4, initializer)          # (None,  8,  8,  128)
    e4, s4 = unet_encoder_block(e3, 256, 4, initializer)          # (None,  4,  4,  256)
#     e5, s5 = unet_encoder_block(e4, 512, 4, initializer)          # (None,  2,  2,  512)
#     e6, s6 = unet_encoder_block(e5, 512, 4, initializer)          # (None,  1,  1,  512)
    
#     d6 = unet_decoder_block(e6, s6, 512, 4, initializer, True)    # (None,  2,  2, 1024)
#     d5 = unet_decoder_block(d6, s5, 512, 4, initializer, True)    # (None,  4,  4, 1024)
    d4 = unet_decoder_block(e4, s4, 256, 4, initializer, True)    # (None,  8,  8,  512)
    d3 = unet_decoder_block(d4, s3, 128, 4, initializer)          # (None, 16, 16,  256)
    d2 = unet_decoder_block(d3, s2,  64, 4, initializer)          # (None, 32, 32,  128)
    d1 = unet_decoder_block(d2, s1,  32, 4, initializer)          # (None, 64, 64,   64)

    output = unet_conv_block(d1, OUTPUT_CHANNELS, 4, initializer,
                             apply_batchnorm=False,
                             activation="tanh", bias=True)        # (None, 64, 64,    4)
    
    return tf.keras.Model(inputs=inputs, outputs=output)


def SegnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS]) #(batch_size, 64, 64, 4)

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
    x = segnet_upsample([x, indices1],  64, 4, 3, apply_dropout=True, init=init),  # (batch_size, 32, 32, 64)
    x = segnet_upsample([x, indices0],  32, 4, 3),  # (batch_size, 64, 64, 32)

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
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.Add()([original_x, x])
    
    # the StarGAN official implementation skips this last activation of the resblock
    # https://github.com/yunjey/stargan/blob/master/model.py
    # x = layers.ReLU()(x)
    return x





def StarGANDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)
    
    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    x = input_image
    
    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    filters = 64
    downsampling_blocks = 5 # 128, 256, 512, 1024, 2048
    for i in range(downsampling_blocks):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = layers.LeakyReLU(0.01)(x)
    
    # 2x2 patches output (2x2x1)
    patches = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=init, use_bias=False, name="discriminator_patches")(x)
    
    # domain classifier output (1x1xdomain)
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, kernel_initializer=init, use_bias=False)(x)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)
    
    return tf.keras.Model(inputs=input_image, outputs=[patches, classification], name="StarGANDiscriminator")


def StarGANGenerator():
    init = tf.random_normal_initializer(0., 0.02)

    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS + NUMBER_OF_DOMAINS])
    x = input_image
    
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
        x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)
    
    x = layers.Conv2D(OUTPUT_CHANNELS, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    activation = layers.Activation("tanh", name="generated_image")(x)

    return tf.keras.Model(inputs=input_image, outputs=activation, name="StarGANGenerator")



# 2-paired star model
# ===================
#
def TwoPairedStarGANDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)
    
    input_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="input_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")
    x = inputs = layers.concatenate([input_image, target_image], axis=-1)
    
    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    filters = 64
    downsampling_blocks = 5 # 128, 256, 512, 1024, 2048
    for i in range(downsampling_blocks):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = layers.LeakyReLU(0.01)(x)
    
    # 2x2 patches output (2x2x1)
    patches = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=init, use_bias=False, name="discriminator_patches")(x)
    
    # domain classifier output (1x1xdomain)
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, kernel_initializer=init, use_bias=False)(x)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)
    
    return tf.keras.Model(inputs=[input_image, target_image], outputs=[patches, classification], name="PairedStarGANDiscriminator")


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

    