import os
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from skimage.io import imread


# scale an array of images to a new size
def _scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def _calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def _load_directory_of_images(path):
    list_files = os.listdir(path)
    image_list = [imread(os.sep.join([path, filename])) for filename in list_files]
    return asarray(image_list)


def _compare_datasets(dataset1_path, dataset2_path, model):
    # loads the images from directories
    images1 = dataset1_path
    images2 = dataset2_path

    if type(dataset1_path) == str:
        images1 = _load_directory_of_images(dataset1_path)
    if type(dataset2_path) == str:
        images2 = _load_directory_of_images(dataset2_path)

    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')

    # resize images
    images1 = _scale_images(images1, (299, 299, 3))
    images2 = _scale_images(images2, (299, 299, 3))

    # pre-process images according to inception v3 expectations
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    fid = _calculate_fid(model, images1, images2)
    return fid


inception_model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))


def compare(dataset1_or_path, dataset2_or_path):
    return _compare_datasets(dataset1_or_path, dataset2_or_path, inception_model)

# import tensorflow as tf
#
#
# # @tf.function
# def _tf_cov(x):
#     mean_x = tf.math.reduce_mean(x, axis=0, keepdims=True)
#     mx = tf.matmul(tf.transpose(mean_x), mean_x)
#     vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], "float32")
#     cov_xx = vx - mx
#     return cov_xx
#
#
# # @tf.function
# def _scale_images(images, new_shape):
#     return tf.image.resize(images, new_shape, method="nearest")
#
#
# # @tf.function
# def _calculate_fid(images1, images2):
#     # calculate activations
#     act1 = inception_model(images1)
#     act2 = inception_model(images2)
#
#     # calculate mean and covariance statistics
#     mu1, sigma1 = tf.math.reduce_mean(act1, axis=0), _tf_cov(act1)
#     mu2, sigma2 = tf.math.reduce_mean(act2, axis=0), _tf_cov(act2)
#     print("mu1", mu1)
#     print("mu2", mu2)
#     print("sigma1", sigma1.numpy().shape)
#     print("sigma2", sigma2.numpy().shape)
#
#
#     # calculate sum squared difference between means
#     ssdiff = tf.math.reduce_sum((mu1 - mu2) ** 2.0)
#     print("ssdiff", ssdiff)
#
#     # calculate sqrt of product between cov
#     dot = tf.experimental.numpy.dot(sigma1, sigma2)
#
#     covmean = tf.linalg.sqrtm(dot)
#     print("covmean", covmean)
#
#     # check and correct imaginary numbers from sqrt
#     if tf.experimental.numpy.iscomplexobj(covmean):
#         covmean = tf.math.real(covmean)
#         print("was complex... turned to real")
#
#
#     # calculate score
#     fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2.0 * covmean)
#
#     return fid
#
#
# def compare(dataset1, dataset2):
#     # resize images to the size expected by Inception v3
#     images1 = _scale_images(dataset1, [299, 299])
#     images2 = _scale_images(dataset2, [299, 299])
#     print("tf.cardinality(dataset1)", dataset1.numpy().shape)
#     print("tf.cardinality(dataset2)", dataset2.numpy().shape)
#
#     # pre-process images according to inception v3 expectations
#     images1 = preprocess_input(images1)
#     images2 = preprocess_input(images2)
#
#     fid = _calculate_fid(images1, images2)
#     print("final FID", fid)
#     return fid
