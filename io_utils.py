import os
import shutil
import io
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import display

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


def plot_to_image(matplotlib_figure, channels=OUTPUT_CHANNELS):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(matplotlib_figure)
    buffer.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buffer.getvalue(), channels=channels)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    

# invoca o gerador e mostra a imagem de entrada, sua imagem objetivo e a imagem gerada
def generate_comparison_input_target_generated(model, examples, save_name=None, step=None, predicted_images=None):
    num_images = len(examples)
    num_columns = 3

    title = ["Input", "Target", "Generated"]
    if step != None:
        title[-1] += f" ({step/1000}k)"

    figure = plt.figure(figsize=(4*num_columns, 4*num_images))

    if predicted_images == None:
        predicted_images = []

    for i, (input_image, target_image) in enumerate(examples):
        if i >= len(predicted_images):
            predicted_images.append(model(input_image, training=True))

        images = [input_image, target_image, predicted_images[i]]
        for j in range(num_columns):
            idx = i*num_columns + j+1
            plt.subplot(num_images, num_columns, idx)
            plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
            plt.imshow(tf.squeeze(images[j]) * 0.5 + 0.5)
            plt.axis("off")
    
    figure.tight_layout()

    if save_name != None:
        plt.savefig(save_name)

    # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
    # only a blank image... hence, let us just display the saved image
    display(figure)
    # plt.show()

    return figure

    

# generates images depicting what the discriminator thinks of a target image and a generated image - 
# how did it find each one's patches as real or fake
def generate_discriminated_image(input_image, target_image, discriminator, generator, invert_discriminator_value=False):
    generated_image = generator(input_image, training=True)

    discriminated_target_image = tf.math.sigmoid(tf.squeeze(discriminator([input_image, target_image], training=True), axis=[0]))
    discriminated_generated_image = tf.math.sigmoid(tf.squeeze(discriminator([input_image, generated_image], training=True), axis=[0]))
    if invert_discriminator_value:
        discriminated_target_image = 1. - discriminated_target_image 
        discriminated_generated_image = 1. - discriminated_generated_image 
    
    # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)}")

    patches = tf.shape(discriminated_target_image).numpy()[0]
    lower_bound_scaling_factor = IMG_SIZE // patches
    # print(f"lower_bound_scaling_factor {lower_bound_scaling_factor}, shape {tf.shape(discriminated_target_image).numpy()}")
    pad_before = (IMG_SIZE - patches*lower_bound_scaling_factor)//2
    pad_after = (IMG_SIZE - patches*lower_bound_scaling_factor) - pad_before
    # print(f"pad_before {pad_before}, pad_after {pad_after}")
    discriminated_target_image = tf.repeat(tf.repeat(discriminated_target_image, lower_bound_scaling_factor, axis=0), lower_bound_scaling_factor, axis=1)
    # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)} - after repeat")
    discriminated_target_image = tf.pad(discriminated_target_image, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
    # print(f"discriminated_target_image.shape {tf.shape(discriminated_target_image)} - after pad")

    discriminated_generated_image = tf.repeat(tf.repeat(discriminated_generated_image, lower_bound_scaling_factor, axis=0), lower_bound_scaling_factor, axis=1)
    discriminated_generated_image = tf.pad(discriminated_generated_image, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

    generated_image = tf.squeeze(generated_image)
    target_image = tf.squeeze(target_image)


    figure = plt.figure(figsize=(8*2, 8*2))
    for i, (image, disc_output, image_label, output_label) in enumerate(zip([target_image, generated_image], [discriminated_target_image, discriminated_generated_image], ["Target", "Generated"], ["Discriminated target", "Discriminated generated"])):
        plt.subplot(2, 2, i*2 + 1)
        plt.title(image_label, fontdict={"fontsize": 28})
        plt.imshow(image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(2, 2, i*2 + 2)
        plt.title(output_label, fontdict={"fontsize": 28})
        plt.imshow(disc_output, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

    
    plt.show()
    # print(discriminated_generated_image)
    
    
# !!!!!!!
# parei de usar pq o report_fid não está mais precisando dos arquivos no file system
def generate_test_images_and_comparisons_OUTTTTTTTT(images, generated_image, number, architecture_name, model_name, steps):
    display_images = [images[0], images[1], generated_image[0]]
    pre_path = os.sep.join([TEMP_FOLDER, "test-dataset-output", architecture_name, model_name])
    delete_folder(pre_path)
    save_paths = [os.sep.join([pre_path, folder]) for folder in ["input", "target", "generated"]]
    for path in save_paths:
        ensure_folder_structure(path)
    
    for i, (image, path) in enumerate(zip(display_images, save_paths)):
        plt.figure(figsize=(8, 8))
        plt.imshow(image * 0.5 + 0.5)
        plt.axis("off")
        plt.savefig(os.sep.join([path, f"{number}.png"]))
        plt.close()
        image_path = os.sep.join([pre_path, f"{number}.png"])
    
    generate_comparison_input_target_generated(None, images, image_path, steps, generated_image)


