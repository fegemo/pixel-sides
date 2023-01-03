import tensorflow as tf
import tensorflowjs as tfjs
import argparse
import io_utils


def load_model(path):
    return tf.keras.models.load_model(path)


def convert_model(model, output_path):
    io_utils.delete_folder(output_path)
    io_utils.ensure_folder_structure(output_path)
    tfjs.converters.save_keras_model(model, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the model to be converted")
    parser.add_argument("output", help="path where to save the output")
    values = parser.parse_args()

    # 1. load the model
    print(f"Loading the model at {values.input}...")
    model = load_model(values.input)

    # 2. write it converted
    print(f"Converting the loaded model to {values.output}...")
    convert_model(model, values.output)

    print("Finished conversion.")


if __name__ == "__main__":
    main()
