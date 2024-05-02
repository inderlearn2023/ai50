import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_dir):
            # process only the files with .ppm extensions
            if filename.endswith(".ppm"):
                filepath = os.path.join(category_dir, filename)
                try:
                    # Open image file using cv2
                    img = cv2.imread(filepath)
                    # Resize image
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    # Convert image to numpy array
                    img_array = np.array(img)
                    # Append image and label to lists
                    images.append(img_array)
                    labels.append(category)
                except Exception as e:
                    print(f"Error loading image: {filepath} - {e}")

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    POOLING_FILTER = 60
    POOL_SIZE = (3,3)
    DROPOUT = 0.2
    ACTIVATION_FUNCTION = "relu"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(POOLING_FILTER, POOL_SIZE,
                            activation=ACTIVATION_FUNCTION, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(DROPOUT),
        # hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 20, activation=ACTIVATION_FUNCTION),
        # hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 10, activation=ACTIVATION_FUNCTION),
        # output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
