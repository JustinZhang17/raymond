import tensorflow as tf
import cv2 as cv
import os

import numpy as np
from matplotlib import pyplot as plt


# True or 1= Has Tumor
# False or 0= No Tumor
DATASET_PATH = os.path.join('data', 'set1', 'harris', 'segmentation')
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

# prevent OOM errors
def gpu_settings() -> None:
    """Sets GPU memory growth to prevent OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# segment images into tumoral or healthy
def img_segmentation(dataset: str, group: str, modality: str) -> None:
    """Segments images into tumoral or health brain images

    Args:
        modality (str): The type of brain scan
    """
    for image in os.listdir(os.path.join('data', dataset, group, modality)):
        if 'True' in image:
            os.rename(os.path.join('data', dataset, group, modality, image),
                      os.path.join('data', dataset, group, modality, 'tumoral', image))
        elif 'False' in image:
            os.rename(os.path.join('data', dataset, group, modality, image),
                      os.path.join('data', dataset, group, modality, 'healthy', image))

# load data from directory as a tensor dataset
def load_data() -> tf.data.Dataset:
    # TODO: Change to allow for all 3 modalities
    # TODO: Experiment with Different batch_sizes
    data = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=32)

    # Scaling Data to speed up ML process
    scaled = data.map(lambda x, y: (x/255, y))

    return scaled

    # DEBUGGING
    # data_iterator_scaled = scaled.as_numpy_iterator()
    # batch_scaled = data_iterator_scaled.next()

    # fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    # for idx, img in enumerate(batch_scaled[0][:4]):
    #     ax[idx].imshow(img.astype(int))
    #     ax[idx].title.set_text(batch_scaled[1][idx])
    # plt.show()

# split data into train, validation, and test sets (70%, 20%, 10%)
def split_data(data: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # TODO: Play around with the split percentages
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train_size = len(data) - val_size - test_size
    print(
        f'Train with {train_size} batches + Validate with {val_size} batches + Test with {test_size} batches = {len(data)}')

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return train, val, test


# NOTE: This is where all the fun is :)
def build_cnn_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()

    # TODO: Experiment with different layers and parameters
    Conv2D = tf.keras.layers.Conv2D
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Flatten = tf.keras.layers.Flatten
    Dense = tf.keras.layers.Dense

    # ----------------- 1st Convolutional Layer -----------------
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling2D())

    # ----------------- 2nd Convolutional Layer -----------------
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    # ----------------- 3rd Convolutional Layer -----------------
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    # ----------------- 4th Convolutional Layer -----------------
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    print(model.summary())
    return model


def train_model(model: tf.keras.models.Sequential, train: tf.data.Dataset, val: tf.data.Dataset) -> tuple[tf.keras.models.Sequential, tf.keras.callbacks.History]:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True) # Used to prevent overfitting

    # TODO: Experiment with different epochs
    hist = model.fit(train, validation_data=val, epochs=30, callbacks=[early_stopping])
    return model, hist


def plot_model_performance(hist: tf.keras.callbacks.History) -> None:
    loss = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    loss.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")

    accuracy = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'],
             color='orange', label='val_accuracy')
    accuracy.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def evaluate_model(model: tf.keras.models.Sequential, test: tf.data.Dataset) -> None:

    # TODO: Try Different Metrics
    pre = tf.keras.metrics.Precision()
    re = tf.keras.metrics.Recall()
    acc = tf.keras.metrics.BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(
        f'Precision: {pre.result()}, Recall: {re.result()}, BinaryAccuracy: {acc.result()}')


def save_model(model: tf.keras.models.Sequential) -> None:
    print(f'Your dataset came from {DATASET_PATH}')
    print('Would you like to save the model? (y/n)')
    while True:
        inp = input()
        if inp == 'y':
            print('What would you like to name the model?')
            model.save(os.path.join('models', f'{input()}.h5'))
            break
        if inp == 'n':
            break


def main():
    # Set GPU memory growth to prevent OOM errors
    gpu_settings()

    data = load_data()
    train, val, test = split_data(data)

    model, hist = train_model(build_cnn_model(), train, val)

    plot_model_performance(hist)

    evaluate_model(model, test)

    save_model(model)

if __name__ == '__main__':
    main()
