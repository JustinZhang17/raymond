import tensorflow as tf
import cv2 as cv
import os

# 0 = Giloma
# 1 = Meningioma
# 2 = Healthy
# 3 = Pituitary

import numpy as np
from matplotlib import pyplot as plt

# Define Constants
DATASET_PATH = os.path.join('data', 'set2', 'kMeans')
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def gpu_settings() -> None:
    """Sets GPU memory growth to prevent OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def load_data() -> tf.data.Dataset:
    data = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=32, label_mode='categorical')

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

def split_data(data: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # split data into train, validation, and test sets (70%, 20%, 10%)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train_size = len(data) - val_size - test_size
    print(
        f'Train with {train_size} batches + Validate with {val_size} batches + Test with {test_size} batches = {len(data)}')

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return train, val, test

def build_cnn_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential()

     # TODO: Experiment with different layers and parameters
    Conv2D = tf.keras.layers.Conv2D
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Flatten = tf.keras.layers.Flatten
    Dense = tf.keras.layers.Dense

    # ----------------- 1st Convolutional Layer -----------------
    model.add(Conv2D(64, (3, 3), 1, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
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
    model.add(Dense(4, activation='softmax'))

    # compile model with adam optimizer and categorical crossentropy loss
    model.compile('adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    print(model.summary())
    return model

def train_model(model: tf.keras.models.Sequential, train: tf.data.Dataset, val: tf.data.Dataset) -> tuple[tf.keras.models.Sequential, tf.keras.callbacks.History]:
    # early stopping to prevent overfitting and save best model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    hist = model.fit(train, validation_data=val, epochs=25, callbacks=[early_stop])
    return model, hist

def plot_model_performance(hist: tf.keras.callbacks.History) -> None:
    # plot performance with validation loss and training loss
    loss = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    loss.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")

    # plot performance with validation accuracy and training accuracy
    accuracy = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'],
             color='orange', label='val_accuracy')
    accuracy.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

def evaluate_model(model: tf.keras.models.Sequential, test: tf.data.Dataset) -> None:
    pre = tf.keras.metrics.Precision()
    re = tf.keras.metrics.Recall()
    acc = tf.keras.metrics.CategoricalAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(
        f'Precision: {pre.result()}, Recall: {re.result()}, CategoricalAccuracy: {acc.result()}')

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
    gpu_settings()
    data = load_data()
    train, val, test = split_data(data)

    model, hist = train_model(build_cnn_model(), train, val)

    plot_model_performance(hist)
    
    evaluate_model(model, test)

    save_model(model)

    

if __name__ == "__main__":
    main()
