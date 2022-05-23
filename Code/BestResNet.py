from keras.datasets import cifar10
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import time

lr = 0.1


class LearningController(Callback):
    def __init__(self, num_epoch=0, learn_minute=0):
        self.num_epoch = num_epoch
        self.learn_second = learn_minute * 60
        if self.learn_second > 0:
            print("Leraning rate is controled by time.")
        elif self.num_epoch > 0:
            print("Leraning rate is controled by epoch.")

    def on_train_begin(self, logs=None):
        if self.learn_second > 0:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.learn_second > 0:
            current_time = time.time()
            if current_time - self.start_time > self.learn_second:
                self.model.stop_training = True
                print("Time is up.")
                return

            if current_time - self.start_time > self.learn_second / 2:
                self.model.optimizer.lr = lr * 0.1
            if current_time - self.start_time > self.learn_second * 3 / 4:
                self.model.optimizer.lr = lr * 0.01

        elif self.num_epoch > 0:
            if epoch > self.num_epoch / 2:
                self.model.optimizer.lr = lr * 0.1
            if epoch > self.num_epoch * 3 / 4:
                self.model.optimizer.lr = lr * 0.01


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = (3, 3)) -> Tensor:
    y = Conv2D(kernel_size=kernel_size, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001),
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001),
               strides=1,
               filters=filters,
               padding="same")(y)
    y = Dropout(0.2)(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def ResNet(epochs, lr):
    opt = SGD(lr=lr, momentum=0.9)  # ,decay=1e-2/epochs)
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=(3, 3), kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001),
               strides=1,
               filters=num_filters,
               padding="same")(t)

    num_blocks_list = [3, 3]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = relu_bn(t)
            t = residual_block(t, downsample=(
                    j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = GlobalAveragePooling2D()(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def summarize_diagnostics(history, model, epOchs):
    epochs = list(range(0, epOchs))

    train_loss = history.history['loss']
    train_acc = history.history['accuracy']

    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    ax[0].plot(epochs, train_loss, color='green', label="Training loss")
    ax[0].plot(epochs, val_loss, color='red', label="Validation loss")

    ax[0].legend()
    ax[0].set(ylabel='Cross Entropy Loss')
    ax[0].grid()

    ax[1].plot(epochs, train_acc, color='green', label="Training accuracy")
    ax[1].plot(epochs, val_acc, color='red', label="Validation accuracy")

    ax[1].legend()
    ax[1].set(xlabel='Epochs', ylabel='Classification Accuracy (%)')
    ax[1].grid()

    # save plot to file
    plt.savefig(model + '_plot.png')
    plt.close()


def load_and_prepare_data():
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    train_norm = trainX.astype('float32')
    test_norm = testX.astype('float32')

    mean_train = np.mean(trainX, axis=(1, 2, 3), keepdims=True)
    std_train = np.std(trainX, axis=(1, 2, 3), keepdims=True)

    mean_test = np.mean(testX, axis=(1, 2, 3), keepdims=True)
    std_test = np.std(testX, axis=(1, 2, 3), keepdims=True)

    trainX = (train_norm - mean_train) / std_train
    testX = (test_norm - mean_test) / std_test

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


trainX, trainY, testX, testY = load_and_prepare_data()
epochs = 100
batch_size = 128
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
checkpoint = ModelCheckpoint(filepath="ResNet-for-CIFAR-10.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
learning_controller = LearningController(epochs)
callbacks = [checkpoint, learning_controller]
model = ResNet(epochs, lr)
# print(model.summary())

# datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
it_train = datagen.flow(trainX, trainY, batch_size=batch_size)
steps = int(trainX.shape[0] / batch_size)
history = model.fit(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks,
                    verbose=1)
summarize_diagnostics(history, 'Best_Model', epochs)
