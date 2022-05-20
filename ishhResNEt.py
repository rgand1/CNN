from keras.datasets import cifar10
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    AveragePooling2D, Flatten, Dense,Add
from tensorflow.keras.models import Model


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, filters: int, kernel_size: int = (3,3)) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1),kernel_initializer='he_uniform',
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,kernel_initializer='he_uniform',
               strides=1,
               filters=filters,
               padding="same")(y)
    out = Add()([x, y])
    out = relu_bn(out)

    return out

def ishhResNet():
    inputs = Input(shape=(32, 32, 3))
    num_filters = 32

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=(3,3),kernel_initializer='he_uniform',
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [3, 3]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            stride = (j == 2)
            t = Conv2D(kernel_size=(3,3),kernel_initializer='he_uniform',
                       strides=(2 if stride else 1),
                       filters=num_filters,
                       padding="same")(t)
            t = relu_bn(t)
        num_filters *= 2
    x = residual_block(t,filters=num_filters/2)
    t = AveragePooling2D(4)(x)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

(trainX, trainY), (testX, testY) = cifar10.load_data()
model = ishhResNet()
#print(model.summary())
epochs = 10
history = model.fit(trainX, trainY, epochs=epochs, batch_size=64, validation_data=(testX, testY), verbose=1)