import tensorflow as tf
from keras import layers
from keras import Model, Input
from audio_data import *

import matplotlib.pyplot as plt
# audio net
input_audio = Input(shape=(257, 200, 1), name='audio')
audio_conv1_1 = layers.Conv2D(64, (3, 3))(input_audio)
x = layers.BatchNormalization()(audio_conv1_1)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(64, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
audio_pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)


x = layers.Conv2D(128, (3, 3))(audio_pool1)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(128, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
audio_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(256, (3, 3))(audio_pool2)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(256, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
audio_pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3))(audio_pool3)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(512, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
audio_pool4 = layers.MaxPooling2D((32, 25), strides=(2, 2))(x)
audio_out = layers.Flatten()(audio_pool4)

audio_fc1 = layers.Dense(128)(audio_out)
x = layers.Dropout(0.5)(audio_fc1)
x = layers.Activation('relu')(x)
x = layers.Dense(8)(x)
audio_result = layers.Activation('softmax')(x)

audio_model = Model(inputs=input_audio, outputs=audio_result)

audio_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# audio_model.summary()

history = audio_model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Train acc')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
'''
'''