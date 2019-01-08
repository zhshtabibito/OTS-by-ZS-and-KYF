
from keras import layers
from keras import Model, Input
from image_data import *
import tensorflow as tf
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

import matplotlib.pyplot as plt
# audio net
input_image = Input(shape=(224, 224, 3), name='image')
image_conv1_1 = layers.Conv2D(64, (3, 3))(input_image)
x = layers.BatchNormalization()(image_conv1_1)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(64, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
image_pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)


x = layers.Conv2D(128, (3, 3))(image_pool1)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(128, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
image_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(256, (3, 3))(image_pool2)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(256, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
image_pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3))(image_pool3)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(512, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D((1, 1))(x)
image_pool4 = layers.MaxPooling2D((28, 28), strides=(2, 2))(x)
image_out = layers.Flatten()(image_pool4)

audio_fc1 = layers.Dense(128)(image_out)
x = layers.Dropout(0.5)(audio_fc1)
x = layers.Activation('relu')(x)
x = layers.Dense(8)(x)
image_result = layers.Activation('softmax')(x)

image_model = Model(inputs=input_image, outputs=image_result)

image_model.compile(optimizer=optimizers.Adam(lr=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# image_model.summary()

history = image_model.fit_generator(
    train_generator,
    steps_per_epoch=12625,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=590
)
image_model.save('img_md2_4.h5')

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
'''
'''