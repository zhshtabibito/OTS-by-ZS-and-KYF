from keras import layers, Input, Model, optimizers

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


# image_model = Model(inputs=input_image, outputs=image_result)
# image_model.summary()
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


# audio_model = Model(inputs=input_audio, outputs=audio_result)
# audio_model.summary()


avc_concat = layers.concatenate(inputs=[image_out, audio_out])
avc_fc1 = layers.Dense(128)(avc_concat)
avc_relu = layers.Activation('relu')(avc_fc1)
avc_fc2 = layers.Dense(2)(avc_relu)
avc_result = layers.Activation('softmax')(avc_fc2)

avc_model = Model(inputs=[input_image, input_audio], outputs=avc_result)

avc_model.summary()

avc_model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
