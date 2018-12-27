from keras import layers
from keras import Input, Model
import tensorflow as tf
from keras.layers import dot
from keras import backend as K
from keras import optimizers


def mysum(x):
    return K.sum(x, axis=-1)


input_image = Input(shape=(224, 224, 3))
image_conv1_1 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(input_image)
image_conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv1_1)
image_pool1 = layers.MaxPooling2D((2, 2))(image_conv1_2)

image_conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_pool1)
image_conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_conv2_1)
image_pool2 = layers.MaxPooling2D((2, 2))(image_conv2_2)

image_conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(image_pool2)
image_conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(image_conv3_1)
image_pool3 = layers.MaxPooling2D((2, 2))(image_conv3_2)

image_conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(image_pool3)
image_conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(image_conv4_1)

image_conv5 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(image_conv4_2)
image_conv6 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(image_conv5)

# audio net
input_audio = Input(shape=(257, 200, 1))
audio_conv1_1 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(input_audio)
audio_conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(audio_conv1_1)
audio_pool1 = layers.MaxPooling2D((2, 2))(audio_conv1_2)

audio_conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(audio_pool1)
audio_conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(audio_conv2_1)
audio_pool2 = layers.MaxPooling2D((2, 2))(audio_conv2_2)

audio_conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(audio_pool2)
audio_conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(audio_conv3_1)
audio_pool3 = layers.MaxPooling2D((2, 2))(audio_conv3_2)

audio_conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(audio_pool3)
audio_conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(audio_conv4_1)

audio_pool4 = layers.MaxPooling2D((16, 12))(audio_conv4_2)
audio_fc1 = layers.Dense(128)(audio_pool4)
audio_fc2 = layers.Dense(128)(audio_fc1)

audio_rsp1 = layers.Reshape([128])(audio_fc2)
audio_rpt = layers.RepeatVector(14 * 14)(audio_rsp1)
audio_rsp2 = layers.Reshape([14, 14, 128])(audio_rpt)

avc_mtp = layers.multiply([image_conv6, audio_rsp2])
avc_sum = layers.Lambda(mysum, name='mysum')(avc_mtp)
avc_rsp = layers.Reshape([14, 14, 1])(avc_sum)

avc_conv7 = layers.Conv2D(1, (1, 1), activation='relu')(avc_rsp)
avc_sgm = layers.Dense(1, activation='sigmoid')(avc_conv7)
avc_maxpool = layers.MaxPooling2D((14, 14))(avc_sgm)
avc_result = layers.Dense(1,activation='sigmoid')(avc_maxpool)
#avc_result = layers.Reshape([1])(avc_maxpool)

avc_model = Model(inputs=[input_image, input_audio], outputs=avc_result)
avc_model.summary()

avc_model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])


'''
avc_apsp = layers.Reshape([14, 14, 1])(pdt)

avc_conv7 = layers.Conv2D(1, (1, 1), activation='relu')(avc_apsp)
avc_sgm = layers.Dense(1, activation='sigmoid')(avc_conv7)
avc_maxpool = layers.MaxPooling2D((14, 14))(avc_sgm)
avc_res = layers.Reshape([1])(avc_maxpool)

image_model = Model(input_image, image_conv6)
audio_model = Model(input_audio, audio_rpt)

avc_model = Model(inputs=[input_image, input_audio], outputs=avc_maxpool)
'''
'''
image_out=image_model.output
audio_out=audio_model.output
# AVC merge
avc_model = models.Sequential()
avc_model.add(layers.Multiply([image_model, audio_model], axes=-1, keepdims=True))
avc_model.add(layers.Conv2D(1, (1, 1), activation='relu', padding='same'))
avc_model.add(layers.Dense(1, activation='sigmoid'))
# 在这里就可以判断哪里有乐器块了
avc_model.add(layers.MaxPooling2D(14, 14))
'''
