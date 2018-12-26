from keras import layers
from keras import Input, Model
import tensorflow as tf

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

image_conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_conv4_2)
image_conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_conv5)

#audio net
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

audio_model = Model(input_audio,audio_fc2)


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
