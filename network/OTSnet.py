#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import backend as K
from keras import layers, Input, Model


def mysum(x):
    return K.sum(x, axis=-1)


class AVCNet():
    def __init__(self):
        self.model = None
        self.discriminate_model = None
        self.model = self.make_model()
        # self.model.summary()
        # self.discriminate_model.summary()

    def __call__(self, model_option):
        if model_option == "train":
            return self.model
        else:
            return None

    def make_model(self):
        input_image = Input(shape=(224, 224, 3), name='image')
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
        input_audio = Input(shape=(200, 257, 1), name='audio')
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

        audio_pool4 = layers.MaxPooling2D((12, 16))(audio_conv4_2)
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
        avc_sl = layers.Reshape([1])(avc_maxpool)
        avc_result = layers.Dense(1, activation='softmax')(avc_sl)

        _model = Model(inputs=[input_image, input_audio], outputs=avc_result)
        return _model
