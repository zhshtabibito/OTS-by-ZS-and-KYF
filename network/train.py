#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
from keras import optimizers
from model import AVCNet
from dataset import AVCDataset

if __name__ == '__main__':
    avc_model = AVCNet()("train")
    # dataset = AVCDataset("F:/视听数据/dataset/audios_spec/solo/*/*/*.png")
    dataset = AVCDataset("F:/视听数据/dataset/audios_spec/solo/accordion/1/*.png")
    batch = dataset.create_batch(128)
    avc_model.summary()
    avc_model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
    avc_model.fit({'image': batch[0], 'audio': batch[1]}, batch[2], epochs=10, batch_size=16)
