#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
from keras import optimizers
from L3net import L3Net
from dataset import AVCDataset

if __name__ == '__main__':
    L3_model = L3Net()("train")
    dataset = AVCDataset("F:\\视听数据\\dataset\\audios_spec\\solo\\*\\*\\*.png")
    # L3_model.summary()
    L3_model.compile(optimizer=optimizers.Adam(lr=1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    batch_size = 1000
    for n_epoch in range(20):
        print('epoch: ', n_epoch)
        for n_batch in range(10887//batch_size):
            print('batch: ', n_batch)
            batch = dataset.create_batch(batch_size)
            L3_model.fit({'image': batch[0], 'audio': batch[1]}, batch[2], validation_split=0.1, epochs=1, batch_size=2)
        L3_model.save('avc1230_' + str(n_epoch) + '.h5')
    # L3_model.save('temp.h5')