#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import os, sys
from PIL import Image
import numpy as np


class AVCDataset():
    def __init__(self, spec_glob, c_mode="rgb"):
        self.spec_paths = glob.glob(spec_glob)
        self.c_mode = c_mode
        self._counter = 0

    def create_batch(self, batch_size):
        if self._counter == 0:
            random.shuffle(self.spec_paths)
        if self._counter + batch_size >= len(self.spec_paths):
            self._counter = 0
            random.shuffle(self.spec_paths)
        batch_spec_paths = self.spec_paths[self._counter: self._counter + batch_size]
        img_batch_arr = []
        spec_batch_arr = []
        label_batch_arr = []
        for spec_path in batch_spec_paths:
            # print(tmp_path)
            try:
                num = int(spec_path.split("\\")[-1].split(".png")[0])
                num = num * 10 - 10 + random.randint(1, 10)

                img_path = "F:\\视听数据\\dataset\\images\\" + \
                           spec_path.split("\\")[4] + '\\' + \
                           spec_path.split("\\")[5] + '\\' + \
                           spec_path.split("\\")[6] + '\\' + \
                           "%06d" % num + ".jpg"
                if not os.path.exists(img_path):
                    # print('img not found', spec_path, img_path)
                    continue
                img = Image.open(img_path)
            except:
                continue
            img_arr = np.asarray(img)
            img_arr = np.resize(img_arr, (224, 224, 3))
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 3)) / 255.
            img_batch_arr.append(img_arr)
            img_batch_arr.append(img_arr)
            # positive gen
            label_batch_arr.append([1., 0.])
            img = Image.open(spec_path)
            img = img.convert('L')
            img_arr = np.asarray(img)
            img_arr = np.resize(img_arr, (200, 257))
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 1)) / 255.
            spec_batch_arr.append(img_arr)
            # negative gen
            label_batch_arr.append([0., 1.])
            while (True):
                tmp_path = random.choice(self.spec_paths)
                if tmp_path.split('\\')[5] != img_path.split('\\')[5]:
                    break
            img = Image.open(tmp_path)
            img = img.convert('L')
            img_arr = np.asarray(img)
            img_arr = np.resize(img_arr, (200, 257))
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 1)) / 255.
            spec_batch_arr.append(img_arr)
        self._counter = self._counter + batch_size
        # print(np.array(img_batch_arr).shape)
        # print(np.array(spec_batch_arr).shape)
        # print(np.array(label_batch_arr).shape)
        return (np.array(img_batch_arr), np.array(spec_batch_arr), np.array(label_batch_arr))
