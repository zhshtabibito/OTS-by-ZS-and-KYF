#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

wavs = glob.glob('F:/视听数据/dataset/audios/solo/*/*.wav')
wavs = glob.glob('F:/视听数据/dataset/audios/solo/accordion/*.wav')
print(len(wavs))

pbar = tqdm(total=len(wavs))
for wavpath in wavs:
    try:
        y, sr = librosa.load(wavpath)
    except:
        continue
    # sr is 44100, should be 48k in paper
    # y = librosa.resample(y, sr, 48000)
    folderpath = "F:/视听数据/dataset/audios_spec/solo/" + wavpath.split("/solo/")[1].split(".wav")[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    for m in range(1, y.shape[0]//88200+1):
        ySlice = y[(m-1)*88200:m*88200]
        S = librosa.feature.melspectrogram(ySlice, sr=sr, n_mels=128)
        # log_S = librosa.logamplitude(S, ref_power=np.max)
        log_S = librosa.amplitude_to_db(S)
        my_dpi = 100.
        fig = plt.figure(figsize=(287./my_dpi, 230.5/my_dpi), dpi=my_dpi)
        librosa.display.specshow(log_S, sr=sr, cmap='gray_r')
        plt.tight_layout()
        # jpg not supported
        # specpath = "F:/视听数据/dataset/audios_spec/solo/" + wavpath.split("/solo/")[1].split(".wav")[0] + ".jpg"
        specpath = folderpath + "/%06d" % m +".png"
        plt.savefig(specpath, bbox_inches="tight", pad_inches=0.0)
        plt.clf()
        plt.close(fig)
    pbar.update(1)
pbar.close()
print("finished.")
