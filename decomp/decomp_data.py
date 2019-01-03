import random
import os
import librosa
import numpy as np


class Data:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, sec, size=1):
        wavfiles = []
        for (root, dirs, files) in os.walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, size)
        mixed, src1, src2 = get_wav(wavfiles, sec, ModelConfig.SR)
        return mixed, src1, src2, wavfiles


def least_dis(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                pwr = 2 ** i
                break
        if abs(pwr - target) < abs(pwr / 2 - target):
            return pwr
        else:
            return int(pwr / 2)
    else:
        return 1


# Model
class ModelConfig:
    SEQ_LEN = 4
    N_MELS = 512
    F_MIN = 0.0
    SR = 16000
    L_FRAME = 1024
    L_HOP = least_dis(L_FRAME / 4)


def get_wav(filenames, sec, sr=ModelConfig.SR):
    src1_src2 = np.array(list(
        map(lambda f: sample_range(pad_wav(librosa.load(f, sr=sr, mono=False)[0], sr, sec), sr, sec), filenames)))
    mixed = np.array(list(map(lambda f: librosa.to_mono(f), src1_src2)))
    src1, src2 = src1_src2[:, 0], src1_src2[:, 1]
    return mixed, src1, src2


def sample_range(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = int(sr * duration)
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def pad_wav(wav, sr, duration):
    assert (wav.ndim <= 2)

    n_samples = int(sr * duration)
    pad_len = np.maximum(0, n_samples - wav.shape[-1])
    if wav.ndim == 1:
        pad_width = (0, pad_len)
    else:
        pad_width = ((0, 0), (0, pad_len))
    wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)
    return wav
