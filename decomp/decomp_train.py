import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
from decomp.decomp_model import Model
from decomp.decomp_data import Data, ModelConfig


# Train
class TrainConfig:
    FINAL_STEP = 90000
    CKPT_STEP = 500
    NUM_WAVFILE = 1
    SECONDS = 8.192
    CASE = str(ModelConfig.SEQ_LEN) + 'state'
    CKPT_PATH = 'checkpoints/' + CASE
    DATA_PATH = 'dataset/audio_train'
    LR = 0.0001
    RE_TRAIN = True
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )


class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v


def train():
    # Model
    model = Model()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    # 亚当optimizer，和后面识别网络使用的一样
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    with tf.Session(config=TrainConfig.session_conf) as sess:

        sess.run(tf.global_variables_initializer())
        model.load_state(sess, TrainConfig.CKPT_PATH)

        data = Data(TrainConfig.DATA_PATH)

        loss = Diff()
        for step in range(global_step.eval(), TrainConfig.FINAL_STEP):
            mixed_wav, src1_wav, src2_wav, _ = data.next_wavs(TrainConfig.SECONDS, TrainConfig.NUM_WAVFILE)

            mixed_spec = wav_to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)

            src1_spec, src2_spec = wav_to_spectrogram(src1_wav), wav_to_spectrogram(src2_wav)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

            src1_batch, _ = model.spec_to_batch(src1_mag)
            src2_batch, _ = model.spec_to_batch(src2_mag)
            mixed_batch, _ = model.spec_to_batch(mixed_mag)

            l, _ = sess.run([loss_fn, optimizer],
                            feed_dict={model.x_mixed: mixed_batch, model.y_src1: src1_batch,
                                       model.y_src2: src2_batch})

            loss.update(l)
            print('step-{}\td_loss={:2.2f}\tloss={}'.format(step, loss.diff * 100, loss.value))

            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


def wav_to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav)))


def get_magnitude(stft_matrixes):
    return np.abs(stft_matrixes)


if __name__ == '__main__':
    setup_path()
    train()
