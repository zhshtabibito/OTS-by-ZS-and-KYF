import os
import cv2
import numpy as np
from dataHelper.Utils import *
import keras, json, time, librosa, glob
import librosa.display
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
import decomp.decomp_model as dm
from keras.preprocessing import image

def Test(imagepath, audiopath, OutputJSONPath, OutputAudio, isFromImage=False, VideoPath=''):
    imagelist = os.listdir(imagepath)
    print(imagelist)
    result = {}
    time_list = []
    # audio decomp
    model_sep = dm.Model()
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False
    )

    audio_recorder = {}
    ckpt_path = 'decomp/checkpoints'
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        model_sep.load_state(sess, ckpt_path)
        for file in imagelist:
            file_key = file + '.mp4'
            audioname = os.path.join(audiopath, file) + '.wav'
            print(file_key)
            # print('read from image file')
            audio = load_audio_only(audioname)
            # print audio.shape
            # audio decompostion
            start = time.clock()
            audio = librosa.resample(audio, 44100, 16000)
            mixed_spec = librosa.stft(audio.flatten(), n_fft=1024, hop_length=256)
            mixed_mag = np.abs(mixed_spec)
            mixed_phase = np.angle(mixed_spec)
            mixed_batch = spec_to_batch(np.array([mixed_mag, ]))
            seg1_mag, seg2_mag = sess.run(model_sep(), feed_dict={model_sep.x_mixed: mixed_batch})
            seg1_mag = batch_to_spec(seg1_mag)[0, :, :mixed_phase.shape[-1]]
            seg2_mag = batch_to_spec(seg2_mag)[0, :, :mixed_phase.shape[-1]]
            # mask
            mask_seg1 = soft_time_freq_mask(seg1_mag, seg2_mag)
            mask_seg2 = 1. - mask_seg1
            seg1_mag = mixed_mag * mask_seg1
            seg2_mag = mixed_mag * mask_seg2
            audio_seg1 = librosa.resample(to_wav(seg1_mag, mixed_phase), 16000, 44100)
            audio_seg2 = librosa.resample(to_wav(seg2_mag, mixed_phase), 16000, 44100)
            end = time.clock()
            time_list.append(end - start)
            # write & return
            audio_recorder[file_key] = {}
            outputfile1 = os.path.join(OutputAudio, file + '_seg1.wav')
            outputfile2 = os.path.join(OutputAudio, file + '_seg2.wav')
            audio_recorder[file_key]["audio_name"] = [outputfile1, outputfile2]
            audio_recorder[file_key]["audio_sep"] = [audio_seg1, audio_seg2]
            sf.write(outputfile1, audio_seg1, 44100, format='wav', subtype='PCM_16')
            sf.write(outputfile2, audio_seg2, 44100, format='wav', subtype='PCM_16')

    # audio recog
    sr = 44100
    model = keras.models.load_model('network/audio_md.h5')
    for file in imagelist:
        file_key = file + '.mp4'
        start = time.clock()
        # location code
        print(file_key)
        imgs1 = audio_to_spec(audio_recorder[file_key]["audio_sep"][0])
        imgs2 = audio_to_spec(audio_recorder[file_key]["audio_sep"][1])
        predict1 = model.predict(imgs1, batch_size=4).sum(axis=0).tolist()
        predict2 = model.predict(imgs2, batch_size=4).sum(axis=0).tolist()
        ins1, pos1 = predict1.index(max(predict1)), max(predict1)
        ins2, pos2 = predict2.index(max(predict2)), max(predict2)
        if ins1 != ins2:
            audio_recorder[file_key]["audio_src"] = [ins1, ins2]
        elif pos1 > pos2:
            predict2[ins2]=0
            audio_recorder[file_key]["audio_src"] = [ins1, predict2.index(max(predict2))]
        else:
            predict1[ins1] = 0
            audio_recorder[file_key]["audio_src"] = [predict1.index(max(predict1)), ins2]
        end = time.clock()
        time_list.append(end - start)

    # video recog
    model = keras.models.load_model('network/img_md2_5.h5')
    for file in imagelist:
        file_key = file + '.mp4'
        print(file_key)
        imagename = os.path.join(imagepath, file)
        video = load_image_only(imagename)
        # print video.shape
        start = time.clock()
        # location code
        img0 = np.array(list(map(lambda v: cv2.resize(v[:, 0:112, :], (224, 224)), video)))
        img1 = np.array(list(map(lambda v: cv2.resize(v[:, 112:224, :], (224, 224)), video)))
        predict1 = model.predict(img0, batch_size=4).sum(axis=0).tolist()
        predict2 = model.predict(img1, batch_size=4).sum(axis=0).tolist()
        if predict1[audio_recorder[file_key]["audio_src"][0]] > predict2[audio_recorder[file_key]["audio_src"][0]] and \
                predict1[audio_recorder[file_key]["audio_src"][1]] < predict2[audio_recorder[file_key]["audio_src"][1]]:
            position = [0, 1]
        elif predict1[audio_recorder[file_key]["audio_src"][1]] > predict2[audio_recorder[file_key]["audio_src"][1]] and \
                predict1[audio_recorder[file_key]["audio_src"][0]] < predict2[audio_recorder[file_key]["audio_src"][0]]:
            position = [1, 0]
        else:
            position = [0,1]
        end = time.clock()
        time_list.append(end - start)
        # write result
        result[file_key] = []
        for i in range(2):
            temp = {}
            temp['audio'] = audio_recorder[file_key]["audio_name"][i]
            temp['position'] = position[i]
            result[file_key].append(temp)
    with open(os.path.join(OutputJSONPath, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    print("test time:", sum(time_list))


def soft_time_freq_mask(target_src, remaining_src):
    mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
    return mask


def to_wav(mag, phase, len_hop=256):
    stft_matrix = mag * np.exp(1.j * phase)
    return np.array(librosa.istft(stft_matrix, hop_length=len_hop))


def batch_to_spec(seg_mag):
    batch_size, seq_len, freq = seg_mag.shape
    seg_mag = np.reshape(seg_mag, (1, -1, freq))
    return seg_mag.transpose(0, 2, 1)


def spec_to_batch(mixed_mag):
    # padding
    n_batch, freq, n_frames = mixed_mag.shape
    pad_len = 0
    if n_frames % 4 > 0:
        pad_len = (4 - (n_frames % 4))
    pad_width = ((0, 0), (0, 0), (0, pad_len))
    padded = np.pad(mixed_mag, pad_width=pad_width, mode='constant', constant_values=0)
    return np.reshape(padded.transpose(0, 2, 1), (-1, 4, freq))


def audio_to_spec(audio):
    for m in range(1, audio.shape[0] // 88200 + 1, 3):
        ySlice = audio[(m - 1) * 88200:m * 88200]
        S = librosa.feature.melspectrogram(ySlice, sr=44100, n_mels=128)
        # log_S = librosa.logamplitude(S, ref_power=np.max)
        log_S = librosa.amplitude_to_db(S)
        my_dpi = 100.
        fig = plt.figure(figsize=(287. / my_dpi, 230.5 / my_dpi), dpi=my_dpi)
        librosa.display.specshow(log_S, sr=44100, cmap='gray_r')
        plt.tight_layout()
        # jpg not supported
        # specpath = "F:/视听数据/dataset/audios_spec/solo/" + wavpath.split("/solo/")[1].split(".wav")[0] + ".jpg"
        specpath = 'temp/' + "/%06d" % m + ".png"
        plt.savefig(specpath, bbox_inches="tight", pad_inches=0.0)
        plt.clf()
        plt.close(fig)
    specs = glob.glob('temp/*.png')
    imgs = []
    for spec in specs:
        img = image.load_img(spec, target_size=(257, 200), color_mode='grayscale')
        img = image.img_to_array(img)
        imgs.append(img)
    for spec in specs:
        os.remove(spec)
    imgs = np.array(imgs)
    return imgs


if __name__ == '__main__':
    ImageFilePath = "testimage"
    VideoPath = "testvideo"
    AudioPath = "gt_audio"
    OutputPath = "result_json"
    OutputAudio = "result_audio"
    if not os.path.exists(OutputPath):
        os.mkdir(OutputPath)
    if not os.path.exists(OutputAudio):
        os.mkdir(OutputAudio)
    Test(ImageFilePath, AudioPath, OutputPath, OutputAudio, True, VideoPath)
