import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
import wave
import struct
import xml.dom.minidom
import random
# from dataHelper.Utils import read_video


def comb_duet(label1, label2, no1, no2, path):
    # only horizontal cropping, crop1 and crop2 are like [x1,w1]
    savename = label1 + '_' + str(no1) + '_' + label2 + '_' + str(no2)
    if os.path.exists(path + savename) == False:
        os.mkdir(path + savename)
    '''
    #if the videos are read, do not read them again
    read_video('dataset/videos/solo/'+label1,'dataset/audios/solo/'+label1,'dataset/images/solo/'+label1,1,no1)
    read_video('dataset/videos/solo/'+label2,'dataset/audios/solo/'+label2,'dataset/images/solo/'+label2,1,no2)
    '''
    f1 = wave.open(u'F:/shiting/dataset/audios/cut/' + label1 + '/' + str(no1) + '.wav')
    params1 = f1.getparams()
    nchannels, sampwidth, framerate, nframes1 = params1[:4]
    strdata = f1.readframes(nframes1)
    wavedata = np.fromstring(strdata, dtype=np.int16)
    wavedata1 = wavedata * 1.0 / (max(abs(wavedata)))
    f2 = wave.open(u'F:/shiting/dataset/audios/cut/' + label2 + '/' + str(no2) + '.wav')
    params2 = f2.getparams()
    nchannels, sampwidth, framerate, nframes2 = params2[:4]
    strdata = f2.readframes(nframes2)
    wavedata = np.fromstring(strdata, dtype=np.int16)
    wavedata2 = wavedata * 1.0 / (max(abs(wavedata)))
    time = min(nframes1, nframes2) / framerate - 1
    print(time)
    wavedata1 = wavedata1[:int(time * framerate)]
    wavedata2 = wavedata2[:int(time * framerate)]
    wavedata4 = wavedata1 / max(max(abs(wavedata1 + wavedata2)), 1)
    wavedata5 = wavedata2 / max(max(abs(wavedata1 + wavedata2)), 1)
    outwave = wave.open(path + savename + '/1.wav', 'wb')
    outwave.setparams((nchannels, sampwidth, framerate, time * framerate, 'NONE', 'not compressed'))
    for v in wavedata4:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))
    outwave.close()
    outwave = wave.open(path + savename + '/2.wav', 'wb')
    outwave.setparams((nchannels, sampwidth, framerate, time * framerate, 'NONE', 'not compressed'))
    for v in wavedata5:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))
    outwave.close()
    wavedata3 = (wavedata1 + wavedata2) / max(max(abs(wavedata1 + wavedata2)), 1)
    outwave = wave.open(path + savename + '/comb.wav', 'wb')
    outwave.setparams((nchannels, sampwidth, framerate, time * framerate, 'NONE', 'not compressed'))
    for v in wavedata3:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))
    outwave.close()


def main():
    audioPath = u'F:\\shiting\\dataset\\audios\\cut\\'
    pathTrain = u'F:\\shiting\\dataset\\audios\\mycomp_py_train\\'
    pathTest = u'F:\\shiting\\dataset\\audios\\mycomp_py_test\\'
    if os.path.exists(pathTrain) == False:
        os.mkdir(pathTrain)
    if os.path.exists(pathTest) == False:
        os.mkdir(pathTest)
    inss = ['accordion', 'acoustic_guitar', 'cello', 'flute',
            'saxophone', 'trumpet', 'violin', 'xylophone']
    global_cnt = 1
    for a in range(7):
        in1 = inss[a]
        aNum = len(glob.glob(audioPath + in1 + '\\*.wav'))
        for b in range(a + 1, 8):
            in2 = inss[b]
            bNum = len(glob.glob(audioPath + in2 + '\\*.wav'))
            for cnt in range(72):
                x = random.randint(1, aNum - int(0.2*aNum))
                y = random.randint(1, bNum - int(0.2*bNum))
                comb_duet(in1, in2, x, y, pathTrain)
                print(global_cnt)
                global_cnt+=1
            for cnt in range(2):
                x = random.randint(aNum-int(0.2 * aNum)+1, aNum)
                y = random.randint(bNum-int(0.2 * aNum)+1, bNum)
                comb_duet(in1, in2, x, y, pathTest)

if __name__ == '__main__':
    main()
