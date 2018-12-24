import numpy as np
import matplotlib.pyplot as plt
import os,sys,cv2
import glob
from pydub import AudioSegment
import wave
import struct
import xml.dom.minidom
import random
sys.path.append("..")
from Datahelper.Utils import read_video

def comb_duet(label1,label2,no1,no2,crop1,crop2):
    #only horizontal cropping, crop1 and crop2 are like [x1,w1]
    savename=label1+'_'+str(no1)+'_'+label2+'_'+str(no2)
    if os.path.exists('comb_duet/'+savename)==False:
        os.mkdir('comb_duet/'+savename)
    if os.path.exists('comb_duet/'+savename+'/images')==False:
        os.mkdir('comb_duet/'+savename+'/images')
    if os.path.exists('comb_duet/'+savename+'/imgay')==False:
        os.mkdir('comb_duet/'+savename+'/imgay')
    '''
    #if the videos are read, do not read them again
    read_video('dataset/videos/solo/'+label1,'dataset/audios/solo/'+label1,'dataset/images/solo/'+label1,1,no1)
    read_video('dataset/videos/solo/'+label2,'dataset/audios/solo/'+label2,'dataset/images/solo/'+label2,1,no2)
    '''
    f1=wave.open('dataset/audios/solo/'+label1+'/'+str(no1)+'.wav')
    params1=f1.getparams()
    nchannels,sampwidth,framerate,nframes1=params1[:4]
    strdata=f1.readframes(nframes1)
    wavedata=np.fromstring(strdata,dtype=np.int16)
    wavedata1=wavedata*1.0/(max(abs(wavedata)))
    f2=wave.open('dataset/audios/solo/'+label2+'/'+str(no2)+'.wav')
    params2=f2.getparams()
    nchannels,sampwidth,framerate,nframes2=params2[:4]
    strdata=f2.readframes(nframes2)
    wavedata=np.fromstring(strdata,dtype=np.int16)
    wavedata2=wavedata*1.0/(max(abs(wavedata)))
    time=min(nframes1,nframes2)/framerate-1
    print time
    wavedata1=wavedata1[:time*framerate]
    wavedata2=wavedata2[:time*framerate]
    wavedata4=wavedata1/max(max(abs(wavedata1+wavedata2)),1)
    wavedata5=wavedata2/max(max(abs(wavedata1+wavedata2)),1)
    outwave=wave.open('comb_duet/'+savename+'/1.wav','wb')
    outwave.setparams((nchannels,sampwidth,framerate,time*framerate,'NONE','not compressed'))
    for v in wavedata4:
        outwave.writeframes(struct.pack('h',int(v*64000/2)))
    outwave.close()
    outwave=wave.open('comb_duet/'+savename+'/2.wav','wb')
    outwave.setparams((nchannels,sampwidth,framerate,time*framerate,'NONE','not compressed'))
    for v in wavedata5:
        outwave.writeframes(struct.pack('h',int(v*64000/2)))
    outwave.close()
    wavedata3=(wavedata1+wavedata2)/max(max(abs(wavedata1+wavedata2)),1)
    outwave=wave.open('comb_duet/'+savename+'/comb.wav','wb')
    outwave.setparams((nchannels,sampwidth,framerate,time*framerate,'NONE','not compressed'))
    for v in wavedata3:
        outwave.writeframes(struct.pack('h',int(v*64000/2)))
    outwave.close()
    imdir1='dataset/images/solo/'+label1+'/'+str(no1)
    imdir2='dataset/images/solo/'+label2+'/'+str(no2)
    imlist1=os.listdir(imdir1)
    imlist2=os.listdir(imdir2)
    len1=len(imlist1)
    len2=len(imlist2)
    im1=cv2.imread(os.path.join(imdir1,'000001.jpg'))
    im2=cv2.imread(os.path.join(imdir2,'000001.jpg'))
    h1,w1,_=im1.shape
    h2,w2,_=im2.shape
    h=max(h1,h2)
    ww1=crop1[1]*h/h1
    ww2=crop2[1]*h/h2
    x1=crop1[0]*h/h1
    x2=crop2[0]*h/h2
    ww=ww1+ww2
    for i in range(1,24*time+1):
        im1=cv2.imread(os.path.join(imdir1,"%06d" % i+'.jpg'))
        im2=cv2.imread(os.path.join(imdir2,"%06d" % i+'.jpg'))
        im=np.zeros([h,ww,3],np.uint8)
        if h1<h:
            im1=cv2.resize(im1,(w1*h/h1,h))
        elif h2<h:
            im2=cv2.resize(im2,(w2*h/h2,h))
        im[:,ww1:ww,:]=im2[:,x2:x2+ww2,:]
        im[:,0:ww1,:]=im1[:,x1:x1+ww1,:]
        savefile='comb_duet/'+savename+'/images/'+"%06d" % i+'.jpg'
        cv2.imwrite(savefile,im)
    cmd_str='ffmpeg -y -r 24 -i comb_duet/'+savename+'/images/%06d.jpg -i comb_duet/'+savename+'/comb.wav -strict -2 comb_duet/'+savename+'/comb.mp4'
    os.system(cmd_str)

def main():
    comb_duet('flute','violin',1,2,[100,200],[50,300])

if __name__=='__main__':
    main()






    
