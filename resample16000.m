close all; clear; clc;
nRes = 6;
audioPath = 'F:\视听数据\testset25\gt_audio\';
destPath = 'F:\视听数据\testset25\gt_audio_16000';

d = dir([audioPath '*.wav']);


for a = 1:length(dir([audioPath '*.wav']))
    path = [audioPath,d(a).name];
    temp = audioread(path);
    temp = resample(temp,16000,44100);
    path = [destPath,d(a).name];
    audiowrite(path,temp,16000);
end
