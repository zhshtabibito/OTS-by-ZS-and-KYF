close all; clear; clc;
seconds = 10;
sr = 16000;
audioPath = 'F:\shiting\dataset\audios\cut\';
destPath1 = 'F:\shiting\dataset\audios\mycomp_ml_train3\';
destPath2 = 'F:\shiting\dataset\audios\mycomp_ml_test3\';
inss = {'accordion','acoustic_guitar','cello','flute',...
    'saxophone','trumpet','violin','xylophone'};
cntGlobal=1;
for a = 1:7
    aPath = [audioPath, inss{a}, '\'];
    aNum = length(dir([aPath '*.wav']));
    for b = (a+1):8
        bPath = [audioPath, inss{b}, '\'];
        bNum = length(dir([bPath '*.wav']));
        for cnt = 1:108
            x=unidrnd(floor(aNum*0.8)); y=unidrnd(floor(bNum*0.8));
            path1 = [aPath, int2str(x), '.wav'];
            path2 = [bPath, int2str(y), '.wav'];
            wav1 = audioread(path1); wav1 = wav1(:,1); 
            % wav1=resample(wav1,sr,44100);
            wav2 = audioread(path2); wav2 = wav2(:,1); 
            % wav2=resample(wav2,sr,44100);
            wav1 = wav1./max(abs(wav1));
            wav2 = wav2./max(abs(wav2));
            wav1 = wav1./max(max(abs(wav1+wav2)),1);
            wav3 = [wav1, wav2];
            audiowrite([destPath1,inss{a},'-',int2str(x),'-',...
                inss{b},'-',int2str(y),'-',int2str(cntGlobal),'.wav'],wav3,sr);
            cntGlobal = cntGlobal+1
        end
        %{
        for cnt = 1:2
            x=floor(aNum*0.8)+unidrnd(floor(aNum*0.2));
            y=floor(bNum*0.8)+unidrnd(floor(bNum*0.2));
            path1 = [aPath, int2str(x), '.wav'];
            path2 = [bPath, int2str(y), '.wav'];
            wav1 = audioread(path1); wav1 = wav1(:,1); 
            wav2 = audioread(path2); wav2 = wav2(:,1); 
            wav1 = wav1./max(abs(wav1));
            wav2 = wav2./max(abs(wav2));
            wav1 = wav1./max(max(abs(wav1+wav2)),1);
            wav3 = [wav1, wav2];
            audiowrite([destPath2,inss{a},'-',int2str(x),'-',...
                inss{b},'-',int2str(y),'-',int2str(cntGlobal),'.wav'],wav3,sr);
            cntGlobal = cntGlobal+1
        end
        %}
    end
end