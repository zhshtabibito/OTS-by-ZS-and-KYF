close all; clear; clc;
seconds = 10;
sr = 16000;
audioPath = 'F:\shiting\dataset\audios\solo\';
destPath = 'F:\shiting\dataset\audios\cut\';
inss = {'accordion','acoustic_guitar','cello','flute',...
    'saxophone','trumpet','violin','xylophone'};
for a = 1:8
    cntGlobal=1;
    aPath = [audioPath, inss{a}, '\'];
    aNum = length(dir([aPath '*.wav']));
    for x = 1:aNum
        path1 = [aPath, int2str(x), '.wav'];
        wav1 = audioread(path1); wav1 = wav1(:,1); 
        wav1=resample(wav1,sr,44100);
        for y = 1:floor(length(wav1)/(10*sr))
            dest = [destPath,inss{a},'\',int2str(cntGlobal),'.wav'];
            wav2 = wav1((y-1)*10*sr+1:y*10*sr);
            wav2 = wav2./max(max(abs(wav2)),1);
            audiowrite(dest, wav2, sr);
            cntGlobal = cntGlobal+1;
            [a,x, cntGlobal]
        end
    end
end