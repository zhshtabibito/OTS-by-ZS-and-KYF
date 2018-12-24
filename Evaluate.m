close all
clear all
clc;

addpath('matlab')

gtPath='gt_audio';
resultPath='result_audio';
load('gt.mat')
load('result.mat')

acc=zeros(1,length(result));
sdr=zeros(2,length(result));
for i=1:length(result)
    [gt{i}.filename,',',result{i}.filename]; % need filename match
    wav_gt1=fullfile(gtPath,gt{i}.audio{1});
    wav_gt2=fullfile(gtPath,gt{i}.audio{2});
    wav_r1=fullfile(resultPath,result{i}.audio{1});
    wav_r2=fullfile(resultPath,result{i}.audio{2});
    [SDR,perm] = bss_wrapper(wav_gt1,wav_gt2,wav_r1,wav_r2,300000);
    result_label=result{i}.position;
    sdr(:,1)=SDR;
    if perm(1)==1 % norm order
        if result_label(1)==0
            acc(i)=1;
        else
            acc(i)=0;
        end
    else   % need adjust
        if result_label(1)==1
            acc(i)=1;
        else
            acc(i)=0;
        end
    end
end
disp(['accuracy is:',num2str(sum(acc)/length(acc))])
disp(['SDR is:',num2str(sum(sum(sdr))/(2*length(sdr)))])
