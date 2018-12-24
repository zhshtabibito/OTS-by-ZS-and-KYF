close all;
clear all;
clc;
addpath('matlab');

% �ļ��ж�ȡͼƬ������Ƶ
ImageFilePath='testimage';
VideoFilePath='testvideo';
AudioPath='gt_audio';
OutputPath='result_json';
OutputAudio='result_audio';
isFromImage= false;

DirImage=dir(ImageFilePath);
NumOfFile=length(DirImage)-2;
ImageCell=cell(3,NumOfFile);
new_h=224;
new_w=224;
channel=3;
for i=3:NumOfFile+2
    if isFromImage
        ImagePath=fullfile(ImageFilePath,DirImage(i).name);
        ImageCell{1,i-2}=ReadDataFromImage(ImagePath,new_w,new_h);
    else
        VideoPath=fullfile(VideoFilePath,DirImage(i).name);
        ImageCell{1,i-2}=ReadDataFromVideo(VideoPath,new_w,new_h,10);
    end
    
    AudioName=strcat(DirImage(i).name,'.wav');
    [SampleData,FS] = audioread(fullfile(AudioPath,AudioName));
    ImageCell{2,i-2}=SampleData;
    ImageCell{3,i-2}=DirImage(i).name;
end

AudioName='';
AudioData=0;
position=0;
result={};
timeLlist=0;
for i=1:NumOfFile
    t1=clock;
    [AudioName,AudioData]=audio_decomp(ImageCell{1,i},ImageCell{2,i},OutputAudio,ImageCell{3,i});
    position= audio_video_sim2(AudioData{1},AudioData{2},ImageCell{1,i});
    t2=clock;
    timeLlist(i)=etime(t2,t1);
    temp.audio=AudioName;
    temp.position=position;
    temp.filename=strcat(ImageCell{3,i},'.mp4');
    result{i}=temp;
end
disp(['����ʱ�䣺',num2str(sum(timeLlist)),'s'])
save  result.mat result



