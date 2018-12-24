close all;
clear all;
clc;
ImageFilePath='testimage'
DirImage=dir(ImageFilePath);
gt={};
for i=3:length(DirImage)
    filename=DirImage(i).name;
    temp.audio={strcat(filename,'_gt1.wav'),strcat(filename,'_gt2.wav')};
    temp.position=[0,1];
    temp.filename=strcat(filename,'.mp4');
    gt{i-2}=temp;
end

save gt.mat gt
