function [Video] = ReadDataFromVideo(VideoPath,w,h,frequency)
    VideoName=strcat(VideoPath,'.mp4');
    obj = VideoReader(VideoName);
    numFrames = obj.NumberOfFrames;
    Video=zeros(floor(numFrames/frequency),w,h,3);
    pos=1;
    for i = 1:frequency:numFrames
        Frame = read(obj,i);
        Image = imresize(Frame,[h,w]);
        Video(pos,:,:,:) = Image;
        pos=pos+1;
    end
end

