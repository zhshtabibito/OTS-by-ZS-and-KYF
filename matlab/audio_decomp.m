function [AudioName,AudioData]=audio_decomp(Video,Audio,OutputAudio,FileName)
    %输入：
    %     input1:Video  四维数组,[len,w,h,channle]
    %     input2:Audio  二维数组,[1,len]
    %     input3:OutputAudio 输出目录
    %     input4:Filename  原始数据文件名
    %输出：
    %     Audio_name:输出音频文件名列表
    %     Audio_data:输出音频文件数据列表,并在返回之前将数据写到OutputAudio文件夹，名字按照给定规则命名
    %% input your decompositon code here
    
    
    AudioName={[FileName,'_seg1.wav'],[FileName,'_seg2.wav']};
    AudioData={ones(1000,1),ones(1000,1)};% 长度需要按照实际长度输出
end
