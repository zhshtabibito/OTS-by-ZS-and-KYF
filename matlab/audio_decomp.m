function [AudioName,AudioData]=audio_decomp(Video,Audio,OutputAudio,FileName)
    %���룺
    %     input1:Video  ��ά����,[len,w,h,channle]
    %     input2:Audio  ��ά����,[1,len]
    %     input3:OutputAudio ���Ŀ¼
    %     input4:Filename  ԭʼ�����ļ���
    %�����
    %     Audio_name:�����Ƶ�ļ����б�
    %     Audio_data:�����Ƶ�ļ������б�,���ڷ���֮ǰ������д��OutputAudio�ļ��У����ְ��ո�����������
    %% input your decompositon code here
    
    
    AudioName={[FileName,'_seg1.wav'],[FileName,'_seg2.wav']};
    AudioData={ones(1000,1),ones(1000,1)};% ������Ҫ����ʵ�ʳ������
end
