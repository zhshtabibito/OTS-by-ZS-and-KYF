function [position]= audio_video_sim2(Audio_1,Audio_2,Video)
    %输入：
    %     input1:Audio_1 输入音频1,数组
    %     input2:Audio_2 输入音频2，数组
    %     input2:Video 输入视频图片，数组
    %输出：
    %    position，音频对应位置，注意匹配，输入音频1对应position[0] 0代表左边，1代表右边
    position=[0,1];
    %% revise code below to implement your location code
    size_pic=size(Video);
    simRatio=zeros(2,size_pic(2),size_pic(3));
    sample1=resample(Audio_1,size_pic(1),length(Audio_1));
%     size(sample1),5000
    sample2=resample(Audio_2,size_pic(1),length(Audio_1));
    channel=1;
    
    for i=1:size_pic(2)
        for j=1:size_pic(3)
           ratio_audio1=relevant(sample1,Video(:,i,j,channel));
           ratio_audio2=relevant(sample2,Video(:,i,j,channel));
           simRatio(:,i,j)=[ratio_audio1,ratio_audio2];
        end
    end
    [~,pos1]=max(simRatio(1,:,:));
    [~,pos2]=max(simRatio(1,:,:));
    if pos1>pos2
        position=[1,0];
    else
        position=[0,1];
    end
end


function sim_ratio=relevant(A,B)
     sim_ratio=dot(A,B)/(norm(A)*norm(B));
end

