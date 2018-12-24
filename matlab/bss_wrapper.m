function [SDR,perm] = bss_wrapper(gt1_path,gt2_path,result1_path,result2_path,MaxL)
%BSS_WRAPPER 此处显示有关此函数的摘要
%   此处显示详细说明

    wav1=audioread(gt1_path);
    wav2=audioread(gt2_path);
    audio_len=length(wav1);
    if audio_len>MaxL
        wav1=resample(wav1,MaxL,length(wav1));
        wav2=resample(wav2,MaxL,length(wav2));
    end
    ref_source=[wav1,wav2]';
    wav1=audioread(result1_path);
    wav2=audioread(result2_path);
    audio_len=length(wav1);
    if audio_len>MaxL
        wav1=resample(wav1,MaxL,length(wav1));
        wav2=resample(wav2,MaxL,length(wav2));
    end
    se_result=[wav1,wav2]';
    [SDR,~,~,perm]=bss_eval_sources(se_result,ref_source);
    
    
    
%     wav2=audioread(fullfile(gt_path,gt{i}.audio{2}));
%     ref_source=[wav1,wav2]';
% %     wav1=audioread(fullfile(result_path,result{i}.audio{1}));
% %     wav2=audioread(fullfile(result_path,result{i}.audio{2}));
%     wav1=audioread(fullfile(gt_path,gt{i}.audio{1}));
%     wav2=audioread(fullfile(gt_path,gt{i}.audio{2}));
%     se_result=[wav1,wav2]';
%     result_label=result{i}.position;
%     [SDR,~,~,perm]=bss_eval_sources(se_result,ref_source);

end

