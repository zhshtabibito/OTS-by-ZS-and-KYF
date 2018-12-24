import os
import json
from dataHelper import nussl,separation
import numpy as np
from random import sample


def Evaluate(jsonpath,ResultAudioPath,gtAudioPath):
	# calculate acc
	with open(os.path.join(jsonpath,"result.json"),"r") as f:
		result=json.load(f)
	with open(os.path.join(jsonpath,"gt.json"),"r") as f:
		gt=json.load(f)
	acc=[]
	sdr_list=[]
	for keys in gt:
		file_prefix=keys.split('.')[0]
		gt_signal1=nussl.AudioSignal(path_to_input_file=os.path.join(gtAudioPath,file_prefix+'_gt1.wav'))
		gt_signal2=nussl.AudioSignal(path_to_input_file=os.path.join(gtAudioPath,file_prefix+'_gt2.wav'))
		result_name=[]
		result_label=[]
		for j in range(2):
			result_name.append(result[keys][j]['audio'])
			result_label.append(result[keys][j]['position'])
		result_signal1=nussl.AudioSignal(path_to_input_file=os.path.join(ResultAudioPath,file_prefix+'_seg1.wav'))
		result_signal2=nussl.AudioSignal(path_to_input_file=os.path.join(ResultAudioPath,file_prefix+'_seg2.wav'))
		ori_ref_sources=np.zeros([2,len(gt_signal1.audio_data[0,:])])
		ori_est_sources=np.zeros([2,len(gt_signal1.audio_data[0,:])])
		ori_ref_sources[0,:]=sum(gt_signal1.audio_data)
		ori_ref_sources[1,:]=sum(gt_signal2.audio_data)
		ori_est_sources[0,:]=sum(result_signal1.audio_data)
		ori_est_sources[1,:]=sum(result_signal2.audio_data)
		MaxL=300000
		ref_sources=np.zeros([2,min(len(gt_signal1.audio_data[0,:]),MaxL)])
		est_sources=np.zeros([2,min(len(gt_signal1.audio_data[0,:]),MaxL)])
		if len(gt_signal1.audio_data[0,:])>MaxL:
			ref_sources[0,:]=sample(ori_ref_sources[0,:],MaxL)
			ref_sources[1,:]=sample(ori_ref_sources[1,:],MaxL)
			est_sources[0,:]=sample(ori_est_sources[0,:],MaxL)
			est_sources[1,:]=sample(ori_est_sources[1,:],MaxL)
		else:
			ref_sources=ori_ref_sources
			est_sources=ori_est_sources


		rvalue=separation.bss_eval_sources(ref_sources,est_sources,compute_permutation=True)
		compare_label=rvalue[3]
		if compare_label[0]==0:
			if result_label[0]==0:
				acc.append(1)
			else:
				acc.append(0)
		else:
			if result_label[0]==1:
				acc.append(1)
			else:
				acc.append(0)
		# print acc
		sdr_list.extend(rvalue[0])


	return acc,sdr_list
if __name__ == '__main__':
	jsonpath='result_json'
	ResultAudioPath='result_audio'
	gtAudioPath='gt_audio'
	acc,sdr=Evaluate(jsonpath,ResultAudioPath,gtAudioPath)
	print 'accuracy:',sum(acc)/len(acc)
	print 'mean sdr:',sum(sdr)/len(sdr)