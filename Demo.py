import os
import cv2
import numpy as np
from dataHelper.Utils import *
import json
import time
from dataHelper.Audio import audio_decomp
def Test(imagepath,audiopath,OutputJSONPath,OutputAudio,isFromImage=False,VideoPath=''):

	imagelist=os.listdir(imagepath)
	print imagelist
	result={}
	time_list=[]
	for file in imagelist:
		file_key=file+'.mp4'
		result[file_key]=[]
		imagename=os.path.join(imagepath,file)
		audioname=os.path.join(audiopath,file)+'.wav'
		# print audioname
		# print imagename
		if isFromImage:
			print('read from image file')
			video,audio=load_data_from_image_file(imagename,audioname)
		else:
			print('read form video')
			Videoname=os.path.join(VideoPath,file_key)
			video,audio=load_data_from_video(Videoname,audioname,10)

		# print video.shape
		# print audio.shape
		# audio decompostion
		start = time.clock()
		audio_name,audio_sep=audio_decomp(video,audio,OutputAudio,file)
		# location code
		# print audio_sep[1].shape
		# print audio.shape
		position= audio_video_sim2(audio_sep[0],audio_sep[1],video) 
		end = time.clock()
		time_list.append(end-start)
		for i in range(len(audio_name)):
			temp={}
			temp['audio']=audio_name[i]
			temp['position']=position[i]
			result[file_key].append(temp)
	with open(os.path.join(OutputPath,"result.json"),"w") as f:
		json.dump(result,f,indent=4)
	print("test time:",sum(time_list))

if __name__ == '__main__':
	ImageFilePath="testimage"
	VideoPath="testvideo"
	AudioPath="gt_audio"
	OutputPath="result_json"
	OutputAudio="result_audio"
	if not os.path.exists(OutputPath):
		os.mkdir(OutputPath)
	if not os.path.exists(OutputAudio):
		os.mkdir(OutputAudio)
	Test(ImageFilePath,AudioPath,OutputPath,OutputAudio,True,VideoPath)